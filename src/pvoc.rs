extern crate apodize;
extern crate rustfft;
#[cfg(test)]
extern crate rand;

use rustfft::num_complex::Complex;
use rustfft::num_traits::{Float, FromPrimitive, ToPrimitive};
use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::Arc;

const ENV_BUFFER_SIZE: usize = 8;

#[allow(non_camel_case_types)]
type c64 = Complex<f64>;

/// Represents a component of the spectrum, composed of a frequency and amplitude.
#[derive(Copy, Clone)]
pub struct Bin {
    pub freq: f64,
    pub amp: f64,
}

impl Bin {
    pub fn new(freq: f64, amp: f64) -> Bin {
        Bin {
            freq: freq,
            amp: amp,
        }
    }
    pub fn empty() -> Bin {
        Bin {
            freq: 0.0,
            amp: 0.0,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Params {
    pub threshold: f64,
    pub detune: f64,
    pub attack: f64,
    pub release: f64,
    pub mix: f64,
    pub gain: f64,
}

impl Params {
    pub fn new(threshold: f64, detune: f64, attack: f64, release: f64, mix: f64, gain: f64) -> Params {
        Params {
            threshold,
            detune,
            attack,
            release,
            mix,
            gain,
        }
    }
    pub fn default() -> Params {
        Params {
            threshold: 0.75,
            detune: 1.0 - 0.5, // 0.943863636,
            attack: 0.0,
            release: 1.0, // 0.75,
            mix: 0.5,
            gain: 0.0,
        }
    }
}

#[derive(Clone)]
struct Envelope {
    buf: [f64; ENV_BUFFER_SIZE],
}

impl Envelope {
    pub fn new() -> Self {
        Self { buf: [0.0; ENV_BUFFER_SIZE] }
    }
    pub fn push(&mut self, val: f64) {
        for i in 0..ENV_BUFFER_SIZE-1 {
            self.buf[i] = self.buf[i+1];
        }
        self.buf[ENV_BUFFER_SIZE-1] = val;
    }
    pub fn mean(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..ENV_BUFFER_SIZE {
            sum += self.buf[i];
        }
        sum / ENV_BUFFER_SIZE as f64
    }
    pub fn mean_std_deviation(&self) -> (f64, f64) {
        let mean = self.mean();
        let mut sum = 0.0;
        for i in 0..ENV_BUFFER_SIZE {
            sum += (self.buf[i] - mean).powi(2);
        }
        (mean, (sum / ENV_BUFFER_SIZE as f64).sqrt())
    }
}

/// A phase vocoder.
///
/// Roughly translated from http://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/
pub struct PhaseVocoder {
    channels: usize,
    sample_rate: f64,
    frame_size: usize,
    time_res: usize,

    samples_waiting: usize,
    in_buf: Vec<VecDeque<f64>>,
    out_buf: Vec<VecDeque<f64>>,
    last_phase: Vec<Vec<f64>>,
    sum_phase: Vec<Vec<f64>>,
    output_accum: Vec<VecDeque<f64>>,

    forward_fft: Arc<dyn rustfft::Fft<f64>>,
    backward_fft: Arc<dyn rustfft::Fft<f64>>,

    window: Vec<f64>,

    fft_in: Vec<c64>,
    fft_out: Vec<c64>,
    fft_scratch: Vec<c64>,
    analysis_out: Vec<Vec<Bin>>,
    synthesis_in: Vec<Vec<Bin>>,

    threshold: f64,
    std_dev_threshold: f64,
    detune: f64,
    q: f64,
    attack: f64,
    release: f64,
    cur_attack: f64,
    cur_release: f64,
    mix: f64,
    envelope: f64,
    env_buf: Vec<Envelope>,
    avg_buf: Vec<Envelope>,
    std_buf: Vec<Envelope>,
    influence: f64,
    active: bool,
    samples_processed: usize,
    gain: f64,
    makeup_smooth: f64,
}

impl PhaseVocoder {
    /// Constructs a new phase vocoder.
    ///
    /// `channels` is the number of channels of audio.
    ///
    /// `sample_rate` is the sample rate.
    ///
    /// `frame_size` is the fourier transform size. It must be `> 1`.
    /// For optimal computation speed, this should be a power of 2.
    /// Will be rounded to a multiple of `time_res`.
    ///
    /// `time_res` is the number of frames to overlap.
    ///
    /// # Panics
    /// Panics if `frame_size` is `<= 1` after rounding.
    pub fn new(
        channels: usize,
        sample_rate: f64,
        frame_size: usize,
        time_res: usize,
    ) -> PhaseVocoder {
        let mut frame_size = frame_size / time_res * time_res;
        if frame_size == 0 {
            frame_size = time_res;
        }

        // If `frame_size == 1`, computing the window would panic.
        assert!(frame_size > 1);

        let mut fft_planner = rustfft::FftPlanner::new();

        let mut pv = PhaseVocoder {
            channels,
            sample_rate,
            frame_size,
            time_res,

            samples_waiting: 0,
            in_buf: vec![VecDeque::new(); channels],
            out_buf: vec![VecDeque::new(); channels],
            last_phase: vec![vec![0.0; frame_size]; channels],
            sum_phase: vec![vec![0.0; frame_size]; channels],
            output_accum: vec![VecDeque::new(); channels],

            forward_fft: fft_planner.plan_fft(frame_size, rustfft::FftDirection::Forward),
            backward_fft: fft_planner.plan_fft(frame_size, rustfft::FftDirection::Inverse),

            window: apodize::hanning_iter(frame_size)
                .map(|x| x.sqrt())
                .collect(),

            fft_in: vec![c64::new(0.0, 0.0); frame_size],
            fft_out: vec![c64::new(0.0, 0.0); frame_size],
            fft_scratch: vec![],
            analysis_out: vec![vec![Bin::empty(); frame_size]; channels],
            synthesis_in: vec![vec![Bin::empty(); frame_size]; channels],

            threshold: 0.75,
            std_dev_threshold: 6.0,
            detune: 1.0 - 0.5, // 0.943863636,
            q: 0.999999,
            attack: 0.0,
            release: 1.0, // 0.75,
            cur_attack: 0.0,
            cur_release: 0.0,
            mix: 1.0,
            envelope: 0.0,
            env_buf: vec![Envelope::new(); channels],
            avg_buf: vec![Envelope::new(); channels],
            std_buf: vec![Envelope::new(); channels],
            influence: 0.5,
            active: false,
            samples_processed: 0,
            gain: 0.0,
            makeup_smooth: 1.0,
        };
        pv.fft_scratch = vec![c64::new(0.0, 0.0); pv.forward_fft.get_outofplace_scratch_len()
            .max(pv.backward_fft.get_outofplace_scratch_len())];
        pv
    }

    pub fn num_channels(&self) -> usize {
        self.channels
    }

    pub fn num_bins(&self) -> usize {
        self.frame_size
    }

    pub fn time_res(&self) -> usize {
        self.time_res
    }

    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    pub fn set_params(&mut self, params: Params) {
        self.threshold = params.threshold;
        self.detune = params.detune;
        self.attack = params.attack;
        self.release = params.release;
        self.mix = params.mix;
        self.gain = params.gain;
    }

    /// Reads samples from `input`, processes the samples, then resynthesizes as many samples as
    /// possible into `output`. Returns the number of frames written to `output`.
    ///
    /// `processor` is a function to manipulate the spectrum before it is resynthesized. Its
    /// arguments are respectively `num_channels`, `num_bins`, `analysis_output` and
    /// `synthesis_input`.
    ///
    /// Samples are expected to be normalized to the range [-1, 1].
    ///
    /// This method can be called multiple times on the same `PhaseVocoder`.
    /// If this happens, in the analysis step, it will be assumed that the `input` is a continuation
    /// of the `input` that was passed during the previous call.
    ///
    /// It is possible that not enough data is available yet to fill `output` completely.
    /// In that case, only the first frames of `output` will be written to.
    /// Conversely, if there is more data available than `output` can hold, the remaining
    /// output is kept in the `PhaseVocoder` and can be retrieved with another call to
    /// `process` when more input data is available.
    ///
    /// # Remark
    /// The `synthesis_input` passed to the `processor_function` is currently initialised to empty
    /// bins. This behaviour may change in a future release, so make sure that your implementation
    /// does not rely on it.
    pub fn process<S>(
        &mut self,
        input: &[&[S]],
        output: &mut [&mut [S]],
        // params: Params,
        // mut processor: F,
    ) -> usize
    where
        S: Float + ToPrimitive + FromPrimitive,
        // F: FnMut(usize, usize, &[Vec<Bin>], &mut [Vec<Bin>]),
    {
        // self.threshold = params.threshold;
        // self.detune = params.detune;
        // self.attack = params.attack;
        // self.release = params.release;

        assert_eq!(input.len(), self.channels);
        assert_eq!(output.len(), self.channels);

        let min_ms = 1.0/self.sample_rate;
        let min_log = (min_ms).log10();

        // if self.mix == 0.0 {
        //     let mut n_written = 0;
        //     for chan in 0..self.channels {
        //         for samp in 0..output[chan].len() {
        //             output[chan][samp] = input[chan][samp];
        //             n_written += 1;
        //         }
        //     }
        //     return n_written / self.channels
        // }

        // push samples to input queue
        for chan in 0..input.len() {
            for sample in input[chan].iter() {
                self.in_buf[chan].push_back(sample.to_f64().unwrap());
                self.samples_waiting += 1;
            }
        }

        // Before the bin loop (per frame/channel) compute a stable pitch ratio:
        let pitch_ratio = 2.0_f64.powf(self.detune); // 2^(semitones/12)
        // If you don’t track target semitones separately, derive it from your detune parameter.
        // Avoid using the envelope-scaled `factor` for this.

        let ratio = pitch_ratio.max(0.25).min(4.0); // keep within [-24, +24] semitones
        let alpha = 0.5;                             // 0=no comp, 0.5=power-ish, 1.0=area
        let mut makeup = (1.0 / ratio).powf(alpha);

        let frame_sizef = self.frame_size as f64;
        let time_resf = self.time_res as f64;
        let step_size = frame_sizef / time_resf;
        while self.samples_waiting >= 2 * self.frame_size * self.channels {

            for _ in 0..self.time_res {
                // Initialise the synthesis bins to empty bins.
                // This may be removed in a future release.
                for synthesis_channel in self.synthesis_in.iter_mut() {
                    for bin in synthesis_channel.iter_mut() {
                        *bin = Bin::empty();
                    }
                }

                // ANALYSIS
                for chan in 0..self.channels {
                    // read in
                    for i in 0..self.frame_size {
                        self.fft_in[i] = c64::new(self.in_buf[chan][i] * self.window[i], 0.0);
                    }

                    self.forward_fft
                        .process_outofplace_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut self.fft_scratch);

                    for i in 0..self.frame_size {
                        let x = self.fft_out[i];
                        let (amp, phase) = x.to_polar();
                        let freq = self.phase_to_frequency(i, phase - self.last_phase[chan][i]);
                        self.last_phase[chan][i] = phase;

                        self.analysis_out[chan][i] = Bin::new(freq, amp * 2.0);
                    }
                }

                // PROCESSING
                let mut cur_amp;
                // self.envelope = 0.0;
                let mut factor = 0.0;
                // let mut transient_factor = 0.0;
                for chan in 0..self.channels {
                    for i in 0..self.frame_size {
                        cur_amp = self.in_buf[chan][i].abs();
                        // detect peaks
                        if (cur_amp - self.avg_buf[chan].buf[ENV_BUFFER_SIZE-1]).abs() > self.std_dev_threshold * self.std_buf[chan].buf[ENV_BUFFER_SIZE-1] {
                            if !self.active || (cur_amp > self.envelope && cur_amp > self.threshold) {
                                // peak detected
                                self.active = true;
                                self.cur_attack = 0.0;
                                self.cur_release = 0.0;
                                self.envelope = cur_amp;
                            } else {
                                // self.envelope = (self.envelope + cur_amp) / 2.0;
                            }
                        }
                        let last_amp = self.env_buf[chan].buf[ENV_BUFFER_SIZE-1];
                        self.env_buf[chan].push(cur_amp * self.influence + last_amp * (1.0 - self.influence));
                        // self.env_buf[chan].pop_front();
                        // calculate the mean and standard deviation for this frame
                        (self.avg_buf[chan].buf[ENV_BUFFER_SIZE-1], self.std_buf[chan].buf[ENV_BUFFER_SIZE-1]) = self.avg_buf[chan].mean_std_deviation();
                        if self.active {
                            // pitch shift effect is active
                            if self.cur_attack < self.attack {
                                // attack stage: pitch scale factor should trend from 1.0 toward self.detune
                                if self.cur_attack == 0.0 {
                                    factor = 1.0;
                                    // transient_factor = 1.0;
                                } else {
                                    let log_val = if self.cur_attack == 0.0 {0.0} else {(self.cur_attack/self.attack).log10()};
                                    let scaled_val = (log_val - min_log) / (0.0 - min_log);
                                    factor = 1.0 - self.envelope * self.detune * scaled_val;
                                    // transient_factor = 1.0 - self.envelope * (1.0 - scaled_val);
                                }
                                self.cur_attack += min_ms;
                            } else if self.cur_release < self.release {
                                // release stage: pitch scale factor should trend from self.detune toward 1.0
                                factor = 1.0 - self.envelope * self.detune * (1.0 - (10.0.powf(self.cur_release/self.release)*0.1)); 
                                self.cur_release += min_ms;
                                // transient_factor = 0.0;
                            } else {
                                // envelope has returned to nominal state
                                // self.envelope = 0.0;
                                self.active = false;
                                // factor = 1.0;
                                // transient_factor = 0.0;
                            }
                        } else {
                            factor = 1.0;
                        }
                        let f_in = self.analysis_out[chan][i].freq;
                        let a_in = self.analysis_out[chan][i].amp;
                        let f_out = f_in * factor;

                        let frame_sizef = self.frame_size as f64;
                        let nyquist = 0.5 * self.sample_rate;
                        let bin_hz = self.sample_rate / frame_sizef;
                        let guard_hz = 2.0 * bin_hz;      // keep a couple of bins below Nyquist
                        let low_cut = 20.0;               // your LF cutoff
                        let hi_cut  = nyquist - guard_hz; // HF cutoff with guard
                        // Soft knee around the cutoffs
                        let knee = guard_hz.max(bin_hz);
                        let lo_w = ((f_out - (low_cut - knee)) / knee).clamp(0.0, 1.0);
                        let hi_w = (((hi_cut + knee) - f_out) / knee).clamp(0.0, 1.0);
                        let edge_weight = lo_w.min(hi_w); // 1.0 in band, fades to 0.0 near edges
                        // self.synthesis_in[chan][i].amp = if f_out < low_cut || f_out > hi_cut {
                        //     0.0
                        // } else {
                        //     a_in * 2.0 // your makeup gain
                        // };
                        // self.synthesis_in[chan][i].amp = a_in * 2.0 * edge_weight;

                        // Optional: smooth across frames to avoid clicks if the target pitch changes
                        // Simple one-pole smoothing with ~5–10 ms time constant:
                        let tau_s   = 0.010;
                        let fs      = self.sample_rate;
                        let a       = (-1.0 / (tau_s * fs)).exp();
                        self.makeup_smooth = a * self.makeup_smooth + (1.0 - a) * makeup;
                        makeup = self.makeup_smooth;

                        // Clamp to sensible dB limits (e.g., ±9 dB)
                        let min_lin = 10_f64.powf(-9.0/20.0); // ≈ 0.3548
                        let max_lin = 10_f64.powf( 9.0/20.0); // ≈ 2.818
                        makeup = makeup.clamp(min_lin, max_lin);

                        self.synthesis_in[chan][i].freq = f_out;
                        // self.synthesis_in[chan][i].amp = a_in * (1.0/factor.max(0.001)).clamp(0.0, 2.0) * edge_weight;
                        self.synthesis_in[chan][i].amp  = a_in * makeup * edge_weight;

                        // if self.analysis_out[chan][i].freq < 20.0 || self.analysis_out[chan][i].freq > self.sample_rate / 2.0 {
                        //     // frequency is outside of audible range, no need to process
                        //     self.synthesis_in[chan][i].freq = self.analysis_out[chan][i].freq;
                        //     self.synthesis_in[chan][i].amp = 0.0;
                        // } else {
                        //     // apply pitch scale factor
                        //     self.synthesis_in[chan][i].freq = self.analysis_out[chan][i].freq * factor;
                        //     // apply makeup gain to compensate for pitch scale factor
                        //     self.synthesis_in[chan][i].amp = self.analysis_out[chan][i].amp * 2.0; // (1.0/factor)).clamp(0.0, 2.0);
                        // }
                    }
                }
                /*
                processor(
                    self.channels,
                    self.frame_size,
                    &self.analysis_out,
                    &mut self.synthesis_in,
                );
                */

                // SYNTHESIS
                for chan in 0..self.channels {
                    for i in 0..self.frame_size {
                        let amp = self.synthesis_in[chan][i].amp;
                        let mut freq = self.synthesis_in[chan][i].freq;

                        let freq_per_bin = self.sample_rate / frame_sizef;
                        freq -= i as f64 * freq_per_bin;
                        freq /= freq_per_bin;
                        freq = 2.0 * PI * freq / self.time_res as f64;
                        freq += i as f64 * 2.0 * PI * step_size / self.frame_size as f64;
                        self.sum_phase[chan][i] += freq;

                        //let phase = self.frequency_to_phase(freq);
                        // self.sum_phase[chan][i] += phase;
                        let phase = self.sum_phase[chan][i];

                        self.fft_in[i] = c64::from_polar(amp, phase);
                    }

                    self.backward_fft
                        .process_outofplace_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut self.fft_scratch);

                    // accumulate
                    for i in 0..self.frame_size {
                        if i == self.output_accum[chan].len() {
                            self.output_accum[chan].push_back(0.0);
                        }
                        self.output_accum[chan][i] +=
                            self.window[i] * self.fft_out[i].re / (frame_sizef * time_resf);
                    }

                    // write out
                    for _ in 0..step_size as usize {
                        self.out_buf[chan].push_back(self.output_accum[chan].pop_front().unwrap());
                        self.in_buf[chan].pop_front();
                    }
                }
            }
            self.samples_waiting -= self.frame_size * self.channels;
        }

        // pop samples from output queue
        let mut n_written = 0;
        for chan in 0..self.channels {
            for samp in 0..output[chan].len() {
                output[chan][samp] = match self.out_buf[chan].pop_front() {
                    Some(x) => FromPrimitive::from_f64(x).unwrap(),
                    None => break,
                };
                n_written += 1;
            }
        }
        n_written / self.channels
    }

    pub fn phase_to_frequency(&self, bin: usize, phase: f64) -> f64 {
        let frame_sizef = self.frame_size as f64;
        let freq_per_bin = self.sample_rate / frame_sizef;
        let time_resf = self.time_res as f64;
        let step_size = frame_sizef / time_resf;
        let expect = 2.0 * PI * step_size / frame_sizef;
        let mut tmp = phase;
        tmp -= (bin as f64) * expect;
        let mut qpd = (tmp / PI) as i32;
        if qpd >= 0 {
            qpd += qpd & 1;
        } else {
            qpd -= qpd & 1;
        }
        tmp -= PI * (qpd as f64);
        tmp = time_resf * tmp / (2.0 * PI);
        tmp = (bin as f64) * freq_per_bin + tmp * freq_per_bin;
        tmp
    }

    pub fn frequency_to_phase(&self, freq: f64) -> f64 {
        let step_size = self.frame_size as f64 / self.time_res as f64;
        2.0 * PI * freq / self.sample_rate * step_size
    }

    fn mean(&self, data: &VecDeque<f64>) -> Option<f64> {
        let sum = data.iter().sum::<f64>();
        let count = data.len();
    
        match count {
            positive if positive > 0 => Some(sum / count as f64),
            _ => None,
        }
    }
    
    fn std_deviation(&self, data: &VecDeque<f64>) -> Option<(f64, f64)> {
        match (self.mean(data), data.len()) {
            (Some(data_mean), count) if count > 0 => {
                let variance: f64 = data.iter().map(|value| {
                    let diff = data_mean - (*value as f64);
    
                    diff * diff
                }).sum::<f64>() / count as f64;
    
                Some((data_mean, variance.sqrt()))
            },
            _ => None
        }
    }
}

#[cfg(test)]
fn identity(channels: usize, bins: usize, input: &[Vec<Bin>], output: &mut [Vec<Bin>]) {
    for i in 0..channels {
        for j in 0..bins {
            output[i][j] = input[i][j];
        }
    }
}

#[cfg(test)]
fn test_data_is_reconstructed(mut pvoc: PhaseVocoder, input_samples: &[f32]) {
    let mut output_samples = vec![0.0; input_samples.len()];
    let frame_size = pvoc.num_bins();
    // Pre-padding, not collecting any output.
    pvoc.process(&[&vec![0.0; frame_size]], &mut [&mut Vec::new()]); // , Params::default(), identity);
    // The data-itself, collecting some output that we will discard
    let mut scratch = vec![0.0; frame_size];
    pvoc.process(&[&input_samples], &mut [&mut scratch]); // , Params::default()), identity);
    // Post-padding and collecting all output
    pvoc.process(
        &[&vec![0.0; frame_size]],
        &mut [&mut output_samples],
        // Params::default(),
        // identity,
    );

    assert_ulps_eq!(input_samples, output_samples.as_slice(), epsilon = 1e-2);
}

// #[bench]
// fn bench_process(b: &mut test::Bencher) {
//     let mut pvoc = PhaseVocoder::new(1, 44100.0, 256, 256 / 4);
//     let input_len = 1024;
//     let input_samples = vec![0.0; input_len];
//     let mut output_samples = vec![0.0; input_len];
//     b.iter(|| {
//         pvoc.process(
//             &[&input_samples],
//             &mut [&mut output_samples],
//             Params::default(),
//             identity,
//         );
//     });
// }

#[test]
fn identity_transform_reconstructs_original_data_hat_function() {
    let window_len = 256;
    let pvoc = PhaseVocoder::new(1, 44100.0, window_len, window_len / 4);
    let input_len = 1024;
    let mut input_samples = vec![0.0; input_len];
    for i in 0..input_len {
        if i < input_len / 2 {
            input_samples[i] = (i as f32) / ((input_len / 2) as f32)
        } else {
            input_samples[i] = 2.0 - (i as f32) / ((input_len / 2) as f32);
        }
    }

    test_data_is_reconstructed(pvoc, input_samples.as_slice());
}

#[test]
fn identity_transform_reconstructs_original_data_random_data() {
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;
    let mut rng = SmallRng::seed_from_u64(1);
    let mut input_samples = [0.0; 16384];
    rng.fill(&mut input_samples[..]);
    let pvoc = PhaseVocoder::new(1, 44100.0, 256, 256 / 4);
    test_data_is_reconstructed(pvoc, &input_samples);
}

#[test]
fn process_works_with_sample_res_equal_to_window() {
    let mut pvoc = PhaseVocoder::new(1, 44100.0, 256, 256);
    let input_len = 1024;
    let input_samples = vec![0.0; input_len];
    let mut output_samples = vec![0.0; input_len];
    pvoc.process(&[&input_samples], &mut [&mut output_samples]); // , Params::default(), identity);
}

#[test]
fn process_works_when_reading_sample_by_sample() {
    let mut pvoc = PhaseVocoder::new(1, 44100.0, 8, 2);
    let input_len = 32;
    let input_samples = vec![0.0; input_len];
    let mut output_samples = vec![0.0; input_len];
    for i in 0..input_samples.len() {
        pvoc.process(
            &[&input_samples[dbg!(i)..i + 1]],
            &mut [&mut output_samples],
            // Params::default(),
            // identity,
        );
    }
}