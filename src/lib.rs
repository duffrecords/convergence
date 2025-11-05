#[cfg(test)]
#[macro_use]
extern crate approx;

mod pvoc;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_MM_HINT_ET1;

use lv2::prelude::*;
// use phase_vocoder::PhaseVocoder;
use crate::pvoc::{Bin, Params, PhaseVocoder};
// use realfft::RealFftPlanner;
// use rustfft::{Fft, FftDirection};
// use rustfft::num_complex::Complex;
// use rustfft::num_traits::Zero;

const MAX_FRAME_LENGTH: usize = 8192;

fn db_to_coef(db: f64) -> f64 {
    if db > -90.0 {
        10.0f64.powf(db / 20.0)
    } else {
        0.0
    }
}

fn coef_to_db(coef: f64) -> f64 {
    20.0 * coef.log10()
}

#[derive(PortCollection)]
pub struct Ports {
    input: InputPort<Audio>,
    output: OutputPort<Audio>,
    threshold: InputPort<Control>,
    detune: InputPort<Control>,
    attack: InputPort<Control>,
    release: InputPort<Control>,
    mix: InputPort<Control>,
    gain: InputPort<Control>,
    // ... add other necessary ports
}

#[derive(FeatureCollection)]
pub struct Features<'a> {
    map: LV2Map<'a>,
    // options: LV2Options,
}

#[uri("https://github.com/duffrecords/convergence")]
struct Convergence {
    phase_vocoder: PhaseVocoder,
    params: Params,
    // mix: f64,
    envelope: f64,
    gain: f64,
}

impl Plugin for Convergence {
    type Ports = Ports;

    type InitFeatures = Features<'static>;
    type AudioFeatures = ();

    fn new(plugin_info: &PluginInfo, features: &mut Features<'static>) -> Option<Self> {
        Some(Convergence {
            phase_vocoder: PhaseVocoder::new(1, plugin_info.sample_rate(), 1024, 16),
            //params: Params::new(0.35, 0.001, 0.999999, 0.001, 1.0, 0.5, 0.0),
            params: Params::new(0.35, 0.001, 0.001, 1.0, 0.5, 0.0),
            // mix: 1.0,
            envelope: 0.0,
            gain: 0.0,
        })
    }
    fn run(&mut self, ports: &mut Ports, _features: &mut (), sample_count: u32) {
        let mut parameters_updated = false;
        if *ports.threshold as f64 != self.params.threshold {
            self.params.threshold = db_to_coef(*ports.threshold as f64);
            parameters_updated = true;
        }
        if *ports.detune as f64 != self.params.detune {
            self.params.detune = 1.0 - 2.0_f64.powf(*ports.detune as f64/12.0);
            parameters_updated = true;
        }
        if *ports.attack as f64 != self.params.attack {
            self.params.attack = *ports.attack as f64;
            parameters_updated = true;
        }
        if *ports.release as f64 != self.params.release {
            self.params.release = *ports.release as f64;
            parameters_updated = true;
        }
        if *ports.mix as f64 != self.params.mix {
            let mix = {
                let m = *ports.mix as f64;
                if m.is_finite() { m } else { 0.0 }
            };
            self.params.mix = mix;
            // self.mix = mix;
            parameters_updated = true;
        }
        if *ports.gain as f64 != self.gain {
            let gain = {
                let l = *ports.gain;
                if l.is_finite() { l } else { 0.0 }
            };
            self.params.gain = db_to_coef(gain as f64);
            parameters_updated = true;
        }

        let mix = {
            let m = *ports.mix;
            if m.is_finite() { m } else { 0.0 }
        };

            // Push params to the vocoder before processing.
        if parameters_updated {
            self.phase_vocoder.set_params(self.params);
        }

        // ---- safe, alias-aware I/O ----
        let n_req = sample_count as usize;

        // LV2 hosts should give buffers sized to the current cycle, but be defensive:
        let in_all: &[f32] = &ports.input;
        let out_all: &mut [f32] = &mut ports.output;
        let n = n_req.min(in_all.len()).min(out_all.len());

        let dry_in: &[f32] = &in_all[..n];
        let out: &mut [f32] = &mut out_all[..n];

        // If pure dry, we can skip processing entirely.
        if mix <= f32::EPSILON {
            // If buffers alias, output already equals input; otherwise copy.
            if !std::ptr::eq(dry_in.as_ptr(), out.as_ptr()) {
                out.copy_from_slice(dry_in);
            }
            return;
        }

        // Create a wet buffer and render vocoder output into it.
        // (We always go through a temp wet buffer to keep aliasing safe and enable mixing.)
        let mut wet = vec![0.0_f32; n];
        {
            let mut wet_slice: &mut [f32] = &mut wet[..];
            // Mono: pass one channel in/out. Extend to N channels if needed.
            // NOTE: set_params() was already called above.
            self.phase_vocoder
                .process(&[dry_in], &mut [&mut wet_slice]);
        }

        // If pure wet, copy straight to out.
        if (1.0 - mix) <= f32::EPSILON {
            out.copy_from_slice(&wet);
            return;
        }

        // Otherwise, blend safely. If input and output alias, preserve a copy of the dry.
        let aliasing = std::ptr::eq(dry_in.as_ptr(), out.as_ptr());
        if aliasing {
            let dry_copy = dry_in.to_vec();
            for i in 0..n {
                out[i] = dry_copy[i] + (wet[i] - dry_copy[i]) * mix;
            }
        } else {
            for i in 0..n {
                out[i] = dry_in[i] + (wet[i] - dry_in[i]) * mix;
            }
        }

        // let input = ports.input;
        // let mut output = ports.output;
        // for (in_frame, out_frame) in Iterator::zip(ports.input.iter(), ports.output.iter_mut()) {
        //     *out_frame = *in_frame;
        // }
        // self.phase_vocoder.process(&[&ports.input[..]], &mut [&mut ports.output[..]], self.process(channels, bins, input, output));
        // self.params.threshold = *ports.threshold as f64;
        // self.params.detune = *ports.detune as f64;
        // self.params.attack = *ports.attack as f64;
        // self.params.release = *ports.release as f64;
        // self.phase_vocoder.process(&[&ports.input[..]], &mut [&mut ports.output[..]]); // , self.params, |channels: usize, bins: usize, analysis_output: &[Vec<Bin>], synthesis_input: &mut [Vec<Bin>]| {
            /*
            let mut factor = 1.0;
            for i in 0..channels {
                let mut envelope = last_envelope[i];
                for j in 0..bins {
                    if analysis_output[i][j].freq > 0.0 {
                        if analysis_output[i][j].amp > 80.0 && analysis_output[i][j].amp > envelope {
                            envelope = analysis_output[i][j].amp; // (envelope + analysis_output[i][j].amp) / 2.0;
                        } else {
                            envelope = (envelope + analysis_output[i][j].amp) / 2.0;
                        }
                        // envelope = (envelope + analysis_output[i][j].amp) / 2.0;
                        // envelope = analysis_output[i][j].amp;
                        factor = 1.0 - (envelope/96.0) * 0.5; //0.056136364;
                        // 0.943863636
                        // 0.056136364
                        synthesis_input[i][j].freq = analysis_output[i][j].freq * factor;
                        synthesis_input[i][j].amp = analysis_output[i][j].amp;
                        if j % 32 == 0 {
                            println!("freq: {:.4}\tamp: {:.3}\tshifted freq: {:.3}\tenvelope: {:?}\tfactor: {:.6}", analysis_output[i][j].freq, analysis_output[i][j].amp, synthesis_input[i][j].freq, envelope, factor);
                        }
                    } else {
                        envelope = (envelope + analysis_output[i][j].amp) / 2.0;
                        // factor = 1.0 - (envelope/1.0) * 0.056136364;
                        synthesis_input[i][j].freq = analysis_output[i][j].freq * factor;
                        synthesis_input[i][j].amp = analysis_output[i][j].amp;
                        if j % 32 == 0 {
                            println!("freq: {:.4}\tamp: same\tshifted freq: same\tenvelope: {:?}\tfactor: {:.6}", analysis_output[i][j].freq, envelope, factor);
                        }
                    };
                }
            }
            */
//        });
    }
}

impl Convergence {
    fn process(
        &mut self,
        // ports: &[f64],
        // _: f64,
        channels: usize,
        bins: usize,
        input: &[Vec<Bin>],
        output: &mut [Vec<Bin>],
    ) {
        let shift = 0.943863636;
        for i in 0..channels {
            for j in 0..bins / 2 {
                let index = ((j as f64) * shift) as usize;
                if index < bins / 2 {
                    output[i][index].freq = input[i][j].freq * shift;
                    output[i][index].amp += input[i][j].amp;
                }
            }
        }
    }
}

lv2_descriptors!(Convergence);
