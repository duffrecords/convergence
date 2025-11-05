# Convergence
An envelope-driven detune plugin

This plugin produces a guitar effect similar to [glide guitar](https://en.wikipedia.org/wiki/Glide_guitar), a technique involving gently pushing or pulling on the whammy bar while strumming, causing a slight detuning before returning to pitch. This is not possible to do on a guitar with a fixed bridge or on a pre-recorded track but the plugin can simulate the effect by applying a pitch envelope when the audio exceeds a threshold. In other words, the harder you strum, the more pronounced the detune effect. Because it relies on peak detection, the plugin should be inserted prior to any compression effects, including overdrive or distortion.

## Requirements
* Linux or macOS
* a working Rust installation. See [rustup](https://rustup.rs/) for more details.
* an LV2 host such as [Ardour](https://ardour.org) or [Carla](https://kx.studio/Applications:Carla)

## Installation
Run the following script:
```bash
./install.sh
```
This will copy the resulting artifact to your user's LV2 directory (`~/.lv2` on Linux or `~/Library/Audio/Plug-Ins/LV2` on macOS). Optionally, you can run `./install.sh debug` to build the plugin with debug symbols included.

## Usage
Open your LV2 host and scan for new plugins. Add Convergence to your signal chain. With the `attack` control set to 0, adjust the `threshold` control until you hear the effect being triggered on the loudest notes.

|parameter|min|max|description|
|---|---|---|---|
|threshold|-60|0|the level at which the effect is triggered|
|detune|-24|0|pitch shift amount, in semitones|
|attack|0|100|how quickly the audio is detuned, in ms|
|release|0|1000|how quickly the audio returns to its original pitch, in ms|
|mix|0|1|dry/wet mix factor|
|level|-80|20|output level|
