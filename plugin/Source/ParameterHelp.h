#pragma once
#include <juce_gui_basics/juce_gui_basics.h>

inline juce::String orbitParameterHelp(const juce::String& id) {
    if(id=="variant")return "Signal Path\nChoose per-grain convolution, convolution before granulation, or grains used as impulse responses. All modes share the final Dry / Wet control.";
    if(id=="longIr")return "Long IR\nThe response length for Convolve then Granulate, from 40 to 300 ms. Longer responses add more texture before grain processing.";
    if(id=="spectrum")return "Output Spectrum\nLive frequency content of the final output, after Dry / Wet, Reverb and Output Gain. Stereo energy is combined without phase cancellation.";
    if(id=="density")return "Density\nGrains triggered per second. Higher values create a denser texture; lower values leave more space.";
    if(id=="grain")return "Grain Size\nThe duration of each grain. Short grains sound more fragmented; longer grains retain more of the source.";
    if(id=="pitch")return "Pitch Scatter\nRandom pitch variation per grain, in semitones. Zero keeps the original pitch.";
    if(id=="mix")return "Dry / Wet\nBlends the original input with the grain effect. 0% is dry; 100% is wet. Reverb is applied after this mix.";
    if(id=="reverb")return "Reverb\nAdds space and a decaying tail. At zero, the halo is white. Higher amounts turn it blue and increase the glow.";
    if(id=="jitter")return "Trigger Jitter\nRandom variation in grain timing. Zero gives regular triggers; higher values create a looser rhythm.";
    if(id=="spread")return "Stereo Spread\nThe width of random grain panning. Zero keeps grains centered; higher values spread them across the stereo field.";
    if(id=="lookback")return "Lookback\nReads grains from earlier input audio. Higher values reach further back into the input history.";
    if(id=="output")return "Output Gain\nAdjusts the final output level, in dB.";
    if(id=="ir")return "IR Length\nThe duration of each grain's impulse response. Shorter responses feel tighter; longer ones add more resonance.";
    if(id=="strategy")return "IR Selection\nChooses a response for each grain: fixed, cycling, random, weighted, or matched by spectral centroid.";
    if(id=="seed")return "Random Seed\nChanges the random sequence. The same input, seed and starting state reproduce the same variation.";
    if(id=="bypass")return "Bypass\nPasses the original input through, skipping the grain effect, reverb and output gain.";
    if(id=="xy")return "XY Pad\nDrag in the circular field: X controls Density and Y controls Grain Size. Up increases grain duration. Pitch Scatter is independent. Dragging outside stays on the circular boundary.";
    if(id=="input")return "Live Input\nShows the left and right input waveforms from your host. The clip indicator lights up when input peaks reach full scale.";
    return {};
}
