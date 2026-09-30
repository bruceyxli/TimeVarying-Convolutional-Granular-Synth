#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include "OrbitEngine.h"

class OrbitProcessor final : public juce::AudioProcessor {
public:
    OrbitProcessor();
    const juce::String getName() const override { return "ORBIT"; }
    void prepareToPlay(double sampleRate, int maximumBlock) override;
    void releaseResources() override {}
    void reset() override { engine.reset(); }
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    void processBlockBypassed(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    bool isBusesLayoutSupported(const BusesLayout&) const override;
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return .4; }
    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override;
    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;
    juce::AudioProcessorParameter* getBypassParameter() const override;
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameters();
    juce::AudioProcessorValueTreeState state;
    orbit::Engine engine;
private:
    enum Index { density, grain, pitch, mix, jitter, spread, lookback, output, ir, strategy, seed, bypass, count };
    std::array<std::atomic<float>*, count> values{};
    juce::AudioBuffer<float> monoScratch;
    void process(juce::AudioBuffer<float>&, bool bypassed);
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OrbitProcessor)
};
