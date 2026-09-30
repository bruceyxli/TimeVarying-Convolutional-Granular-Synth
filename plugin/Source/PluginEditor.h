#pragma once
#include "PluginProcessor.h"
#include "UserPresetStore.h"

struct PresetCombo : juce::ComboBox {
    std::function<void()> onOpen;
    void showPopup() override { if(onOpen)onOpen();juce::ComboBox::showPopup(); }
};
struct PresetSavePanel : juce::Component {
    void paint(juce::Graphics& g) override {
        g.setColour(juce::Colour(0xff121c29));g.fillRoundedRectangle(getLocalBounds().toFloat(),8);
        g.setColour(juce::Colour(0xff344c63));g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(.5f),8,1);
    }
};

struct OrbitLook : juce::LookAndFeel_V4 {
    OrbitLook();
    juce::Typeface::Ptr face;
    juce::Font font(float size) const;
    juce::Font getComboBoxFont(juce::ComboBox&) override { return font(14); }
    juce::Font getTextButtonFont(juce::TextButton&,int) override { return font(12); }
    juce::Font getLabelFont(juce::Label&) override { return font(12); }
    void drawLinearSlider(juce::Graphics&,int,int,int,int,float,float,float,juce::Slider::SliderStyle,juce::Slider&) override;
    void drawRotarySlider(juce::Graphics&,int,int,int,int,float,float,float,juce::Slider&) override;
};

class InputScope final : public juce::Component, private juce::Timer {
public:
    InputScope(orbit::ScopeQueue&,OrbitLook&);
    void paint(juce::Graphics&) override;
private:
    void timerCallback() override;
    orbit::ScopeQueue& queue;
    OrbitLook& look;
    std::array<orbit::ScopeFrame,256> history{};
    size_t head=0;
    double lastFrame=0;
    bool stale=true;
};

class OrbitPad final : public juce::Component {
public:
    explicit OrbitPad(juce::AudioProcessorValueTreeState&);
    ~OrbitPad() override;
    void paint(juce::Graphics&) override;
    void mouseDown(const juce::MouseEvent&) override;
    void mouseDrag(const juce::MouseEvent&) override;
    void mouseUp(const juce::MouseEvent&) override;
    void mouseMove(const juce::MouseEvent&) override;
    void mouseExit(const juce::MouseEvent&) override;
    void updateVisuals();
private:
    juce::AudioProcessorValueTreeState& state;
    bool dragging=false;
    bool handleHovered=false;
    float handleActivity=0;
    std::array<float,5> visual{};
    std::array<float,5> targets() const;
    void move(juce::Point<float>);
    void finish();
};

class OrbitEditor final : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit OrbitEditor(OrbitProcessor&);
    ~OrbitEditor() override;
    void paint(juce::Graphics&) override;
    void resized() override;
private:
    OrbitProcessor& processor;
    OrbitLook look;
    InputScope scope;
    OrbitPad pad;
    juce::Slider density,grain,pitch,mix,jitter,spread,lookback,output,seed,reverb;
    PresetCombo preset;
    juce::ComboBox ir,strategy;
    UserPresetStore presetStore;
    std::vector<UserPresetStore::Entry> userPresets;
    juce::TextButton savePreset{"Save"},confirmSave{"Save"},cancelSave{"Cancel"};
    PresetSavePanel savePanel;
    juce::TextEditor presetName;
    juce::Label savePrompt,saveError;
    juce::TextButton previous{"<"},next{">"},details{"Details +"};
    juce::ToggleButton bypass{"Bypass"};
    using SliderAttachment=juce::AudioProcessorValueTreeState::SliderAttachment;
    using ComboAttachment=juce::AudioProcessorValueTreeState::ComboBoxAttachment;
    std::vector<std::unique_ptr<SliderAttachment>> sliders;
    std::unique_ptr<ComboAttachment> irAttachment,strategyAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;
    std::array<float,13> previousValues{};
    bool expanded=false;
    void timerCallback() override;
    void applyPreset(int);
    void syncPreset();
    void refreshPresets();
    void showSavePreset();
    void commitPreset();
    void setExpanded(bool);
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OrbitEditor)
};
