#pragma once
#include "PluginProcessor.h"
#include "UserPresetStore.h"
#include <cstdlib>
#include <cmath>

class ParameterReadout final : public juce::Label {
public:
    void bind(juce::AudioProcessorValueTreeState& state,const char* id,float displayScale,int places) {
        parameter=state.getParameter(id);raw=state.getRawParameterValue(id);scale=displayScale;decimals=places;
        setEditable(false,true,true);setWantsKeyboardFocus(true);setMouseCursor(juce::MouseCursor::IBeamCursor);
        setBorderSize(juce::BorderSize<int>(0));
        onTextChange=[this]{commit(getText());setText(formatted(),juce::dontSendNotification);};sync();
    }
    bool commit(const juce::String& input) {
        const auto clean=input.trim();const auto bytes=clean.toRawUTF8();char* end=nullptr;
        const double number=std::strtod(bytes,&end);
        if(clean.isEmpty() || end==bytes || *end!='\0' || !std::isfinite(number))return false;
        const auto& range=parameter->getNormalisableRange();
        const auto bounded=static_cast<float>(std::clamp(number/static_cast<double>(scale),static_cast<double>(range.start),static_cast<double>(range.end)));
        parameter->beginChangeGesture();parameter->setValueNotifyingHost(parameter->convertTo0to1(range.snapToLegalValue(bounded)));parameter->endChangeGesture();
        sync();return true;
    }
    void sync() { if(raw && !isBeingEdited())setText(formatted(),juce::dontSendNotification); }
    bool keyPressed(const juce::KeyPress& key) override {
        if(key==juce::KeyPress::returnKey || key==juce::KeyPress::spaceKey){showEditor();return true;}
        return juce::Label::keyPressed(key);
    }
protected:
    juce::TextEditor* createEditorComponent() override {
        auto* editor=juce::Label::createEditorComponent();editor->setInputRestrictions(12,"0123456789.-+");
        editor->setFont(getFont());editor->setJustification(juce::Justification::centredLeft);
        editor->setColour(juce::TextEditor::backgroundColourId,juce::Colour(0xff121c29));
        editor->setColour(juce::TextEditor::textColourId,juce::Colour(0xffdfebf8));
        editor->setColour(juce::TextEditor::focusedOutlineColourId,juce::Colour(0xff82cfff));
        return editor;
    }
private:
    juce::String formatted() const {return juce::String(raw->load()*scale,decimals);}
    juce::RangedAudioParameter* parameter=nullptr;
    std::atomic<float>* raw=nullptr;
    float scale=1;int decimals=0;
};

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
    juce::Font getLabelFont(juce::Label& label) override { return dynamic_cast<ParameterReadout*>(&label)?label.getFont():font(12); }
    void drawLinearSlider(juce::Graphics&,int,int,int,int,float,float,float,juce::Slider::SliderStyle,juce::Slider&) override;
    void drawRotarySlider(juce::Graphics&,int,int,int,int,float,float,float,juce::Slider&) override;
};

// Both displays use the same scales and can be cycled independently.
class AudioDisplay final : public juce::Button, private juce::Timer {
public:
    AudioDisplay(orbit::OutputQueue&,OrbitLook&,std::atomic<int>&,const juce::String&);
    void paintButton(juce::Graphics&,bool,bool) override;
    int viewMode() const noexcept { return mode.load(); }
    void refresh();
private:
    void timerCallback() override;
    void updateHelp();
    orbit::OutputQueue& queue;
    OrbitLook& look;
    std::atomic<int>& mode;
    juce::String caption;
    orbit::SpectrumAnalyzer analyzer;
    std::array<float,64> levels{};
    std::array<orbit::ScopeFrame,256> waveform{};
    orbit::ScopeFrame partial{};
    int waveHead=0,waveCount=0,spectroHead=0,emptyColumns=192;
    int helpMode=-1;
    bool waveEmpty=true;
    uint32_t generation=~uint32_t{0};
    double rate=0,lastData=0;
    juce::Image spectrogram{juce::Image::RGB,192,64,true};
};

class OrbitPad final : public juce::Component, public juce::SettableTooltipClient {
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
    std::array<float,7> visual{};
    std::array<float,7> targets() const;
    void move(juce::Point<float>);
    void finish();
};

class OrbitEditor final : public juce::AudioProcessorEditor, public juce::TooltipClient, private juce::Timer {
public:
    explicit OrbitEditor(OrbitProcessor&);
    ~OrbitEditor() override;
    void paint(juce::Graphics&) override;
    void resized() override;
    juce::String getTooltip() override;
private:
    OrbitProcessor& processor;
    OrbitLook look;
    AudioDisplay scope,spectrum;
    OrbitPad pad;
    juce::Slider density,grain,pitch,mix,jitter,spread,lookback,output,seed,reverb,longIr;
    std::array<ParameterReadout,4> readouts;
    PresetCombo preset;
    juce::ComboBox ir,strategy,variant;
    std::array<juce::TextButton,3> modeButtons;
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
    std::unique_ptr<ComboAttachment> irAttachment,strategyAttachment,variantAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;
    std::array<float,15> previousValues{};
    bool expanded=false;
    juce::TooltipWindow tooltips{this,550};
    void timerCallback() override;
    void applyPreset(int);
    void syncPreset();
    void refreshPresets();
    void showSavePreset();
    void commitPreset();
    void setExpanded(bool);
    void updateModeControls();
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OrbitEditor)
};
