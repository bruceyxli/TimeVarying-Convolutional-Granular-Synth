#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "UserPresetStore.h"
#include <iostream>
#include <stdexcept>

static void require(bool value,const char* reason) { if(!value) throw std::runtime_error(reason); }
int main(int argc,char** argv) {
    juce::ScopedJuceInitialiser_GUI gui;
    try {
        OrbitProcessor processor;
        auto* density=processor.state.getParameter("density");
        density->setValueNotifyingHost(density->convertTo0to1(117));
        auto* reverb=processor.state.getParameter("reverb");
        reverb->setValueNotifyingHost(.72f);
        processor.state.getParameter("variant")->setValueNotifyingHost(.5f);
        auto* longIr=processor.state.getParameter("longIr");
        longIr->setValueNotifyingHost(longIr->convertTo0to1(235));
        processor.inputView.store(2);processor.outputView.store(0);
        juce::MemoryBlock saved;
        processor.getStateInformation(saved);
        density->setValueNotifyingHost(0);
        reverb->setValueNotifyingHost(0);
        processor.state.getParameter("variant")->setValueNotifyingHost(0);
        longIr->setValueNotifyingHost(0);
        processor.inputView.store(0);processor.outputView.store(1);
        processor.setStateInformation(saved.getData(),static_cast<int>(saved.getSize()));
        require(processor.state.getRawParameterValue("density")->load()==117,"State roundtrip");
        require(std::abs(processor.state.getRawParameterValue("reverb")->load()-.72f)<.001f,"Reverb state roundtrip");
        require(processor.state.getRawParameterValue("variant")->load()==1 && processor.state.getRawParameterValue("longIr")->load()==235,"IR mode state roundtrip");
        require(processor.inputView.load()==2 && processor.outputView.load()==0,"Display modes survive session recall");
        auto legacy=processor.state.copyState();
        for(const auto* id:{"variant","longIr"})legacy.removeChild(legacy.getChildWithProperty("id",id),nullptr);
        juce::MemoryBlock legacyData;
        juce::AudioProcessor::copyXmlToBinary(*legacy.createXml(),legacyData);
        processor.setStateInformation(legacyData.getData(),static_cast<int>(legacyData.getSize()));
        require(processor.state.getRawParameterValue("variant")->load()==0 && processor.state.getRawParameterValue("longIr")->load()==120,"Legacy session defaults");
        processor.setStateInformation(saved.getData(),static_cast<int>(saved.getSize()));
        processor.setStateInformation("invalid",7);
        require(processor.state.getRawParameterValue("density")->load()==117,"Invalid state changed parameters");

        // Disk persistence is exercised in an isolated temporary directory.
        const auto presetFolder=juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("orbit-preset-test-"+juce::Uuid().toString());
        UserPresetStore library(presetFolder);juce::String savedName;
        const auto expected=UserPresetStore::capture(processor.state);
        require(library.save(juce::String::fromUTF8("\xe5\x86\xb0\xe8\x93\x9d Room"),processor.state,savedName).wasOk(),"Preset file save");
        const auto firstName=savedName;
        require(library.save(firstName,processor.state,savedName).wasOk() && savedName==firstName+" (2)","Duplicate preset must create a copy");
        require(library.save(" ",processor.state,savedName).failed(),"Empty preset name rejected");
        for(const auto& id:UserPresetStore::ids())processor.state.getParameter(id)->setValueNotifyingHost(0);
        UserPresetStore reopenedLibrary(presetFolder);
        const auto restored=reopenedLibrary.list(processor.state);
        require(restored.size()==2,"Saved presets survive a new library instance");
        require(UserPresetStore::apply(restored.front(),processor.state),"Preset recall");
        require(UserPresetStore::matches({firstName,expected},processor.state),"All preset parameters roundtrip including Reverb");
        require(presetFolder.getChildFile("broken.orbitpreset").replaceWithText("{broken"),"Corrupt fixture write");
        require(reopenedLibrary.list(processor.state).size()==2,"Malformed preset ignored without losing valid entries");
        juce::var invalid=expected.clone();invalid.getDynamicObject()->setProperty("mix",2.0);
        require(!UserPresetStore::apply({"invalid",invalid},processor.state),"Out-of-range preset rejected atomically");
        require(UserPresetStore::matches({firstName,expected},processor.state),"Invalid preset must not partially change parameters");
        auto oldValues=expected.clone();
        oldValues.getDynamicObject()->removeProperty("variant");oldValues.getDynamicObject()->removeProperty("longIr");
        auto* oldObject=new juce::DynamicObject;juce::var oldPreset(oldObject);
        oldObject->setProperty("format","ORBIT_PRESET");oldObject->setProperty("version",1);
        oldObject->setProperty("name","Legacy");oldObject->setProperty("parameters",oldValues);
        require(presetFolder.getChildFile("legacy.orbitpreset").replaceWithText(juce::JSON::toString(oldPreset)),"Legacy preset fixture");
        const auto migrated=reopenedLibrary.list(processor.state);
        require(migrated.size()==3,"Legacy preset is retained");
        for(const auto& entry:migrated)if(entry.name=="Legacy")
            require(static_cast<int>(entry.parameters["variant"])==0 && static_cast<int>(entry.parameters["longIr"])==120,"Legacy preset gets Standard defaults");
        oldObject->setProperty("version",2);
        require(presetFolder.getChildFile("partial.orbitpreset").replaceWithText(juce::JSON::toString(oldPreset)),"Partial preset fixture");
        require(reopenedLibrary.list(processor.state).size()==3,"Incomplete new presets rejected");
        for(const auto& file:presetFolder.findChildFiles(juce::File::findFiles,false))require(file.deleteFile(),"Remove temporary preset fixture");
        require(presetFolder.deleteFile(),"Remove empty preset test folder");

        auto layout=processor.getBusesLayout();
        layout.inputBuses.set(0,juce::AudioChannelSet::mono());
        layout.outputBuses.set(0,juce::AudioChannelSet::mono());
        require(processor.setBusesLayout(layout),"Mono layout rejected");
        processor.prepareToPlay(48000,32);
        processor.getBypassParameter()->setValueNotifyingHost(1);
        juce::AudioBuffer<float> audio(1,2048);
        for(int i=0;i<audio.getNumSamples();++i) audio.setSample(0,i,.1f);
        juce::MidiBuffer midi;
        processor.processBlock(audio,midi);
        for(int i=0;i<audio.getNumSamples();++i) require(audio.getSample(0,i)==.1f,"Oversized mono block / bypass");
        orbit::OutputFrame frame;int captured=0;
        while(processor.outputSpectrum.pop(frame)) {
            require(frame.sampleRate==48000,"Spectrum sample rate");
            for(size_t i=0;i<frame.left.size();++i)require(frame.left[i]==audio.getSample(0,captured++) && frame.right[i]==frame.left[i],"Spectrum captures actual mono output");
        }
        require(captured==audio.getNumSamples(),"Spectrum captures entire oversized block");
        processor.getBypassParameter()->setValueNotifyingHost(0);
        processor.state.getParameter("variant")->setValueNotifyingHost(0);
        density->setValueNotifyingHost(density->convertTo0to1(80));
        reverb->setValueNotifyingHost(0);
        processor.state.getParameter("mix")->setValueNotifyingHost(0);
        auto* output=processor.state.getParameter("output");output->setValueNotifyingHost(output->convertTo0to1(-6));
        processor.reset();
        for(int i=0;i<audio.getNumSamples();++i)audio.setSample(0,i,.2f*std::sin(2*orbit::pi*750*i/48000));
        processor.processBlock(audio,midi);captured=0;
        while(processor.outputSpectrum.pop(frame))for(size_t i=0;i<frame.left.size();++i)
            require(frame.left[i]==audio.getSample(0,captured++) && frame.right[i]==frame.left[i],"Output spectrum must be after gain and mono summing");
        require(captured==audio.getNumSamples() && std::abs(audio.getSample(0,16)-.2f*std::pow(10.0f,-6.0f/20))<.00001f,"Post-gain output differs from input");
        // Leave actual processed tone frames for the editor snapshot.
        processor.outputSpectrum.capture(audio.getReadPointer(0),nullptr,audio.getNumSamples(),48000);
        processor.state.getParameter("mix")->setValueNotifyingHost(.65f);
        processor.inputView.store(0);processor.outputView.store(1);
        {
            std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
            require(editor && editor->getWidth()>0,"Editor creation");
            auto snapshot=editor->createComponentSnapshot(editor->getLocalBounds(),true);
            require(snapshot.isValid(),"Native editor rendering");
            AudioDisplay *inputDisplay=nullptr,*outputDisplay=nullptr;
            std::array<juce::TextButton*,3> modes{};
            for(auto* child:editor->getChildren()) {
                if(auto* display=dynamic_cast<AudioDisplay*>(child)) {
                    if(child->getTitle()=="INPUT visualization")inputDisplay=display;else outputDisplay=display;
                }
                if(auto* button=dynamic_cast<juce::TextButton*>(child)) {
                    if(button->getButtonText()=="PER GRAIN")modes[0]=button;
                    if(button->getButtonText()=="PRE CONV")modes[1]=button;
                    if(button->getButtonText()=="GRAIN IR")modes[2]=button;
                }
            }
            require(inputDisplay && outputDisplay && modes[0] && modes[1] && modes[2],"Main-page visualization and signal-path controls");
            const auto beforeViews=UserPresetStore::capture(processor.state);
            inputDisplay->onClick();require(inputDisplay->viewMode()==1 && outputDisplay->viewMode()==1,"Input cycles independently");
            inputDisplay->onClick();outputDisplay->onClick();
            require(inputDisplay->viewMode()==2 && outputDisplay->viewMode()==2,"Spectrogram selectable on both sides");
            require(UserPresetStore::matches({"views",beforeViews},processor.state),"Display switching never changes sound parameters");
            for(int frameIndex=0;frameIndex<96;++frameIndex) {
                processor.inputSpectrum.capture(audio.getReadPointer(0),nullptr,audio.getNumSamples(),48000);
                processor.outputSpectrum.capture(audio.getReadPointer(0),nullptr,audio.getNumSamples(),48000);
                inputDisplay->refresh();outputDisplay->refresh();
            }
            for(int mode=0;mode<3;++mode) {
                modes[static_cast<size_t>(mode)]->onClick();
                require(processor.state.getRawParameterValue("variant")->load()==mode,"Main switch drives the automatable signal-path parameter");
                for(auto* child:editor->getChildren())if(auto* orbitPad=dynamic_cast<OrbitPad*>(child))for(int step=0;step<30;++step)orbitPad->updateVisuals();
                if(argc>1) {
                    const juce::File original(juce::String::fromUTF8(argv[1]));
                    auto stream=original.getSiblingFile(original.getFileNameWithoutExtension()+"-mode-"+juce::String(mode)+".png").createOutputStream();
                    require(stream!=nullptr,"Main mode screenshot");
                    require(juce::PNGImageFormat().writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds(),true),*stream),"Main mode screenshot encoding");
                }
            }
            inputDisplay->onClick();outputDisplay->onClick();outputDisplay->onClick();
            require(inputDisplay->viewMode()==0 && outputDisplay->viewMode()==1,"Display cycle wraps to original views");
            modes[0]->onClick();
            juce::TextButton* navigation=nullptr;
            for(auto* child:editor->getChildren())
                if(auto* button=dynamic_cast<juce::TextButton*>(child);button && button->getButtonText()=="Details >")navigation=button;
            require(navigation!=nullptr,"Details navigation available");
            const auto initialBounds=editor->getBounds();
            const auto initialSettings=UserPresetStore::capture(processor.state);
            navigation->onClick();
            require(navigation->getButtonText()=="< Back" && editor->getBounds()==initialBounds,"Details changes page without resizing the host window");
            if(argc>1) {
                const juce::File path(juce::String::fromUTF8(argv[1]));
                const auto detailImage=path.getSiblingFile(path.getFileNameWithoutExtension()+"-details.png");
                auto stream=detailImage.createOutputStream();
                require(stream!=nullptr,"Details screenshot output");
                require(juce::PNGImageFormat().writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds(),true),*stream),"Details screenshot encoding");
            }
            navigation->onClick();
            require(navigation->getButtonText()=="Details >" && editor->getBounds()==initialBounds,"Return to main page");
            require(UserPresetStore::matches({"before navigation",initialSettings},processor.state),"Page navigation preserves every parameter");
            if(argc>1) {
                juce::File path(juce::String::fromUTF8(argv[1]));
                path.deleteFile();
                auto stream=path.createOutputStream();
                require(stream!=nullptr,"Screenshot output");
                require(juce::PNGImageFormat().writeImageToStream(snapshot,*stream),"Screenshot encoding");
            }
        }
        {
            reverb->setValueNotifyingHost(.8f);
            std::unique_ptr<juce::AudioProcessorEditor> reopened(processor.createEditor());
            require(reopened!=nullptr,"Editor reopen");
            if(argc>1) {
                const juce::File original(juce::String::fromUTF8(argv[1]));
                const auto wet=original.getSiblingFile(original.getFileNameWithoutExtension()+"-reverb.png");
                wet.deleteFile();auto stream=wet.createOutputStream();
                require(stream!=nullptr,"Reverb screenshot output");
                require(juce::PNGImageFormat().writeImageToStream(reopened->createComponentSnapshot(reopened->getLocalBounds(),true),*stream),"Reverb screenshot encoding");
            }
        }
        processor.releaseResources();
        for(int mode=1;mode<=2;++mode) {
            processor.state.getParameter("variant")->setValueNotifyingHost(static_cast<float>(mode)*.5f);
            std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
            juce::TextButton* navigation=nullptr;juce::Component *longControl=nullptr,*micro=nullptr,*selection=nullptr;
            for(auto* child:editor->getChildren()) {
                if(auto* button=dynamic_cast<juce::TextButton*>(child);button && button->getButtonText()=="Details >")navigation=button;
                if(child->getTitle()=="Long IR")longControl=child;
                if(child->getTitle()=="IR length")micro=child;
                if(child->getTitle()=="Signal path")selection=child;
            }
            require(navigation && longControl && micro && selection,"Mode controls exist");navigation->onClick();
            require(selection->isVisible() && !micro->isVisible() && longControl->isVisible()==(mode==1),"Details exposes only relevant mode controls");
            if(argc>1) {
                const juce::File original(juce::String::fromUTF8(argv[1]));
                const auto path=original.getSiblingFile(original.getFileNameWithoutExtension()+"-details-"+juce::String(mode)+".png");
                auto stream=path.createOutputStream();require(stream!=nullptr,"Mode screenshot output");
                require(juce::PNGImageFormat().writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds(),true),*stream),"Mode screenshot encoding");
            }
        }
        std::cout<<"PASS: state restore, invalid state, oversized mono block, bypass, native editor paint/reopen\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<"\n"; return 1; }
}
