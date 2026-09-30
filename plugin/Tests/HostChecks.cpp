#include "PluginProcessor.h"
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
        juce::MemoryBlock saved;
        processor.getStateInformation(saved);
        density->setValueNotifyingHost(0);
        reverb->setValueNotifyingHost(0);
        processor.setStateInformation(saved.getData(),static_cast<int>(saved.getSize()));
        require(processor.state.getRawParameterValue("density")->load()==117,"State roundtrip");
        require(std::abs(processor.state.getRawParameterValue("reverb")->load()-.72f)<.001f,"Reverb state roundtrip");
        processor.setStateInformation("invalid",7);
        require(processor.state.getRawParameterValue("density")->load()==117,"Invalid state changed parameters");

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
        processor.getBypassParameter()->setValueNotifyingHost(0);
        density->setValueNotifyingHost(density->convertTo0to1(80));
        reverb->setValueNotifyingHost(0);
        {
            std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
            require(editor && editor->getWidth()>0,"Editor creation");
            auto snapshot=editor->createComponentSnapshot(editor->getLocalBounds(),true);
            require(snapshot.isValid(),"Native editor rendering");
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
        std::cout<<"PASS: state restore, invalid state, oversized mono block, bypass, native editor paint/reopen\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<"\n"; return 1; }
}
