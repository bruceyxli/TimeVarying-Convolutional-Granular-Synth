#include "PluginProcessor.h"
#include "PluginEditor.h"

juce::AudioProcessorValueTreeState::ParameterLayout OrbitProcessor::createParameters() {
    juce::AudioProcessorValueTreeState::ParameterLayout p;
    auto add=[&](const char* id,const char* name,float lo,float hi,float step,float initial) {
        p.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{id,1},name,
            juce::NormalisableRange<float>{lo,hi,step},initial));
    };
    add("density","Density",10,120,1,80);
    add("grain","Grain size",5,50,.1f,15);
    add("pitch","Pitch scatter",0,12,.1f,5);
    add("mix","Dry / Wet",0,1,.001f,.65f);
    add("jitter","Trigger jitter",0,.5f,.001f,.08f);
    add("spread","Stereo spread",0,1,.001f,.6f);
    add("lookback","Lookback",0,200,1,40);
    add("output","Output gain",-24,6,.1f,-3);
    p.add(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID{"ir",1},"IR length",juce::StringArray{"8 ms","16 ms","24 ms","32 ms"},1));
    p.add(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID{"strategy",1},"IR selection",juce::StringArray{"Fixed","Cycle","Random","Weighted","Centroid"},3));
    p.add(std::make_unique<juce::AudioParameterInt>(juce::ParameterID{"seed",1},"Random seed",0,65535,2025));
    p.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"bypass",1},"Bypass",false));
    return p;
}
OrbitProcessor::OrbitProcessor()
    : AudioProcessor(BusesProperties().withInput("Input",juce::AudioChannelSet::stereo(),true)
                                    .withOutput("Output",juce::AudioChannelSet::stereo(),true)),
      state(*this,nullptr,"ORBIT_STATE",createParameters()) {
    const char* ids[]{"density","grain","pitch","mix","jitter","spread","lookback","output","ir","strategy","seed","bypass"};
    for(size_t i=0;i<values.size();++i) values[i]=state.getRawParameterValue(ids[i]);
}
void OrbitProcessor::prepareToPlay(double rate,int maximumBlock) {
    engine.prepare(rate);
    // Hosts may send larger blocks; split them into this preallocated capacity.
    monoScratch.setSize(1,std::max(1,maximumBlock));
    setLatencySamples(0); // lookback is a creative delay, not FFT buffering latency
}
bool OrbitProcessor::isBusesLayoutSupported(const BusesLayout& layout) const {
    const auto channels=layout.getMainOutputChannelSet();
    return (channels==juce::AudioChannelSet::mono() || channels==juce::AudioChannelSet::stereo())
        && layout.getMainInputChannelSet()==channels;
}
void OrbitProcessor::process(juce::AudioBuffer<float>& buffer,bool hostBypassed) {
    juce::ScopedNoDenormals noDenormals;
    auto v=[&](Index index){return values[static_cast<size_t>(index)]->load(std::memory_order_relaxed);};
    orbit::Parameters p;
    p.density=v(density); p.grainMs=v(grain); p.pitch=v(pitch); p.mix=v(mix);
    p.jitter=v(jitter); p.spread=v(spread); p.lookbackMs=v(lookback); p.outputDb=v(output);
    p.irLength=juce::roundToInt(v(ir)); p.strategy=juce::roundToInt(v(strategy));
    p.seed=static_cast<uint32_t>(v(seed)); p.bypass=hostBypassed || v(bypass)>.5f;
    for(int c=getTotalNumInputChannels();c<buffer.getNumChannels();++c) buffer.clear(c,0,buffer.getNumSamples());
    if(buffer.getNumChannels()==0) return;
    if(buffer.getNumChannels()>1) engine.process(buffer.getWritePointer(0),buffer.getWritePointer(1),buffer.getNumSamples(),p);
    else {
        const int capacity=monoScratch.getNumSamples();
        if(capacity==0) return;
        for(int offset=0;offset<buffer.getNumSamples();offset+=capacity) {
            const int length=std::min(capacity,buffer.getNumSamples()-offset);
            auto* l=buffer.getWritePointer(0)+offset; auto* r=monoScratch.getWritePointer(0);
            std::copy_n(l,length,r);
            engine.process(l,r,length,p);
            for(int i=0;i<length;++i) l[i]=(l[i]+r[i])*.5f;
        }
    }
}
void OrbitProcessor::processBlock(juce::AudioBuffer<float>& b,juce::MidiBuffer&) { process(b,false); }
void OrbitProcessor::processBlockBypassed(juce::AudioBuffer<float>& b,juce::MidiBuffer&) { process(b,true); }
juce::AudioProcessorParameter* OrbitProcessor::getBypassParameter() const { return state.getParameter("bypass"); }
juce::AudioProcessorEditor* OrbitProcessor::createEditor() { return new OrbitEditor(*this); }
void OrbitProcessor::getStateInformation(juce::MemoryBlock& data) {
    auto tree=state.copyState(); tree.setProperty("schemaVersion",1,nullptr);
    if(auto xml=tree.createXml()) copyXmlToBinary(*xml,data);
}
void OrbitProcessor::setStateInformation(const void* data,int size) {
    if(auto xml=getXmlFromBinary(data,size))
        if(xml->hasTagName(state.state.getType())) state.replaceState(juce::ValueTree::fromXml(*xml));
}
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new OrbitProcessor(); }
