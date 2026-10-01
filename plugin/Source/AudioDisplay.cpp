#include "PluginEditor.h"
#include <cmath>

namespace {
const juce::Colour background{0xff0c1118},line{0xff263447},muted{0xff91a5bc},blue{0xff82cfff};
const char* views[]{"Waveform","Spectrum","Spectrogram"};
juce::Colour heat(float value) {
    value=juce::jlimit(0.0f,1.0f,value);
    if(value<.35f)return background.interpolatedWith(juce::Colour(0xff204365),value/.35f);
    if(value<.75f)return juce::Colour(0xff204365).interpolatedWith(juce::Colour(0xff4faaf1),(value-.35f)/.4f);
    return juce::Colour(0xff4faaf1).interpolatedWith(juce::Colour(0xffe4f7ff),(value-.75f)/.25f);
}
}

AudioDisplay::AudioDisplay(orbit::OutputQueue& q,OrbitLook& l,std::atomic<int>& m,const juce::String& name)
    :Button(name+" visualization"),queue(q),look(l),mode(m),caption(name) {
    setTitle(name+" visualization");setWantsKeyboardFocus(true);setMouseCursor(juce::MouseCursor::PointingHandCursor);
    spectrogram.clear(spectrogram.getBounds(),background);
    onClick=[this]{mode.store((mode.load()+1)%3);updateHelp();repaint();};
    updateHelp();refresh();startTimerHz(30);
}
void AudioDisplay::updateHelp() {
    if(helpMode==mode.load())return;
    helpMode=mode.load();
    const auto view=juce::String(views[juce::jlimit(0,2,mode.load())]);
    setDescription(view+". Click or press Enter/Space to cycle Waveform, Spectrum, Spectrogram.");
    setTooltip(caption+" / "+view+"\nClick to cycle Waveform, Spectrum and Spectrogram. Spectrum: low to high frequency. Spectrogram: time moves right, frequency rises upward; brighter means louder. Both sides use the same scales.");
}
void AudioDisplay::timerCallback() { refresh(); }
void AudioDisplay::refresh() {
    orbit::OutputFrame frame;bool received=false;
    while(queue.pop(frame)) {
        if(generation!=frame.generation || rate!=frame.sampleRate) {
            generation=frame.generation;rate=frame.sampleRate;waveform.fill({});partial={};waveHead=waveCount=0;
            levels.fill(0);spectrogram.clear(spectrogram.getBounds(),background);spectroHead=0;emptyColumns=192;
        }
        analyzer.append(frame);received=true;
        const int stride=std::max(1,static_cast<int>(rate/256));
        for(size_t i=0;i<frame.left.size();++i) {
            partial.minL=std::min(partial.minL,frame.left[i]);partial.maxL=std::max(partial.maxL,frame.left[i]);
            partial.minR=std::min(partial.minR,frame.right[i]);partial.maxR=std::max(partial.maxR,frame.right[i]);
            if(++waveCount>=stride){waveform[static_cast<size_t>(waveHead)]=partial;waveHead=(waveHead+1)%256;partial={};waveCount=0;}
        }
    }
    const double now=juce::Time::getMillisecondCounterHiRes();
    bool changed=received || helpMode!=mode.load();
    if(received){analyzer.analyze();lastData=now;emptyColumns=0;waveEmpty=false;}
    const bool stale=now-lastData>250;
    if(stale && now-lastData>1000 && !waveEmpty){waveform.fill({});waveEmpty=true;changed=true;}
    const float maximum=std::min(20000.0f,static_cast<float>(analyzer.sampleRate()*.5));
    const bool writeColumn=received || (stale && emptyColumns<192);
    for(size_t i=0;i<levels.size();++i) {
        const float low=20*std::pow(maximum/20,static_cast<float>(i)/64),high=20*std::pow(maximum/20,static_cast<float>(i+1)/64);
        const float target=stale?0:juce::jlimit(0.0f,1.0f,(analyzer.level(low,high)+90)/90);
        auto& value=levels[i];const float old=value;value+=(target-value)*(target>value?.65f:.16f);if(value<.001f)value=0;
        changed=changed || std::abs(value-old)>.00001f;
        if(writeColumn)spectrogram.setPixelAt(spectroHead,63-static_cast<int>(i),heat(target));
    }
    if(writeColumn){spectroHead=(spectroHead+1)%192;if(!received)++emptyColumns;}
    updateHelp();
    if(isVisible() && (changed || writeColumn))repaint();
}
void AudioDisplay::paintButton(juce::Graphics& g,bool over,bool) {
    const int view=juce::jlimit(0,2,mode.load());
    const float w=static_cast<float>(getWidth()),top=27,bottom=static_cast<float>(getHeight()-17),height=bottom-top;
    const float maximum=std::min(20000.0f,static_cast<float>(analyzer.sampleRate()*.5));
    g.fillAll(background);g.setFont(look.font(11));g.setColour(muted);g.drawText(caption,0,0,65,18,juce::Justification::left);
    g.setFont(look.font(9));g.setColour(over||hasKeyboardFocus(true)?blue:muted.withAlpha(.75f));
    g.drawText(juce::String(views[view]).toUpperCase()+" >",65,0,getWidth()-65,18,juce::Justification::right);
    if(view==0) {
        bool clipping=false;
        for(int channel=0;channel<2;++channel) {
            const float centre=top+height*(channel==0?.25f:.75f),amplitude=height*.23f;
            g.setColour(line);g.drawHorizontalLine(static_cast<int>(centre),0,w);
            g.setColour(blue.withAlpha(channel==0?1.0f:.65f));
            for(size_t i=0;i<waveform.size();++i) {
                const auto& f=waveform[(static_cast<size_t>(waveHead)+i)%256];
                const float low=channel==0?f.minL:f.minR,high=channel==0?f.maxL:f.maxR,x=static_cast<float>(i)*w/256;
                clipping=clipping || low<=-1 || high>=1;
                g.drawLine(x,centre-juce::jlimit(-1.0f,1.0f,high)*amplitude,x,centre-juce::jlimit(-1.0f,1.0f,low)*amplitude,1);
            }
        }
        if(clipping){g.setColour(juce::Colour(0xfff29ab3));g.fillEllipse(w-5,top,4,4);}
    } else if(view==1) {
        g.setColour(line);for(int i=0;i<3;++i)g.drawHorizontalLine(static_cast<int>(top+height*static_cast<float>(i)/2),0,w);
        juce::Path trace;trace.startNewSubPath(0,bottom-levels[0]*height);
        for(size_t i=1;i<levels.size();++i)trace.lineTo(static_cast<float>(i)*w/63,bottom-levels[i]*height);
        auto fill=trace;fill.lineTo(w,bottom);fill.lineTo(0,bottom);fill.closeSubPath();
        g.setGradientFill(juce::ColourGradient(blue.withAlpha(.24f),0,top,blue.withAlpha(.01f),0,bottom,false));g.fillPath(fill);
        g.setColour(blue.withAlpha(.13f));g.strokePath(trace,juce::PathStrokeType(4));g.setColour(blue);g.strokePath(trace,juce::PathStrokeType(1.1f));
    } else {
        // A fixed-size circular image; no image allocation or pixel shifting per frame.
        const int first=192-spectroHead,split=juce::roundToInt(w*static_cast<float>(first)/192);
        g.setImageResamplingQuality(juce::Graphics::lowResamplingQuality);
        g.drawImage(spectrogram,0,static_cast<int>(top),split,static_cast<int>(height),spectroHead,0,first,64);
        if(spectroHead)g.drawImage(spectrogram,split,static_cast<int>(top),getWidth()-split,static_cast<int>(height),0,0,spectroHead,64);
        g.setFont(look.font(8));g.setColour(muted);g.drawText("20k",2,static_cast<int>(top),30,10,juce::Justification::left);
        g.drawText("20",2,static_cast<int>(bottom)-11,30,10,juce::Justification::left);
    }
    g.setFont(look.font(9));g.setColour(muted);
    g.drawText(view==0?"-1s":view==1?"20":"-6.4s",0,getHeight()-13,40,13,juce::Justification::left);
    g.drawText(view==1?juce::String(maximum/1000,0)+"k":"NOW",getWidth()-40,getHeight()-13,40,13,juce::Justification::right);
    if(view==1)g.drawText("1k",static_cast<int>(w*std::log(50.0f)/std::log(maximum/20))-17,getHeight()-13,35,13,juce::Justification::centred);
    if(hasKeyboardFocus(true)){g.setColour(blue.withAlpha(.7f));g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(.5f),3,1);}
}
