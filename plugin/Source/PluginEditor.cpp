#include "PluginEditor.h"
#include "ParameterHelp.h"
#include <BinaryData.h>
#include <cmath>

namespace {
const juce::Colour background{0xff0c1118}, line{0xff263447}, text{0xffdfebf8}, muted{0xff91a5bc};
const juce::Colour blue{0xff82cfff}, cyan{0xffa4deee};
struct Preset { const char* name; float density,grain,pitch,mix,jitter,spread; int ir,strategy; };
const std::array<Preset,5> presets{{
    {"Original-like",100,35,1,.3f,.02f,.2f,0,0},
    {"Airy shimmer",80,15,5,.65f,.08f,.6f,1,3},
    {"Warm body",60,25,2,.5f,.05f,.3f,2,4},
    {"Sparse sparkles",30,12,7,.7f,.12f,.9f,0,2},
    {"Percussive microroom",50,10,0,.4f,.04f,.4f,0,0}
}};
const std::array<const char*,13> parameterIds{{"density","grain","pitch","mix","jitter","spread","lookback","output","ir","strategy","seed","bypass","reverb"}};
}

OrbitLook::OrbitLook() {
    face=juce::Typeface::createSystemTypefaceFor(BinaryData::oxanium_ttf,BinaryData::oxanium_ttfSize);
    setColour(juce::ComboBox::backgroundColourId,background);
    setColour(juce::ComboBox::textColourId,text);
    setColour(juce::ComboBox::outlineColourId,line);
    setColour(juce::ComboBox::arrowColourId,muted);
    setColour(juce::PopupMenu::backgroundColourId,background);
    setColour(juce::PopupMenu::textColourId,text);
    setColour(juce::PopupMenu::highlightedBackgroundColourId,line);
    setColour(juce::TextButton::buttonColourId,background);
    setColour(juce::TextButton::textColourOffId,muted);
    setColour(juce::ToggleButton::textColourId,muted);
    setColour(juce::ToggleButton::tickColourId,blue);
    setColour(juce::Slider::textBoxTextColourId,text);
    setColour(juce::Slider::textBoxOutlineColourId,juce::Colours::transparentBlack);
    setColour(juce::Slider::textBoxBackgroundColourId,background);
    setColour(juce::TooltipWindow::backgroundColourId,juce::Colour(0xff121c29));
    setColour(juce::TooltipWindow::textColourId,text);
    setColour(juce::TooltipWindow::outlineColourId,line);
}
juce::Font OrbitLook::font(float size) const { return juce::Font(juce::FontOptions(face).withHeight(size)); }
void OrbitLook::drawLinearSlider(juce::Graphics& g,int x,int y,int width,int height,float position,float,float,juce::Slider::SliderStyle,juce::Slider& slider) {
    const float mid=static_cast<float>(y)+static_cast<float>(height)*.5f;
    const float start=static_cast<float>(x),end=start+static_cast<float>(width);
    const bool active=slider.isMouseOverOrDragging() || slider.hasKeyboardFocus(true);
    const auto accent=slider.isEnabled()?blue:muted;
    // A recessed rail, understated divisions and a grippable illuminated cap.
    for(int i=0;i<=10;++i) {
        const float tick=start+static_cast<float>(i)*static_cast<float>(width)/10;
        g.setColour(tick<=position?accent.withAlpha(.35f):line);
        g.drawLine(tick,mid+9,tick,mid+(i%5==0?13.0f:11.0f),.7f);
    }
    g.setColour(juce::Colour(0xff070c13));g.fillRoundedRectangle(start-1,mid-3,end-start+2,6,3);
    g.setColour(line);g.fillRoundedRectangle(start,mid-1.5f,end-start,3,1.5f);
    if(active) {g.setColour(accent.withAlpha(.12f));g.fillRoundedRectangle(start,mid-4,position-start,8,4);}
    g.setGradientFill(juce::ColourGradient(accent.withAlpha(.4f),start,mid,accent,position,mid,false));
    g.fillRoundedRectangle(start,mid-1,position-start,2,1);
    if(active) {g.setColour(accent.withAlpha(.09f));g.fillRoundedRectangle(position-12,mid-14,24,28,8);}
    const juce::Rectangle<float> cap(position-7,mid-10,14,20);
    g.setColour(juce::Colours::black.withAlpha(.5f));g.fillRoundedRectangle(cap.translated(0,2),4);
    g.setGradientFill(juce::ColourGradient(juce::Colour(0xff24374a),position,mid-10,background,position,mid+10,false));
    g.fillRoundedRectangle(cap,4);
    g.setColour(active?accent:juce::Colour(0xff58748d));g.drawRoundedRectangle(cap.reduced(.5f),3.5f,1);
    g.setColour(text);g.fillRoundedRectangle(position-1,mid-5,2,10,1);
}

void OrbitLook::drawRotarySlider(juce::Graphics& g,int x,int y,int width,int height,float position,float start,float end,juce::Slider& slider) {
    const auto bounds=juce::Rectangle<float>(static_cast<float>(x),static_cast<float>(y),static_cast<float>(width),static_cast<float>(height)).reduced(7);
    const auto centre=bounds.getCentre();const float radius=std::min(bounds.getWidth(),bounds.getHeight())*.5f;
    const auto colour=juce::Colours::white.interpolatedWith(blue,std::sqrt(position));
    juce::Path track,arc;
    track.addCentredArc(centre.x,centre.y,radius,radius,0,start,end,true);
    arc.addCentredArc(centre.x,centre.y,radius,radius,0,start,start+position*(end-start),true);
    g.setColour(line);g.strokePath(track,juce::PathStrokeType(2));
    if(position>0) {g.setColour(blue.withAlpha(position*.15f));g.strokePath(arc,juce::PathStrokeType(7));}
    g.setColour(colour);g.strokePath(arc,juce::PathStrokeType(2));
    g.setColour(slider.isMouseOverOrDragging()?line:background);g.fillEllipse(bounds.reduced(5));
    const float angle=start+position*(end-start)-orbit::pi*.5f;
    g.setColour(colour);g.drawLine(centre.x+std::cos(angle)*radius*.38f,centre.y+std::sin(angle)*radius*.38f,centre.x+std::cos(angle)*radius*.68f,centre.y+std::sin(angle)*radius*.68f,2);
}

InputScope::InputScope(orbit::ScopeQueue& q,OrbitLook& l):queue(q),look(l) { startTimerHz(30); }
void InputScope::timerCallback() {
    orbit::ScopeFrame frame;
    bool changed=false;
    while(queue.pop(frame)) { history[head]=frame; head=(head+1)%history.size(); changed=true; }
    const auto now=juce::Time::getMillisecondCounterHiRes();
    if(changed) { lastFrame=now; stale=false; repaint(); }
    else if(!stale && now-lastFrame>1000) { history.fill({}); stale=true; repaint(); }
}
void InputScope::paint(juce::Graphics& g) {
    g.fillAll(background);
    g.setFont(look.font(11)); g.setColour(muted); g.drawText("INPUT",0,0,getWidth(),18,juce::Justification::left);
    const float w=static_cast<float>(getWidth()),height=static_cast<float>(getHeight()-28),top=26;
    bool clip=false;
    for(int channel=0;channel<2;++channel) {
        const float centre=top+height*(channel==0?.25f:.75f),amplitude=height*.21f;
        g.setColour(line); g.drawHorizontalLine(static_cast<int>(centre),0,w);
        g.setColour(channel==0?blue:cyan.withAlpha(.7f));
        for(size_t i=0;i<history.size();++i) {
            const auto& f=history[(head+i)%history.size()];
            const float low=channel==0?f.minL:f.minR,high=channel==0?f.maxL:f.maxR;
            clip=clip || low<=-1 || high>=1;
            const float x=static_cast<float>(i)*w/static_cast<float>(history.size());
            g.drawLine(x,centre-juce::jlimit(-1.0f,1.0f,high)*amplitude,x,centre-juce::jlimit(-1.0f,1.0f,low)*amplitude,1);
        }
    }
    if(clip) { g.setColour(juce::Colour{0xfff29ab3}); g.fillEllipse(w-6,5,4,4); }
}

OrbitPad::OrbitPad(juce::AudioProcessorValueTreeState& s):state(s) {
    visual=targets();setBufferedToImage(true);
    setMouseCursor(juce::MouseCursor::CrosshairCursor);
    setTitle("XY: density and pitch scatter");
    setDescription("Use the Density and Pitch sliders for keyboard control.");
}
OrbitPad::~OrbitPad() { finish(); }
void OrbitPad::finish() {
    if(dragging) { state.getParameter("density")->endChangeGesture(); state.getParameter("pitch")->endChangeGesture(); dragging=false; }
}
void OrbitPad::mouseDown(const juce::MouseEvent& event) {
    if(!event.mods.isLeftButtonDown()) return;
    finish(); dragging=true;
    state.getParameter("density")->beginChangeGesture(); state.getParameter("pitch")->beginChangeGesture(); move(event.position);
}
void OrbitPad::mouseDrag(const juce::MouseEvent& event) { if(dragging) move(event.position); }
void OrbitPad::mouseUp(const juce::MouseEvent& event) { finish();mouseMove(event); }
void OrbitPad::mouseMove(const juce::MouseEvent& event) {
    const auto target=targets();
    const juce::Point<float> handle(static_cast<float>(getWidth())*(.2f+target[0]*.6f),static_cast<float>(getHeight())*(.8f-target[2]*.6f));
    handleHovered=event.position.getDistanceFrom(handle)<22;
    setMouseCursor(handleHovered?juce::MouseCursor::PointingHandCursor:juce::MouseCursor::CrosshairCursor);
}
void OrbitPad::mouseExit(const juce::MouseEvent&) { handleHovered=false; }
void OrbitPad::move(juce::Point<float> p) {
    state.getParameter("density")->setValueNotifyingHost(juce::jlimit(0.0f,1.0f,(p.x/static_cast<float>(getWidth())-.2f)/.6f));
    state.getParameter("pitch")->setValueNotifyingHost(juce::jlimit(0.0f,1.0f,(.8f-p.y/static_cast<float>(getHeight()))/.6f));
    repaint();
}
std::array<float,5> OrbitPad::targets() const {
    auto value=[this](const char* id){return state.getRawParameterValue(id)->load();};
    const bool bypassed=value("bypass")>.5f;
    return {{(value("density")-10)/110,(value("grain")-5)/45,value("pitch")/12,value("mix"),bypassed?0:value("reverb")}};
}
void OrbitPad::updateVisuals() {
    const auto target=targets();bool changed=false;
    const float activity=dragging?1.0f:(handleHovered?.55f:0.0f);
    if(handleActivity!=activity) {
        handleActivity+=(activity-handleActivity)*.38f;
        if(std::abs(activity-handleActivity)<.001f)handleActivity=activity;
        changed=true;
    }
    for(size_t i=0;i<visual.size();++i) if(visual[i]!=target[i]) {
        visual[i]+=(target[i]-visual[i])*.38f;
        if(std::abs(visual[i]-target[i])<.001f)visual[i]=target[i];
        changed=true;
    }
    if(changed)repaint();
}
void OrbitPad::paint(juce::Graphics& g) {
    const float w=static_cast<float>(getWidth()),h=static_cast<float>(getHeight()),s=std::min(w,h),cx=w*.5f,cy=h*.5f;
    const float x=visual[0],grainSize=visual[1],y=visual[2],wet=visual[3],space=visual[4];
    const auto colour=juce::Colours::white.interpolatedWith(juce::Colour(0xff69bfff),std::sqrt(space));
    g.setColour(line); g.drawEllipse(cx-s*.414f,cy-s*.414f,s*.828f,s*.828f,.6f);
    for(int i=0;i<80;++i) {
        const float a=static_cast<float>(i)*orbit::pi/40,inner=s*(i%5==0?.425f:.431f),outer=s*.438f;
        g.drawLine(cx+std::cos(a)*inner,cy+std::sin(a)*inner,cx+std::cos(a)*outer,cy+std::sin(a)*outer,.6f);
    }
    if(space>0) {
        juce::ColourGradient glow(blue.withAlpha(0.0f),cx,cy,blue.withAlpha(0.0f),cx+s*.46f,cy,true);
        glow.addColour(.48,blue.withAlpha(space*.025f));
        glow.addColour(.69,blue.withAlpha(space*.14f));
        glow.addColour(.86,blue.withAlpha(space*.025f));
        g.setGradientFill(glow);g.fillEllipse(cx-s*.46f,cy-s*.46f,s*.92f,s*.92f);
    }
    const float rings=28+x*36;
    for(int ring=0;ring<64;++ring) {
        const float visibility=juce::jlimit(0.0f,1.0f,rings-static_cast<float>(ring));
        if(visibility==0)break;
        const float t=static_cast<float>(ring)/std::max(1.0f,rings-1),phase=t*2*orbit::pi;
        const float radius=s*(.272f+(t-.5f)*(.085f+.13f*grainSize));
        juce::Path curve;
        for(int j=0;j<=160;++j) {
            const float a=static_cast<float>(j)/160*2*orbit::pi;
            const float wave=(std::sin(a*3+phase*1.4f+x*2)*.019f+std::sin(a*5-phase+y*3)*.012f)*s*(.25f+y*1.3f);
            const float twist=a+std::sin(phase+a*2)*(.035f+x*.075f);
            const float px=cx+std::cos(twist)*(radius+wave)+std::sin(phase)*s*.012f;
            const float py=cy+std::sin(twist)*(radius+wave)*(.95f+.03f*std::cos(phase));
            if(j==0) curve.startNewSubPath(px,py); else curve.lineTo(px,py);
        }
        const float alpha=(.20f+.48f*std::sin(t*orbit::pi))*(.65f+.35f*wet)*visibility;
        if(space>0 && ring%5==0) {
            g.setColour(blue.withAlpha(alpha*space*.055f));g.strokePath(curve,juce::PathStrokeType(12+space*8));
            g.setColour(blue.withAlpha(alpha*space*.14f));g.strokePath(curve,juce::PathStrokeType(3));
        }
        g.setColour(colour.withAlpha(alpha));g.strokePath(curve,juce::PathStrokeType(.65f+.25f*wet));
    }
    g.setColour(muted.withAlpha(.4f)); g.drawLine(cx-3,cy,cx+3,cy,.7f); g.drawLine(cx,cy-3,cx,cy+3,.7f);
    const auto target=targets();const float hx=w*(.2f+target[0]*.6f),hy=h*(.8f-target[2]*.6f);
    const float activity=handleActivity,radius=21+activity*7;
    juce::ColourGradient bloom(colour.withAlpha(.38f+activity*.13f),hx,hy,colour.withAlpha(0.0f),hx+radius,hy,true);
    bloom.addColour(.20,colour.withAlpha(.22f+activity*.12f));
    bloom.addColour(.48,colour.withAlpha(.07f+activity*.05f));
    bloom.addColour(.76,colour.withAlpha(.016f));
    g.setGradientFill(bloom);g.fillEllipse(hx-radius,hy-radius,radius*2,radius*2);
    // Fine interrupted locator ring, separated from the luminous core.
    const float locator=9.5f+activity*2;
    for(int quadrant=0;quadrant<4;++quadrant) {
        const float start=static_cast<float>(quadrant)*orbit::pi*.5f+.22f;
        juce::Path arc;arc.addCentredArc(hx,hy,locator,locator,0,start,start+.94f,true);
        g.setColour(colour.withAlpha(.28f+activity*.25f));g.strokePath(arc,juce::PathStrokeType(.7f));
    }
    const float core=4.0f+activity*.5f;
    g.setGradientFill(juce::ColourGradient(juce::Colours::white,hx-2,hy-3,colour.withMultipliedBrightness(.8f),hx+3,hy+4,false));
    g.fillEllipse(hx-core,hy-core,core*2,core*2);
    g.setColour(juce::Colours::white.withAlpha(.8f));g.drawEllipse(hx-core,hy-core,core*2,core*2,.65f);
    g.setColour(juce::Colours::white.withAlpha(.95f));g.fillEllipse(hx-2,hy-2.5f,2.4f,2.4f);
}

OrbitEditor::OrbitEditor(OrbitProcessor& p):AudioProcessorEditor(p),processor(p),scope(p.engine.scope,look),pad(p.state) {
    setLookAndFeel(&look); setOpaque(true);
    auto attach=[&](juce::Slider& slider,const char* id,const char* name,bool small=false) {
        slider.setSliderStyle(juce::Slider::LinearHorizontal);
        slider.setTextBoxStyle(small?juce::Slider::TextBoxRight:juce::Slider::NoTextBox,false,58,24);
        slider.setTitle(name); slider.setName(name); slider.setNumDecimalPlacesToDisplay(2);
        slider.setMouseDragSensitivity(240);
        slider.setScrollWheelEnabled(false);
        slider.setVelocityModeParameters(.3,1,0.0,true,juce::ModifierKeys::shiftModifier);
        if(auto* parameter=p.state.getParameter(id))
            slider.setDoubleClickReturnValue(true,parameter->convertFrom0to1(parameter->getDefaultValue()));
        slider.setTooltip(orbitParameterHelp(id));
        slider.setDescription(orbitParameterHelp(id));
        addAndMakeVisible(slider);
        sliders.push_back(std::make_unique<SliderAttachment>(p.state,id,slider));
    };
    attach(density,"density","Density"); attach(grain,"grain","Grain size");
    attach(pitch,"pitch","Pitch scatter"); attach(mix,"mix","Dry / Wet");
    attach(reverb,"reverb","Reverb");reverb.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    reverb.setRotaryParameters(orbit::pi*1.25f,orbit::pi*2.75f,true);
    attach(jitter,"jitter","Trigger jitter",true); attach(spread,"spread","Stereo spread",true);
    attach(lookback,"lookback","Lookback",true); attach(output,"output","Output gain",true); attach(seed,"seed","Random seed",true);
    grain.setTextValueSuffix(" ms"); pitch.setTextValueSuffix(" st"); lookback.setTextValueSuffix(" ms"); output.setTextValueSuffix(" dB");
    ir.addItemList({"8 ms","16 ms","24 ms","32 ms"},1);
    strategy.addItemList({"Fixed","Cycle","Random","Weighted","Centroid"},1);
    ir.setTitle("IR length"); strategy.setTitle("IR selection"); preset.setTitle("Preset");
    ir.setTooltip(orbitParameterHelp("ir"));strategy.setTooltip(orbitParameterHelp("strategy"));
    bypass.setTooltip(orbitParameterHelp("bypass"));pad.setTooltip(orbitParameterHelp("xy"));
    scope.setTooltip(orbitParameterHelp("input"));
    preset.setTextWhenNothingSelected("Custom");
    for(auto* c:std::initializer_list<juce::Component*>{&scope,&pad,&preset,&ir,&strategy,&previous,&next,&details,&bypass,&savePreset}) addAndMakeVisible(c);
    addChildComponent(savePanel);
    for(auto* c:std::initializer_list<juce::Component*>{&presetName,&savePrompt,&saveError,&confirmSave,&cancelSave})savePanel.addAndMakeVisible(c);
    savePrompt.setText("Save preset",juce::dontSendNotification);
    saveError.setColour(juce::Label::textColourId,juce::Colour(0xfff29ab3));
    presetName.setFont(look.font(15));presetName.setInputRestrictions(64);presetName.setTitle("Preset name");
    presetName.setColour(juce::TextEditor::backgroundColourId,background);
    presetName.setColour(juce::TextEditor::textColourId,text);
    presetName.setColour(juce::TextEditor::focusedOutlineColourId,blue);
    savePreset.setTitle("Save preset");
    savePreset.onClick=[this]{showSavePreset();};confirmSave.onClick=[this]{commitPreset();};
    cancelSave.onClick=[this]{savePanel.setVisible(false);savePreset.grabKeyboardFocus();};
    presetName.onReturnKey=[this]{commitPreset();};presetName.onEscapeKey=cancelSave.onClick;
    irAttachment=std::make_unique<ComboAttachment>(p.state,"ir",ir);
    strategyAttachment=std::make_unique<ComboAttachment>(p.state,"strategy",strategy);
    bypassAttachment=std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(p.state,"bypass",bypass);
    preset.onChange=[this]{if(preset.getSelectedId()>0) applyPreset(preset.getSelectedId()-1);};
    preset.onOpen=[this]{refreshPresets();};
    previous.onClick=[this]{refreshPresets();const int count=preset.getNumItems();applyPreset(preset.getItemId((std::max(0,preset.getSelectedItemIndex())+count-1)%count)-1);};
    next.onClick=[this]{refreshPresets();applyPreset(preset.getItemId((preset.getSelectedItemIndex()+1)%preset.getNumItems())-1);};
    details.onClick=[this]{setExpanded(!expanded);};
    setResizable(true,true); setExpanded(false); setSize(1000,620);
    refreshPresets(); startTimerHz(30);
}
OrbitEditor::~OrbitEditor() { stopTimer(); setLookAndFeel(nullptr); }
juce::String OrbitEditor::getTooltip() {
    // Parameter names and large readouts are painted on the editor itself.
    const auto point=getMouseXYRelative().toFloat()*(1000.0f/static_cast<float>(getWidth()));
    const auto hit=[&](int x,int y,int w,int h){return juce::Rectangle<float>(static_cast<float>(x),static_cast<float>(y),static_cast<float>(w),static_cast<float>(h)).contains(point);};
    if(expanded) {
        if(hit(40,238,265,57))return orbitParameterHelp("jitter");
        if(hit(40,329,265,57))return orbitParameterHelp("spread");
        if(hit(40,420,265,57))return orbitParameterHelp("seed");
        if(hit(368,238,265,62))return orbitParameterHelp("ir");
        if(hit(368,329,265,59))return orbitParameterHelp("strategy");
        if(hit(368,420,265,57))return orbitParameterHelp("lookback");
        if(hit(690,238,268,57))return orbitParameterHelp("output");
        if(hit(690,329,268,72))return orbitParameterHelp("reverb");
        return {};
    }
    if(hit(40,255,220,102))return orbitParameterHelp("density");
    if(hit(40,380,220,63))return orbitParameterHelp("grain");
    if(hit(735,169,220,103))return orbitParameterHelp("pitch");
    if(hit(735,310,220,133))return orbitParameterHelp("mix");
    if(hit(440,503,170,58))return orbitParameterHelp("reverb");
    return {};
}
void OrbitEditor::setExpanded(bool open) {
    expanded=open;
    for(auto* c:std::initializer_list<juce::Component*>{&jitter,&spread,&lookback,&output,&seed,&ir,&strategy}) c->setVisible(open);
    for(auto* c:std::initializer_list<juce::Component*>{&scope,&pad,&density,&grain,&pitch,&mix}) c->setVisible(!open);
    details.setButtonText(open?"< Back":"Details >");
    const int height=620;
    setResizeLimits(800,height*8/10,1600,height*16/10);
    getConstrainer()->setFixedAspectRatio(1000.0/height);
    if(getWidth()>0) setSize(getWidth(),getWidth()*height/1000);
    resized(); repaint();
}
void OrbitEditor::applyPreset(int index) {
    if(index>=1000) {
        const auto i=static_cast<size_t>(index-1000);
        if(i<userPresets.size())UserPresetStore::apply(userPresets[i],processor.state);
        syncPreset();repaint();return;
    }
    const auto& p=presets[static_cast<size_t>(juce::jlimit(0,4,index))];
    const std::array<std::pair<const char*,float>,13> targets{{{"density",p.density},{"grain",p.grain},{"pitch",p.pitch},{"mix",p.mix},{"jitter",p.jitter},{"spread",p.spread},{"ir",static_cast<float>(p.ir)},{"strategy",static_cast<float>(p.strategy)},{"reverb",0},{"lookback",40},{"output",-3},{"seed",2025},{"bypass",0}}};
    for(const auto& target:targets) { auto* parameter=processor.state.getParameter(target.first); parameter->beginChangeGesture(); parameter->setValueNotifyingHost(parameter->convertTo0to1(target.second)); parameter->endChangeGesture(); }
    syncPreset(); repaint(); pad.repaint();
}
void OrbitEditor::syncPreset() {
    const int current=preset.getSelectedId()-1001;
    if(current>=0 && static_cast<size_t>(current)<userPresets.size() && UserPresetStore::matches(userPresets[static_cast<size_t>(current)],processor.state))return;
    int selected=0;
    auto matches=[&](const char* id,float v){return std::abs(processor.state.getRawParameterValue(id)->load()-v)<.002f;};
    for(size_t i=0;i<presets.size();++i) {
        const auto& p=presets[i];
        if(matches("density",p.density)&&matches("grain",p.grain)&&matches("pitch",p.pitch)&&matches("mix",p.mix)&&matches("jitter",p.jitter)&&matches("spread",p.spread)&&matches("ir",static_cast<float>(p.ir))&&matches("strategy",static_cast<float>(p.strategy))&&matches("reverb",0)&&matches("lookback",40)&&matches("output",-3)&&matches("seed",2025)&&matches("bypass",0)) selected=static_cast<int>(i)+1;
    }
    for(size_t i=0;i<userPresets.size();++i)if(UserPresetStore::matches(userPresets[i],processor.state))selected=1001+static_cast<int>(i);
    preset.setSelectedId(selected,juce::dontSendNotification);
}
void OrbitEditor::refreshPresets() {
    const auto previousName=preset.getSelectedId()>=1001?preset.getText():juce::String();
    userPresets=presetStore.list(processor.state);
    preset.clear(juce::dontSendNotification);preset.addSectionHeading("Factory");
    for(size_t i=0;i<presets.size();++i)preset.addItem(presets[i].name,static_cast<int>(i)+1);
    if(!userPresets.empty())preset.addSectionHeading("User");
    for(size_t i=0;i<userPresets.size();++i)preset.addItem(userPresets[i].name,1001+static_cast<int>(i));
    for(size_t i=0;i<userPresets.size();++i)if(userPresets[i].name==previousName)preset.setSelectedId(1001+static_cast<int>(i),juce::dontSendNotification);
    syncPreset();
}
void OrbitEditor::showSavePreset() {
    saveError.setText({},juce::dontSendNotification);
    presetName.setText(preset.getSelectedId()>=1001?preset.getText():"My preset",false);
    savePanel.setVisible(true);savePanel.toFront(false);presetName.grabKeyboardFocus();presetName.selectAll();
}
void OrbitEditor::commitPreset() {
    juce::String savedName;const auto result=presetStore.save(presetName.getText(),processor.state,savedName);
    if(result.failed()){saveError.setText(result.getErrorMessage(),juce::dontSendNotification);return;}
    refreshPresets();
    for(size_t i=0;i<userPresets.size();++i)if(userPresets[i].name==savedName)preset.setSelectedId(1001+static_cast<int>(i),juce::dontSendNotification);
    savePanel.setVisible(false);savePreset.grabKeyboardFocus();
}
void OrbitEditor::timerCallback() {
    bool changed=false;
    for(size_t i=0;i<parameterIds.size();++i) {
        const auto v=processor.state.getRawParameterValue(parameterIds[i])->load(std::memory_order_relaxed);
        if(v!=previousValues[i]) { previousValues[i]=v; changed=true; }
    }
    if(changed) { syncPreset(); repaint(); }
    pad.updateVisuals();
}
void OrbitEditor::resized() {
    const float s=static_cast<float>(getWidth())/1000;
    auto place=[&](juce::Component& c,int x,int y,int w,int h){c.setBounds(juce::Rectangle<float>(static_cast<float>(x)*s,static_cast<float>(y)*s,static_cast<float>(w)*s,static_cast<float>(h)*s).toNearestInt());};
    place(preset,390,28,220,32); place(previous,344,28,30,32); place(next,626,28,30,32); place(bypass,865,28,95,32);
    place(savePreset,675,28,60,32);place(savePanel,320,92,360,160);
    const float panelScale=static_cast<float>(savePanel.getWidth())/360;
    auto inPanel=[&](juce::Component& c,int x,int y,int w,int h){c.setBounds(juce::roundToInt(x*panelScale),juce::roundToInt(y*panelScale),juce::roundToInt(w*panelScale),juce::roundToInt(h*panelScale));};
    inPanel(savePrompt,18,10,320,24);inPanel(presetName,20,42,320,32);inPanel(saveError,18,78,324,28);
    inPanel(cancelSave,183,117,72,28);inPanel(confirmSave,268,117,72,28);
    place(scope,45,116,210,106); place(pad,288,99,420,388);
    place(density,40,330,220,26); place(grain,40,416,220,26);
    place(pitch,735,245,220,26); place(mix,735,416,220,26);
    place(details,40,515,100,28);
    place(reverb,445,503,58,58);
    place(jitter,40,265,265,30); place(spread,40,356,265,30);place(seed,40,447,265,30);
    place(lookback,368,447,265,30); place(output,690,265,268,30);
    place(ir,368,268,245,32); place(strategy,368,356,245,32);
    if(expanded) {place(details,40,112,90,28);place(reverb,690,340,58,58);}
}
void OrbitEditor::paint(juce::Graphics& graphics) {
    graphics.fillAll(background);
    auto& g=graphics; g.addTransform(juce::AffineTransform::scale(static_cast<float>(getWidth())/1000));
    auto label=[&](const juce::String& str,float x,float y,float w,float h,float size,juce::Colour colour=text,juce::Justification align=juce::Justification::left){g.setFont(look.font(size));g.setColour(colour);g.drawText(str,juce::Rectangle<float>(x,y,w,h),align);};
    g.setColour(blue); g.drawEllipse(45,33,21,21,1.4f);g.drawEllipse(49,37,13,13,1);
    label("O R B I T",82,25,225,38,26);
    g.setColour(line);g.drawHorizontalLine(82,40,960);
    if(expanded) {
        label("DETAILS",155,108,220,36,26);
        label("GRAIN MOTION",45,193,240,20,11,blue);
        label("CONVOLUTION",373,193,240,20,11,blue);
        label("SPACE / OUTPUT",695,193,260,20,11,blue);
        g.setColour(line);g.drawHorizontalLine(224,40,305);g.drawHorizontalLine(224,368,633);g.drawHorizontalLine(224,690,958);
        label("JITTER",45,242,220,18,10,muted);label("SPREAD",45,333,220,18,10,muted);
        label("SEED",45,424,220,18,10,muted);label("LOOKBACK",373,424,220,18,10,muted);
        label("IR LENGTH",373,242,220,18,10,muted);label("SELECTION",373,333,220,18,10,muted);
        label("OUTPUT",695,242,220,18,10,muted);label("REVERB",763,347,160,18,10,muted);
        label(reverb.getValue()<.0005?"OFF":juce::String(reverb.getValue()*100,0)+"%",763,369,160,24,20,reverb.getValue()>0?blue:text);
        g.setColour(line);g.drawHorizontalLine(539,40,960);
        label("VST3",916,561,44,20,10,muted,juce::Justification::right);
        return;
    }
    g.setColour(line);g.drawHorizontalLine(497,40,960);
    label("DENSITY",45,259,180,20,11,muted); label("X",239,259,16,20,10,muted);
    label(juce::String(density.getValue(),0),45,280,95,48,44);label("grains/s",142,300,75,20,11,muted);
    label("GRAIN SIZE",45,383,125,20,11,muted);label(juce::String(grain.getValue(),1)+" ms",170,383,85,20,11,muted,juce::Justification::right);
    label("PITCH SCATTER",740,172,180,20,11,muted);label("Y",939,172,16,20,10,muted);
    label("+/-",740,207,36,25,16,blue);label(juce::String(pitch.getValue(),1),779,191,132,50,42);label("st",916,215,35,20,11,muted);
    juce::Path arc;arc.addCentredArc(781,353,38,38,0,orbit::pi*1.25f,orbit::pi*(1.25f+1.5f*static_cast<float>(mix.getValue())),true);
    g.setColour(line);g.drawEllipse(743,315,76,76,2);g.setColour(blue);g.strokePath(arc,juce::PathStrokeType(2));
    label(juce::String(mix.getValue()*100,0),751,332,60,34,29,text,juce::Justification::centred);label("%",774,367,20,12,9,muted,juce::Justification::centred);
    label("DRY / WET",840,341,116,24,11,muted);
    label("REVERB",513,513,95,17,10,muted);
    label(reverb.getValue()<.0005?"OFF":juce::String(reverb.getValue()*100,0)+"%",513,532,85,20,15,reverb.getValue()>0?blue:text);
    label("VST3",916,561,44,20,10,muted,juce::Justification::right);
}
