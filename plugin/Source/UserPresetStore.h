#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <cmath>

// Called only by the editor/tests, never by the audio callback. Each save creates
// a separate file, so existing presets are not overwritten by duplicate names.
class UserPresetStore {
public:
    struct Entry { juce::String name; juce::var parameters; };
    explicit UserPresetStore(juce::File folder=defaultFolder()):directory(std::move(folder)) {}
    static juce::File defaultFolder() {
        return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory).getChildFile("ORBIT/Presets");
    }
    static juce::StringArray ids() {
        return {"density","grain","pitch","mix","jitter","spread","lookback","output","ir","strategy","seed","bypass","reverb"};
    }
    static juce::var capture(juce::AudioProcessorValueTreeState& state) {
        auto* object=new juce::DynamicObject;
        for(const auto& id:ids())object->setProperty(id,state.getRawParameterValue(id)->load());
        return juce::var(object);
    }
    static bool valid(const juce::var& values,juce::AudioProcessorValueTreeState& state) {
        if(!values.isObject() || values.getDynamicObject()->getProperties().size()!=ids().size())return false;
        for(const auto& id:ids()) {
            auto* p=state.getParameter(id);const auto v=values.getProperty(id,juce::var());
            if(!p || !(v.isInt()||v.isInt64()||v.isDouble()))return false;
            const double number=static_cast<double>(v);const auto& range=p->getNormalisableRange();
            if(!std::isfinite(number)||number<range.start-.00001||number>range.end+.00001)return false;
            if(std::abs(range.snapToLegalValue(static_cast<float>(number))-number)>.0001)return false;
        }
        return true;
    }
    static bool matches(const Entry& entry,juce::AudioProcessorValueTreeState& state) {
        for(const auto& id:ids())
            if(std::abs(state.getRawParameterValue(id)->load()-static_cast<float>(entry.parameters.getProperty(id,juce::var())))>.0001f)return false;
        return true;
    }
    static bool apply(const Entry& entry,juce::AudioProcessorValueTreeState& state) {
        if(!valid(entry.parameters,state))return false;
        for(const auto& id:ids()) {
            auto* parameter=state.getParameter(id);
            parameter->beginChangeGesture();
            parameter->setValueNotifyingHost(parameter->convertTo0to1(static_cast<float>(entry.parameters.getProperty(id,juce::var()))));
            parameter->endChangeGesture();
        }
        return true;
    }
    std::vector<Entry> list(juce::AudioProcessorValueTreeState& state) const {
        std::vector<Entry> entries;
        for(const auto& file:directory.findChildFiles(juce::File::findFiles,false,"*.orbitpreset")) {
            if(file.getSize()>32768)continue;
            const auto data=juce::JSON::parse(file.loadFileAsString());
            if(data.getProperty("format",juce::var()).toString()!="ORBIT_PRESET" || static_cast<int>(data.getProperty("version",0))!=1)continue;
            const auto name=data.getProperty("name",juce::var()).toString().trim();
            const auto values=data.getProperty("parameters",juce::var());
            if(name.isEmpty()||name.length()>80||!valid(values,state))continue;
            entries.push_back({name,values});
        }
        std::sort(entries.begin(),entries.end(),[](const Entry& a,const Entry& b){return a.name.compareNatural(b.name)<0;});
        return entries;
    }
    juce::Result save(juce::String name,juce::AudioProcessorValueTreeState& state,juce::String& savedName) const {
        name=name.trim();
        if(name.isEmpty()||name.length()>64||name.containsAnyOf("\r\n\t"))return juce::Result::fail("Use a name with 1–64 characters.");
        const auto existing=list(state);
        if(existing.size()>=256)return juce::Result::fail("The user preset library is full (256 presets).");
        juce::StringArray names;for(const auto& p:existing)names.add(p.name);
        auto unique=name;int suffix=2;
        while(names.contains(unique,true))unique=name+" ("+juce::String(suffix++)+")";
        auto* object=new juce::DynamicObject;
        const juce::var data(object);
        object->setProperty("format","ORBIT_PRESET");object->setProperty("version",1);
        object->setProperty("name",unique);object->setProperty("parameters",capture(state));
        if(!valid(data.getProperty("parameters",juce::var()),state))return juce::Result::fail("Cannot save invalid parameter values.");
        auto result=directory.createDirectory();if(result.failed())return juce::Result::fail("Cannot create the preset folder.");
        const auto destination=directory.getChildFile(juce::Uuid().toString()+".orbitpreset");
        juce::TemporaryFile temporary(destination);
        if(!temporary.getFile().replaceWithText(juce::JSON::toString(data)) || !temporary.overwriteTargetFileWithTemporary())
            return juce::Result::fail("Could not write the preset. Check folder permissions.");
        savedName=unique;return juce::Result::ok();
    }
private:
    juce::File directory;
};
