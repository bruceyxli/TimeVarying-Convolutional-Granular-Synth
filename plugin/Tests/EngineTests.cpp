#include "OrbitEngine.h"
#include "OrbitSpectrum.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>

static thread_local bool trackAllocations=false;
static thread_local size_t audioAllocations=0;
void* operator new(std::size_t n) {
    if(trackAllocations) ++audioAllocations;
    if(void* p=std::malloc(n?n:1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void operator delete[](void* p,std::size_t) noexcept { std::free(p); }

void require(bool condition,const std::string& message) { if(!condition) throw std::runtime_error(message); }
using Stereo=std::array<std::vector<float>,2>;
Stereo input(int length,double sr) {
    Stereo audio;
    for(auto& channel:audio) channel.resize(static_cast<size_t>(length));
    for(int i=0;i<length;++i) {
        audio[0][static_cast<size_t>(i)]=.2f*std::sin(static_cast<float>(i)*2*orbit::pi*220/static_cast<float>(sr));
        audio[1][static_cast<size_t>(i)]=.13f*std::sin(static_cast<float>(i)*2*orbit::pi*330/static_cast<float>(sr));
    }
    return audio;
}
void run(orbit::Engine& engine,Stereo& data,const orbit::Parameters& params,const std::vector<int>& blocks) {
    int offset=0,index=0,total=static_cast<int>(data[0].size());
    while(offset<total) {
        const int length=std::min(blocks[static_cast<size_t>(index++)%blocks.size()],total-offset);
        trackAllocations=true;
        engine.process(data[0].data()+offset,data[1].data()+offset,length,params);
        trackAllocations=false;
        offset+=length;
    }
}
void testFFT() {
    orbit::FFT fft; fft.prepare(256);
    std::vector<std::complex<float>> samples(256), original(256);
    for(int i=0;i<256;++i) samples[static_cast<size_t>(i)]={std::sin(static_cast<float>(i)),std::cos(static_cast<float>(i))};
    original=samples; fft.transform(samples.data(),false); fft.transform(samples.data(),true);
    for(size_t i=0;i<samples.size();++i) require(std::abs(samples[i]-original[i])<1e-5f,"FFT roundtrip");
    std::vector<std::complex<float>> a(256),b(256);
    for(int i=0;i<64;++i) a[static_cast<size_t>(i)]=std::sin(static_cast<float>(i))*.1f;
    for(int i=0;i<17;++i) b[static_cast<size_t>(i)]=.05f*static_cast<float>(i-8);
    auto aa=a,bb=b;
    fft.transform(a.data(),false);fft.transform(b.data(),false);
    for(int i=0;i<256;++i) a[static_cast<size_t>(i)]*=b[static_cast<size_t>(i)];
    fft.transform(a.data(),true);
    for(int i=0;i<256;++i) {
        float expected=0;
        for(int j=0;j<64;++j) if(i-j>=0 && i-j<17) expected+=aa[static_cast<size_t>(j)].real()*bb[static_cast<size_t>(i-j)].real();
        require(std::abs(a[static_cast<size_t>(i)].real()-expected)<1e-5f,"FFT convolution including tail");
    }
}
void testBlockInvariance(orbit::Engine& engine) {
    orbit::Parameters p;
    for(int strategy=0;strategy<5;++strategy) {
        p.strategy=strategy;
        auto reference=input(48000,48000),candidate=reference;
        engine.reset();run(engine,reference,p,{48000});
        engine.reset();run(engine,candidate,p,{32,64,128,256,512,1024,3,127});
        for(size_t c=0;c<2;++c) for(size_t i=0;i<reference[c].size();++i)
            require(reference[c][i]==candidate[c][i],"Block-size invariance / deterministic RNG");
    }
}
void testLongConvolution() {
    orbit::LongConvolver convolver;convolver.prepare(48000);
    for(float duration:{40.0f,125.0f,300.0f}) {
        const auto ir=orbit::LongConvolver::makeImpulse(48000,duration);
        std::array<float,91> source{};for(size_t i=0;i<source.size();++i)source[i]=std::sin(static_cast<float>(i)*.37f)*.2f;
        convolver.reset();
        for(int n=0;n<static_cast<int>(ir.size()+source.size())+orbit::LongConvolver::hop+512;++n) {
            float l=n<static_cast<int>(source.size())?source[static_cast<size_t>(n)]:0,r=-l;
            trackAllocations=true;convolver.process(l,r,duration);trackAllocations=false;
            const int index=n-orbit::LongConvolver::hop;double expected=0;
            for(size_t j=0;j<source.size();++j)if(index>=static_cast<int>(j) && index-static_cast<int>(j)<static_cast<int>(ir.size()))expected+=source[j]*ir[static_cast<size_t>(index-static_cast<int>(j))];
            require(std::abs(l-expected)<.00002 && std::abs(r+expected)<.00002,"Partitioned convolution must match direct convolution including latency and tail");
        }
    }
}
void testVariants(orbit::Engine& engine) {
    for(int mode=1;mode<=2;++mode) {
        orbit::Parameters p;p.variant=mode;p.mix=1;p.outputDb=-6;p.longIrMs=300;
        auto reference=input(48000,48000),split=reference;
        engine.reset();run(engine,reference,p,{48000});engine.reset();run(engine,split,p,{1,63,257,1024});
        require(reference==split,"Variant A/B block invariance");
        float energy=0;for(float x:reference[0])energy+=std::abs(x);
        require(energy>1,"Variant must produce a non-silent wet output");
        auto dry=input(24000,48000),original=dry;p.mix=0;p.outputDb=0;
        engine.reset();run(engine,dry,p,{64});require(dry==original,"All modes retain the common dry endpoint");
        Stereo silent;for(auto& c:silent)c.assign(48000,0);p.mix=1;
        engine.reset();run(engine,silent,p,{64});for(auto& c:silent)for(float x:c)require(x==0,"Variants preserve silence");
        p.bypass=true;auto bypass=original;engine.reset();run(engine,bypass,p,{64});require(bypass==original,"All modes bypass to dry unity");
    }
    orbit::Parameters p;p.variant=2;p.mix=1;p.outputDb=-6;
    auto normal=input(24000,48000),inverted=normal;
    for(auto& c:inverted)for(auto& x:c)x=-x;
    engine.reset();run(engine,normal,p,{64});engine.reset();run(engine,inverted,p,{64});
    for(size_t c=0;c<2;++c)for(size_t i=0;i<normal[c].size();++i)require(std::abs(normal[c][i]-inverted[c][i])<1e-6f,"Grains-as-IR must convolve two source-derived signals");
    p.variant=1;p.longIrMs=40;normal=input(48000,48000);inverted=normal;
    engine.reset();run(engine,normal,p,{64});p.longIrMs=300;engine.reset();run(engine,inverted,p,{64});
    double difference=0;for(size_t i=0;i<normal[0].size();++i)difference+=std::abs(normal[0][i]-inverted[0][i]);
    require(difference>.1,"Long IR length must change Variant A audio");
    // Changing modes on silence after earlier activity must not replay old A history.
    p.variant=0;for(auto& c:normal)std::fill(c.begin(),c.end(),0);run(engine,normal,p,{64});
    p.variant=1;for(auto& c:normal)std::fill(c.begin(),c.end(),0);run(engine,normal,p,{64});
    for(auto& c:normal)for(float x:c)require(x==0,"Returning to A must not resurrect stale history");
}
void testSpectrum() {
    orbit::OutputQueue queue;orbit::SpectrumAnalyzer analyzer;
    std::array<float,4096> left{},right{};
    constexpr float frequency=93.0f*48000/4096;
    for(size_t i=0;i<left.size();++i){left[i]=.25f*std::sin(2*orbit::pi*frequency*static_cast<float>(i)/48000);right[i]=-left[i];}
    trackAllocations=true;queue.capture(left.data(),right.data(),4096,48000);trackAllocations=false;
    orbit::OutputFrame frame;while(queue.pop(frame))analyzer.append(frame);analyzer.analyze();
    require(std::abs(analyzer.level(frequency,frequency)+12.0412f)<.1f,"Output FFT must recover a known tone level without stereo phase cancellation");
    require(analyzer.level(8000,10000)<-70,"Spectrum must distinguish bands rather than display amplitude only");
    queue.reset();left.fill(0);right.fill(0);queue.capture(left.data(),right.data(),4096,44100);
    while(queue.pop(frame))analyzer.append(frame);analyzer.analyze();
    require(analyzer.sampleRate()==44100 && analyzer.level(100,20000)==-90,"Spectrum resets on silence and sample-rate changes");
    for(int i=0;i<20;++i){trackAllocations=true;queue.capture(left.data(),right.data(),4096,44100);trackAllocations=false;}
    int frames=0;while(queue.pop(frame))++frames;require(frames==64,"Full spectrum queue must drop display frames with bounded memory");
    queue.capture(left.data(),right.data(),256,44100);require(queue.pop(frame),"Output queue recovery");
}
void testBypassAndSilence(orbit::Engine& engine) {
    orbit::Parameters p;p.mix=0;p.outputDb=0;
    auto reference=input(48000,48000),data=reference;
    engine.reset();run(engine,data,p,{64});
    require(data==reference,"Dry endpoint must be transparent below ceiling");
    p.mix=1; data=reference; engine.reset();run(engine,data,p,{64});
    p.bypass=true; data=reference;run(engine,data,p,{64});
    for(size_t c=0;c<2;++c) for(size_t i=47000;i<48000;++i)
        require(data[c][i]==reference[c][i],"Bypass must reach exact unity");
    Stereo silent;for(auto& c:silent)c.assign(48000,0);
    engine.reset();p.bypass=false;run(engine,silent,p,{64});
    for(const auto& c:silent) for(float sample:c) require(sample==0,"Silence must remain silent");
}
void testStereoAndScope(orbit::Engine& engine) {
    orbit::ScopeFrame ignored;while(engine.scope.pop(ignored)){}
    orbit::Parameters p;p.mix=1;p.spread=0;p.outputDb=0;
    auto data=input(48000,48000);
    for(size_t i=0;i<data[0].size();++i)data[1][i]=-data[0][i];
    engine.reset();run(engine,data,p,{128});
    float peak=0;
    for(size_t i=0;i<data[0].size();++i) { peak=std::max(peak,std::abs(data[0][i]));require(std::abs(data[0][i]+data[1][i])<1e-6f,"Stereo polarity preserved"); }
    require(peak>.005f,"Opposite-phase input must not cancel in wet signal");
    bool received=false;
    while(engine.scope.pop(ignored)) { received=true; require(std::abs(ignored.minL+ignored.maxR)<1e-6f && std::abs(ignored.maxL+ignored.minR)<1e-6f,"Scope stereo separation"); }
    require(received,"Live input scope frames");
    orbit::ScopeQueue queue;
    for(int i=0;i<1024;++i)require(queue.push({}),"Scope queue capacity");
    require(!queue.push({}),"Scope overload must drop visualization data");
    require(queue.pop(ignored) && queue.push({}),"Scope queue recovery");
}
void testRatesAndAutomation() {
    for(double sr:{44100.0,96000.0,192000.0}) {
        orbit::Engine engine;engine.prepare(sr);
        auto audio=input(static_cast<int>(sr/2),sr);
        orbit::Parameters p;p.density=120;p.grainMs=50;p.pitch=12;p.irLength=3;p.reverb=1;
        for(int mode=0;mode<3;++mode) {
            p.variant=mode;p.longIrMs=300;engine.reset();audio=input(static_cast<int>(sr/2),sr);
            run(engine,audio,p,{64,1024,1});
            for(const auto& c:audio)for(float sample:c)require(std::isfinite(sample)&&std::abs(sample)<=1,"All modes at rate extremes remain finite and bounded");
            require(engine.droppedGrains()==0,"Supported range must fit voice pool");
        }
    }
    orbit::Engine e;e.prepare(48000);
    auto audio=input(24000,48000);
    orbit::Parameters p;
    for(int offset=0;offset<24000;offset+=64) {
        p.pitch=(offset%128)?0.0f:12.0f;p.mix=(offset%128)?0.0f:1.0f;p.irLength=(offset/64)%4;
        p.reverb=(offset%128)?0.0f:1.0f;
        p.variant=(offset/256)%3;p.longIrMs=(offset%128)?40.0f:300.0f;
        trackAllocations=true;e.process(audio[0].data()+offset,audio[1].data()+offset,std::min(64,24000-offset),p);trackAllocations=false;
    }
    for(const auto& c:audio)for(float sample:c)require(std::isfinite(sample),"Automation must remain finite");
}
void testReverb(orbit::Engine& engine) {
    orbit::Parameters p;p.mix=0;p.outputDb=0;p.reverb=1;
    Stereo impulse;for(auto& c:impulse)c.assign(144000,0);impulse[0][0]=.4f;
    auto split=impulse;
    engine.reset();run(engine,impulse,p,{144000});
    engine.reset();run(engine,split,p,{1,32,128,509});
    require(impulse==split,"Reverb block invariance");
    float tail=0,end=0,right=0;
    for(size_t i=1000;i<24000;++i) {tail+=std::abs(impulse[0][i]);right+=std::abs(impulse[1][i]);}
    for(size_t i=120000;i<144000;++i)end+=std::abs(impulse[0][i]);
    require(tail>.01f && right>.001f,"Reverb must create a real stereo tail");
    require(end<tail*.001f,"Reverb tail must decay");
    engine.reset();p.reverb=0;auto dry=input(48000,48000),original=dry;run(engine,dry,p,{64});
    require(dry==original,"Reverb zero must preserve the dry path exactly");
    p.reverb=1;run(engine,dry,p,{64});p.bypass=true;dry=original;run(engine,dry,p,{64});
    for(size_t i=47000;i<48000;++i)require(dry[0][i]==original[0][i],"Reverb must settle to unity on bypass");
}
void benchmark(orbit::Engine& engine) {
  for(int mode=0;mode<3;++mode) {
    orbit::Parameters p;p.density=120;p.grainMs=50;p.irLength=3;p.pitch=12;p.strategy=4;p.reverb=1;
    p.variant=mode;p.longIrMs=300;
    constexpr int block=64,iterations=3000;
    auto audio=input(block*iterations,48000);
    std::vector<double> times;times.reserve(iterations);
    engine.reset();
    for(int i=0;i<iterations;++i) {
        auto start=std::chrono::steady_clock::now();
        engine.process(audio[0].data()+i*block,audio[1].data()+i*block,block,p);
        times.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());
    }
    std::sort(times.begin(),times.end());
    std::cout<<"Mode "<<mode<<" 64-sample dense blocks at 48 kHz (microseconds): p50="<<times[1500]<<", p95="<<times[2850]<<", p99="<<times[2970]<<", max="<<times.back()<<", deadline=1333.33\n";
    std::cout<<"Deadline misses: "<<std::count_if(times.begin(),times.end(),[](double t){return t>1333.333;})<<" / "<<iterations<<"\n";
  }
}
int main() {
    try {
        testFFT();
        testLongConvolution();testSpectrum();
        orbit::Engine engine;engine.prepare(48000);
        testBlockInvariance(engine);testBypassAndSilence(engine);testStereoAndScope(engine);testRatesAndAutomation();
        testReverb(engine);
        testVariants(engine);
        require(audioAllocations==0,"process() performed heap allocations");
        benchmark(engine);
        std::cout<<"PASS: FFT, five selectors, block invariance, bypass, silence, stereo polarity, scope queue, 44.1/48/96/192 kHz, automation; audio allocations="<<audioAllocations<<"\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<"\n";return 1; }
}
