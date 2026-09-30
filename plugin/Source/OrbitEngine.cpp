#include "OrbitEngine.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace orbit {
static_assert(std::atomic<uint32_t>::is_always_lock_free, "Scope indices must be lock-free");
bool ScopeQueue::push(ScopeFrame value) noexcept {
    const auto w = write.load(std::memory_order_relaxed);
    if (w - read.load(std::memory_order_acquire) >= capacity) return false;
    frames[w % capacity] = value;
    write.store(w + 1, std::memory_order_release);
    return true;
}
bool ScopeQueue::pop(ScopeFrame& value) noexcept {
    const auto r = read.load(std::memory_order_relaxed);
    if (r == write.load(std::memory_order_acquire)) return false;
    value = frames[r % capacity];
    read.store(r + 1, std::memory_order_release);
    return true;
}
void FFT::prepare(int size) {
    n = size;
    reversed.resize(static_cast<size_t>(n));
    roots.resize(static_cast<size_t>(n / 2));
    int bits = 0;
    for (int s = n; s > 1; s >>= 1) ++bits;
    for (int i = 0; i < n; ++i) {
        unsigned x = static_cast<unsigned>(i), r = 0;
        for (int b = 0; b < bits; ++b) { r = (r << 1) | (x & 1u); x >>= 1; }
        reversed[static_cast<size_t>(i)] = r;
    }
    for (int i = 0; i < n / 2; ++i) {
        const double a = -2.0 * static_cast<double>(pi) * i / n;
        roots[static_cast<size_t>(i)] = {static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a))};
    }
}
void FFT::transform(std::complex<float>* data, bool inverse) const noexcept {
    for (int i = 0; i < n; ++i) {
        const auto j = reversed[static_cast<size_t>(i)];
        if (j > static_cast<unsigned>(i)) std::swap(data[i], data[j]);
    }
    for (int length = 2; length <= n; length *= 2) {
        const int half = length / 2, stride = n / length;
        for (int base = 0; base < n; base += length)
            for (int j = 0; j < half; ++j) {
                auto root = roots[static_cast<size_t>(j * stride)];
                if (inverse) root = std::conj(root);
                const auto a = data[base+j], b = data[base+j+half] * root;
                data[base+j] = a+b; data[base+j+half] = a-b;
            }
    }
    if (inverse) for (int i = 0; i < n; ++i) data[i] *= 1.0f / static_cast<float>(n);
}

float Engine::random() noexcept {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return static_cast<float>(rng >> 8) * (1.0f / 16777216.0f);
}
void Engine::prepare(double sampleRate) {
    if (!std::isfinite(sampleRate) || sampleRate < 8000 || sampleRate > 192000)
        throw std::invalid_argument("ORBIT supports sample rates from 8 to 192 kHz");
    sr = sampleRate;
    historySize = static_cast<int>(std::ceil(sr));
    maxGrain = static_cast<int>(std::ceil(sr * .05));
    historyL.assign(static_cast<size_t>(historySize), 0);
    historyR.assign(static_cast<size_t>(historySize), 0);
    const int maximum = maxGrain + static_cast<int>(std::ceil(sr * .032)) + 1;
    for (auto& voice : voices) {
        voice.left.resize(static_cast<size_t>(maximum));
        voice.right.resize(static_cast<size_t>(maximum));
    }
    for (int i = 0; i < fftCount; ++i) fft[static_cast<size_t>(i)].prepare(256 << i);
    scratchL.resize(16384); scratchR.resize(16384);
    for (size_t i = 0; i < window.size(); ++i)
        window[i] = .5f - .5f * std::cos(2.0f*pi*static_cast<float>(i)/4096.0f);

    // Antialiasing tables: 0.50..2.00 input samples per output sample, 128 phases.
    // All sinc/filter design is outside process().
    interpolation.resize(static_cast<size_t>(rates * phases * taps));
    for (int r = 0; r < rates; ++r) {
        const double cutoff = .9 / std::max(1.0, .5 + r * .01);
        for (int phase = 0; phase < phases; ++phase) {
            double sum = 0;
            for (int t = 0; t < taps; ++t) {
                const double x = t - (taps/2-1) - static_cast<double>(phase)/phases;
                const double a = static_cast<double>(pi) * x * cutoff;
                const double sinc = std::abs(a) < 1e-9 ? cutoff : cutoff * std::sin(a)/a;
                const double win = .5 + .5 * std::cos(static_cast<double>(pi) * x / (taps/2));
                auto& coefficient = interpolation[static_cast<size_t>((r*phases+phase)*taps+t)];
                coefficient = static_cast<float>(sinc * win); sum += coefficient;
            }
            for (int t = 0; t < taps; ++t)
                interpolation[static_cast<size_t>((r*phases+phase)*taps+t)] /= static_cast<float>(sum);
        }
    }
    rng = 2025;
    for (int lengthIndex = 0; lengthIndex < 4; ++lengthIndex) {
        const int length = std::max(8, static_cast<int>(sr * .008 * (lengthIndex+1)));
        for (int k = 0; k < bankSize; ++k) {
            auto& kernel = kernels[static_cast<size_t>(lengthIndex)][static_cast<size_t>(k)];
            kernel.length = length;
            std::vector<float> samples(static_cast<size_t>(length));
            const float centre = std::min(static_cast<float>(sr)*.35f, 250.0f * std::pow(1.55f, static_cast<float>(k)));
            const float omega = 2*pi*centre/static_cast<float>(sr), alpha = std::sin(omega)/1.6f;
            const float b0 = alpha/(1+alpha), b2 = -b0;
            const float a1 = -2*std::cos(omega)/(1+alpha), a2 = (1-alpha)/(1+alpha);
            float x1=0,x2=0,y1=0,y2=0, energy=0;
            for (int i=0;i<length;++i) {
                const float x = random()*2-1;
                const float y = b0*x+b2*x2-a1*y1-a2*y2;
                x2=x1; x1=x; y2=y1; y1=y;
                const float sample = y*std::exp(-6.0f*static_cast<float>(i)/static_cast<float>(length));
                samples[static_cast<size_t>(i)]=sample; energy+=sample*sample;
            }
            const float normal = .25f / std::sqrt(energy+1e-12f);
            for (auto& sample:samples) sample*=normal;
            samples[0]+=.75f;
            kernel.centroid=centre; // centroid measured below for content-aware selection
            for (int f=0;f<fftCount;++f) {
                const int size=fft[static_cast<size_t>(f)].size();
                if (size < length) continue;
                auto& spectrum=kernel.spectra[static_cast<size_t>(f)];
                spectrum.assign(static_cast<size_t>(size),{});
                for(int i=0;i<length;++i) spectrum[static_cast<size_t>(i)]=samples[static_cast<size_t>(i)];
                fft[static_cast<size_t>(f)].transform(spectrum.data(),false);
                if(f==fftCount-1) {
                    double weighted=0,total=0;
                    for(int i=1;i<=size/2;++i) { const auto m=std::abs(spectrum[static_cast<size_t>(i)]); weighted+=m*i*sr/size; total+=m; }
                    kernel.centroid=static_cast<float>(weighted/(total+1e-12));
                }
            }
        }
    }
    scopeStride=std::max(1,static_cast<int>(sr/256));
    smoothing=1.0f-std::exp(-1.0f/static_cast<float>(sr*.02));
    reset();
}
void Engine::reset() noexcept {
    std::fill(historyL.begin(),historyL.end(),0); std::fill(historyR.begin(),historyR.end(),0);
    for(auto& v:voices) v.length=v.position=0;
    historyHead=0; nextTrigger=0; rng=lastSeed=2025; cycle=0; dropped=0;
    scopeCount=0; scopeFrame={}; initialParameters=true;
}
float Engine::readSample(const std::vector<float>& history,double position,int rate) const noexcept {
    const int base=static_cast<int>(std::floor(position));
    const int phase=std::clamp(static_cast<int>((position-base)*phases),0,phases-1);
    const float* coefficients=interpolation.data()+(rate*phases+phase)*taps;
    float result=0;
    for(int t=0;t<taps;++t) {
        int index=(base+t-(taps/2-1))%historySize;
        if(index<0) index+=historySize;
        result+=history[static_cast<size_t>(index)]*coefficients[t];
    }
    return result;
}
void Engine::trigger(const Parameters& p) noexcept {
    Voice* voice=nullptr;
    for(auto& candidate:voices) if(candidate.position>=candidate.length) { voice=&candidate; break; }
    // Bounded overload policy: drop the new event, preserving existing tails.
    if(!voice) { ++dropped; return; }
    const int length=std::clamp(static_cast<int>(sr*p.grainMs*.001),8,maxGrain);
    const float semi=(random()*2-1)*p.pitch;
    const int rate=std::clamp(static_cast<int>(std::round(std::pow(2.0f,semi/12.0f)*100))-50,0,rates-1);
    const double ratio=.5+rate*.01;
    const int irLength=std::clamp(p.irLength,0,3);
    auto& bank=kernels[static_cast<size_t>(irLength)];
    const int convolutionLength=length+bank[0].length-1;
    int fftIndex=0;
    while(fftIndex<fftCount-1 && fft[static_cast<size_t>(fftIndex)].size()<convolutionLength) ++fftIndex;
    const int size=fft[static_cast<size_t>(fftIndex)].size();
    std::fill_n(scratchL.data(),size,std::complex<float>{});
    std::fill_n(scratchR.data(),size,std::complex<float>{});
    // Read a past snapshot. Extra interpolation guard guarantees no future reads.
    const double start=historyHead-1-std::ceil(length*ratio)-taps-static_cast<double>(p.lookbackMs)*sr*.001;
    for(int i=0;i<length;++i) {
        const float w=static_cast<float>(i)*4096/static_cast<float>(length-1);
        const auto wi=static_cast<size_t>(std::min(4095,static_cast<int>(w)));
        const float hann=window[wi]+(window[wi+1]-window[wi])*(w-static_cast<float>(wi));
        scratchL[static_cast<size_t>(i)]=readSample(historyL,start+i*ratio,rate)*hann;
        scratchR[static_cast<size_t>(i)]=readSample(historyR,start+i*ratio,rate)*hann;
    }
    auto& transform=fft[static_cast<size_t>(fftIndex)];
    transform.transform(scratchL.data(),false); transform.transform(scratchR.data(),false);
    int ir=0;
    switch(p.strategy) {
        case 1: ir=cycle; cycle=(cycle+1)%bankSize; break;
        case 2: ir=std::min(bankSize-1,static_cast<int>(random()*bankSize)); break;
        case 3: {
            float choice=random()*12.0f;
            for(ir=0;ir<bankSize-1;++ir) { choice-=1+static_cast<float>(ir)/7; if(choice<=0) break; }
            break;
        }
        case 4: {
            double total=0,weighted=0;
            for(int i=1;i<=size/2;++i) {
                const float magnitude=std::abs(scratchL[static_cast<size_t>(i)])+std::abs(scratchR[static_cast<size_t>(i)]);
                total+=magnitude; weighted+=magnitude*i*sr/size;
            }
            const float centre=static_cast<float>(weighted/(total+1e-12));
            float distance=std::numeric_limits<float>::max();
            for(int k=0;k<bankSize;++k) if(std::abs(bank[static_cast<size_t>(k)].centroid-centre)<distance) {
                ir=k; distance=std::abs(bank[static_cast<size_t>(k)].centroid-centre);
            }
            break;
        }
        default: break;
    }
    const auto& spectrum=bank[static_cast<size_t>(ir)].spectra[static_cast<size_t>(fftIndex)];
    for(int i=0;i<size;++i) { scratchL[static_cast<size_t>(i)]*=spectrum[static_cast<size_t>(i)]; scratchR[static_cast<size_t>(i)]*=spectrum[static_cast<size_t>(i)]; }
    transform.transform(scratchL.data(),true); transform.transform(scratchR.data(),true);
    for(int i=0;i<convolutionLength;++i) { voice->left[static_cast<size_t>(i)]=scratchL[static_cast<size_t>(i)].real(); voice->right[static_cast<size_t>(i)]=scratchR[static_cast<size_t>(i)].real(); }
    voice->length=convolutionLength; voice->position=0;
    const float pan=(random()*2-1)*p.spread;
    const float normal=1.0f/std::max(1.0f,p.density*p.grainMs*.0005f);
    voice->gainL=1.41421356f*std::cos((pan+1)*pi*.25f)*normal;
    voice->gainR=1.41421356f*std::sin((pan+1)*pi*.25f)*normal;
}

static float finite(float value) noexcept { return std::isfinite(value)? std::clamp(value,-16.0f,16.0f):0.0f; }
static float ceiling(float value) noexcept {
    const float magnitude=std::abs(value);
    return magnitude<=.9f?value:std::copysign(.9f+.1f*std::tanh((magnitude-.9f)*10),value);
}
void Engine::process(float* left,float* right,int samples,const Parameters& params) noexcept {
    if(historySize==0 || !left || !right || samples<=0) return;
    Parameters p=params;
    p.density=std::clamp(finite(p.density/10)*10,10.0f,120.0f);
    p.grainMs=std::clamp(finite(p.grainMs/10)*10,5.0f,50.0f);
    p.pitch=std::clamp(finite(p.pitch),0.0f,12.0f);
    p.jitter=std::clamp(finite(p.jitter),0.0f,.5f);
    p.spread=std::clamp(finite(p.spread),0.0f,1.0f);
    p.lookbackMs=std::clamp(finite(p.lookbackMs/100)*100,0.0f,200.0f);
    const float targetMix=p.bypass?0:std::clamp(finite(p.mix),0.0f,1.0f);
    const float targetGain=p.bypass?1:std::pow(10.0f,std::clamp(finite(p.outputDb/10)*10,-24.0f,6.0f)/20);
    if(initialParameters) { mix=targetMix; gain=targetGain; initialParameters=false; }
    if(p.seed!=lastSeed) { rng=p.seed==0?1:p.seed; lastSeed=p.seed; cycle=0; }
    for(int i=0;i<samples;++i) {
        const float inL=finite(left[i]),inR=finite(right[i]);
        scopeFrame.minL=std::min(scopeFrame.minL,inL); scopeFrame.maxL=std::max(scopeFrame.maxL,inL);
        scopeFrame.minR=std::min(scopeFrame.minR,inR); scopeFrame.maxR=std::max(scopeFrame.maxR,inR);
        if(++scopeCount>=scopeStride) { scope.push(scopeFrame); scopeCount=0; scopeFrame={}; }
        historyL[static_cast<size_t>(historyHead)]=inL; historyR[static_cast<size_t>(historyHead)]=inR;
        historyHead=(historyHead+1)%historySize;
        if(nextTrigger<=0) {
            trigger(p);
            nextTrigger+=sr/p.density*(1+(random()*2-1)*p.jitter);
        }
        --nextTrigger;
        float wetL=0,wetR=0;
        for(auto& voice:voices) if(voice.position<voice.length) {
            wetL+=voice.left[static_cast<size_t>(voice.position)]*voice.gainL;
            wetR+=voice.right[static_cast<size_t>(voice.position)]*voice.gainR;
            ++voice.position;
        }
        mix+=(targetMix-mix)*smoothing; gain+=(targetGain-gain)*smoothing;
        if(std::abs(targetMix-mix)<1e-5f) mix=targetMix;
        if(std::abs(targetGain-gain)<1e-4f) gain=targetGain;
        if(p.bypass && mix<1e-6f && std::abs(gain-1)<1e-5f) { left[i]=inL; right[i]=inR; }
        else { left[i]=ceiling((inL*(1-mix)+wetL*mix)*gain); right[i]=ceiling((inR*(1-mix)+wetR*mix)*gain); }
    }
}
} // namespace orbit
