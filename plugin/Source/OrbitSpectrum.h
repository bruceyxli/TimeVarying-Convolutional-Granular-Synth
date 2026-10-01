#pragma once
#include "OrbitEngine.h"
#include <algorithm>
#include <cmath>

namespace orbit {
struct OutputFrame {
    std::array<float,256> left{},right{};
    double sampleRate=48000;
    uint32_t generation=0;
};
// Audio producer / editor consumer. Full queues drop display data, never audio.
class OutputQueue {
public:
    void reset() noexcept { position=0;++generation; }
    void capture(const float* left,const float* right,int count,double rate) noexcept {
        for(int i=0;i<count;++i) {
            pending.left[static_cast<size_t>(position)]=std::isfinite(left[i])?left[i]:0;
            const float r=right?right[i]:left[i];pending.right[static_cast<size_t>(position)]=std::isfinite(r)?r:0;
            if(++position==256) {
                pending.sampleRate=rate;pending.generation=generation;
                const auto w=write.load(std::memory_order_relaxed);
                if(w-read.load(std::memory_order_acquire)<capacity) {
                    frames[w%capacity]=pending;write.store(w+1,std::memory_order_release);
                }
                position=0;
            }
        }
    }
    bool pop(OutputFrame& frame) noexcept {
        const auto r=read.load(std::memory_order_relaxed);
        if(r==write.load(std::memory_order_acquire))return false;
        frame=frames[r%capacity];read.store(r+1,std::memory_order_release);return true;
    }
private:
    static constexpr uint32_t capacity=64;
    std::array<OutputFrame,capacity> frames{};
    OutputFrame pending;
    std::atomic<uint32_t> read{0},write{0};
    uint32_t generation=0;
    int position=0;
};

// All FFT work is performed by the editor, not the audio callback.
class SpectrumAnalyzer {
public:
    static constexpr int size=4096,bins=size/2+1;
    SpectrumAnalyzer() {
        fft.prepare(size);
        for(int i=0;i<size;++i){window[static_cast<size_t>(i)]=.5f-.5f*std::cos(2*pi*static_cast<float>(i)/static_cast<float>(size-1));windowSum+=window[static_cast<size_t>(i)];}
        clear();
    }
    void clear() noexcept {left.fill(0);right.fill(0);db.fill(-90);head=0;}
    void append(const OutputFrame& frame) noexcept {
        if(rate!=frame.sampleRate || generation!=frame.generation){clear();rate=frame.sampleRate;generation=frame.generation;}
        for(size_t i=0;i<frame.left.size();++i){left[static_cast<size_t>(head)]=frame.left[i];right[static_cast<size_t>(head)]=frame.right[i];head=(head+1)%size;}
    }
    void analyze() noexcept {
        for(int i=0;i<size;++i) {
            const auto n=static_cast<size_t>((head+i)%size),j=static_cast<size_t>(i);
            l[j]=left[n]*window[j];r[j]=right[n]*window[j];
        }
        fft.transform(l.data(),false);fft.transform(r.data(),false);
        for(int i=0;i<bins;++i) {
            const auto j=static_cast<size_t>(i);
            const float amplitude=2*std::sqrt((std::norm(l[j])+std::norm(r[j]))*.5f)/windowSum;
            db[j]=std::clamp(20*std::log10(std::max(amplitude,.000001f)),-90.0f,6.0f);
        }
    }
    float level(float low,float high) const noexcept {
        const int first=std::clamp(static_cast<int>(std::floor(low*size/rate)),1,bins-1);
        const int last=std::clamp(static_cast<int>(std::ceil(high*size/rate)),first,bins-1);
        float peak=-90;for(int i=first;i<=last;++i)peak=std::max(peak,db[static_cast<size_t>(i)]);return peak;
    }
    double sampleRate() const noexcept {return rate;}
private:
    FFT fft;
    std::array<float,size> left{},right{},window{};
    std::array<float,bins> db{};
    std::array<std::complex<float>,size> l{},r{};
    double rate=48000;
    uint32_t generation=0;
    int head=0;
    float windowSum=0;
};
} // namespace orbit
