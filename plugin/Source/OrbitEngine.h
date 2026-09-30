#pragma once
#include <array>
#include <atomic>
#include <complex>
#include <cstdint>
#include <vector>
#include "OrbitReverb.h"

namespace orbit {
constexpr float pi = 3.14159265358979323846f;

struct Parameters {
    float density = 80.0f, grainMs = 15.0f, pitch = 5.0f, mix = .65f;
    float jitter = .08f, spread = .6f, lookbackMs = 40.0f, outputDb = -3.0f;
    int irLength = 1; // 8, 16, 24, 32 ms
    int strategy = 3; // fixed, cycle, random, weighted, centroid
    uint32_t seed = 2025;
    bool bypass = false;
    float reverb = 0;
};

struct ScopeFrame { float minL=0, maxL=0, minR=0, maxR=0; };
class ScopeQueue {
public:
    bool push(ScopeFrame value) noexcept;
    bool pop(ScopeFrame& value) noexcept;
private:
    static constexpr uint32_t capacity = 1024;
    std::array<ScopeFrame, capacity> frames{};
    std::atomic<uint32_t> read{0}, write{0};
};

class FFT {
public:
    void prepare(int size);
    void transform(std::complex<float>* data, bool inverse) const noexcept;
    int size() const noexcept { return n; }
private:
    int n = 0;
    std::vector<unsigned> reversed;
    std::vector<std::complex<float>> roots;
};

class Engine {
public:
    void prepare(double sampleRate);
    void reset() noexcept;
    // Preparation is the only allocating phase. Audio data can be in-place.
    void process(float* left, float* right, int samples, const Parameters& parameters) noexcept;
    ScopeQueue scope;
    uint64_t droppedGrains() const noexcept { return dropped; } // audio thread/tests only
private:
    static constexpr int voicesCount = 48, bankSize = 8, fftCount = 7;
    static constexpr int phases = 128, taps = 16, rates = 151;
    struct Voice {
        std::vector<float> left, right;
        int length = 0, position = 0;
        float gainL = 0, gainR = 0;
    };
    struct Kernel {
        std::array<std::vector<std::complex<float>>, fftCount> spectra;
        float centroid = 0;
        int length = 0;
    };
    double sr = 48000, nextTrigger = 0;
    int historyHead = 0, historySize = 0, maxGrain = 0;
    std::vector<float> historyL, historyR, interpolation;
    std::array<float, 4097> window{};
    std::array<FFT, fftCount> fft;
    std::array<std::array<Kernel, bankSize>, 4> kernels;
    std::array<Voice, voicesCount> voices;
    std::vector<std::complex<float>> scratchL, scratchR;
    uint32_t rng = 2025, lastSeed = 2025;
    int cycle = 0, scopeCount = 0, scopeStride = 187;
    ScopeFrame scopeFrame{};
    float mix = .65f, gain = .7079458f, smoothing = .001f;
    bool initialParameters = true;
    RoomReverb room;
    float reverbAmount=0;
    uint64_t dropped = 0;
    float random() noexcept;
    float readSample(const std::vector<float>& history, double position, int rate) const noexcept;
    void trigger(const Parameters& p) noexcept;
};
} // namespace orbit
