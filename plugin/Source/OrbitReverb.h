#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>

namespace orbit {
// Short, damped stereo room: four parallel feedback delays and two diffusers.
// All buffers are allocated by prepare(); the sample path is bounded and lock-free.
class RoomReverb {
    struct Delay {
        std::vector<float> data;
        size_t head=0;
        float damped=0;
        void prepare(int length) { data.assign(static_cast<size_t>(std::max(1,length)),0);head=0;damped=0; }
        void reset() { std::fill(data.begin(),data.end(),0);head=0;damped=0; }
        float comb(float input) noexcept {
            const float out=data[head];
            damped=.75f*out+.25f*damped;
            if(std::abs(damped)<1e-20f) damped=0;
            data[head]=input+.76f*damped;
            if(++head==data.size()) head=0;
            return out;
        }
        float diffuse(float input) noexcept {
            const float delayed=data[head],out=delayed-.5f*input;
            data[head]=input+.5f*out;
            if(std::abs(data[head])<1e-20f) data[head]=0;
            if(++head==data.size()) head=0;
            return out;
        }
    };
    std::array<std::array<Delay,4>,2> combs;
    std::array<std::array<Delay,2>,2> diffusers;
public:
    void prepare(double rate) {
        const double times[]{.0297,.0371,.0411,.0437};
        for(size_t c=0;c<2;++c) {
            for(size_t i=0;i<4;++i) combs[c][i].prepare(static_cast<int>(rate*(times[i]+c*.00079)));
            diffusers[c][0].prepare(static_cast<int>(rate*(.005+c*.00041)));
            diffusers[c][1].prepare(static_cast<int>(rate*(.0017+c*.00019)));
        }
    }
    void reset() noexcept { for(auto& c:combs)for(auto& d:c)d.reset();for(auto& c:diffusers)for(auto& d:c)d.reset(); }
    void process(float& left,float& right,float amount) noexcept {
        const float inputs[]{left,right};
        float room[2]{};
        for(size_t c=0;c<2;++c) {
            // Feed the room even at amount zero; transitions reveal a continuous tail.
            const float input=inputs[c]*.85f+inputs[1-c]*.15f;
            for(auto& d:combs[c])room[c]+=d.comb(input)*.25f;
            for(auto& d:diffusers[c])room[c]=d.diffuse(room[c]);
        }
        if(amount>0) {
            left=left*(1-.3f*amount)+room[0]*amount*.7f;
            right=right*(1-.3f*amount)+room[1]*amount*.7f;
        }
    }
};
}
