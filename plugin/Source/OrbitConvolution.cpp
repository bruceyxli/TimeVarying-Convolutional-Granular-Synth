#include "OrbitEngine.h"
#include <algorithm>
#include <cmath>

namespace orbit {
std::vector<float> LongConvolver::makeImpulse(double rate,float milliseconds) {
    const int length=std::max(8,static_cast<int>(rate*milliseconds*.001));
    std::vector<float> impulse(static_cast<size_t>(length));
    uint32_t rng=0x4f524249u;double energy=0;
    for(int i=0;i<length;++i) {
        rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
        const float noise=static_cast<float>(rng>>8)*(2.0f/16777216.0f)-1;
        const float value=noise*std::exp(-6.0f*static_cast<float>(i)/static_cast<float>(length));
        impulse[static_cast<size_t>(i)]=value;energy+=value*value;
    }
    const float scale=.35f/static_cast<float>(std::sqrt(energy+1e-20));
    for(auto& value:impulse)value*=scale;
    impulse[0]+=.65f;
    return impulse;
}
void LongConvolver::prepare(double rate) {
    fft.prepare(size);
    for(int k=0;k<choices;++k) {
        const auto impulse=makeImpulse(rate,40.0f+5.0f*static_cast<float>(k));
        parts[static_cast<size_t>(k)]=(static_cast<int>(impulse.size())+hop-1)/hop;
        auto& bank=kernels[static_cast<size_t>(k)];
        bank.assign(static_cast<size_t>(parts[static_cast<size_t>(k)]*size),{});
        for(int p=0;p<parts[static_cast<size_t>(k)];++p) {
            auto* block=bank.data()+p*size;
            for(int i=0;i<hop && p*hop+i<static_cast<int>(impulse.size());++i)block[i]=impulse[static_cast<size_t>(p*hop+i)];
            fft.transform(block,false);
        }
    }
    maximumParts=parts.back();
    historyL.resize(static_cast<size_t>(maximumParts*size));historyR.resize(historyL.size());
    smoothing=1-std::exp(-1.0f/static_cast<float>(rate*.05));
    reset();
}
void LongConvolver::reset() noexcept {
    std::fill(historyL.begin(),historyL.end(),std::complex<float>{});
    std::fill(historyR.begin(),historyR.end(),std::complex<float>{});
    inputL.fill({});inputR.fill({});outputL.fill(0);outputR.fill(0);overlapL.fill(0);overlapR.fill(0);
    position=head=0;initial=true;
}
void LongConvolver::process(float& left,float& right,float milliseconds) noexcept {
    if(maximumParts==0)return;
    const float target=std::clamp((milliseconds-40)*.2f,0.0f,52.0f);
    if(initial){selection=target;initial=false;}
    selection+=(target-selection)*smoothing;
    if(std::abs(target-selection)<.00001f)selection=target;
    inputL[static_cast<size_t>(position)]=left;inputR[static_cast<size_t>(position)]=right;
    left=outputL[static_cast<size_t>(position)];right=outputR[static_cast<size_t>(position)];
    if(++position==hop){render();position=0;}
}
void LongConvolver::render() noexcept {
    std::fill(inputL.begin()+hop,inputL.end(),std::complex<float>{});
    std::fill(inputR.begin()+hop,inputR.end(),std::complex<float>{});
    fft.transform(inputL.data(),false);fft.transform(inputR.data(),false);
    std::copy(inputL.begin(),inputL.end(),historyL.begin()+head*size);
    std::copy(inputR.begin(),inputR.end(),historyR.begin()+head*size);
    sumL.fill({});sumR.fill({});
    const int lo=static_cast<int>(selection),hi=std::min(choices-1,lo+1);
    const float blend=selection-static_cast<float>(lo);
    const int count=std::max(parts[static_cast<size_t>(lo)],parts[static_cast<size_t>(hi)]);
    for(int p=0;p<count;++p) {
        const int index=(head-p+maximumParts)%maximumParts;
        const auto* l=historyL.data()+index*size;const auto* r=historyR.data()+index*size;
        const auto* a=p<parts[static_cast<size_t>(lo)]?kernels[static_cast<size_t>(lo)].data()+p*size:nullptr;
        const auto* b=p<parts[static_cast<size_t>(hi)]?kernels[static_cast<size_t>(hi)].data()+p*size:nullptr;
        for(int bin=0;bin<size;++bin) {
            const auto av=a?a[bin]:std::complex<float>{},bv=b?b[bin]:std::complex<float>{};
            const auto kernel=av+(bv-av)*blend;
            sumL[static_cast<size_t>(bin)]+=l[bin]*kernel;sumR[static_cast<size_t>(bin)]+=r[bin]*kernel;
        }
    }
    fft.transform(sumL.data(),true);fft.transform(sumR.data(),true);
    for(int i=0;i<hop;++i) {
        const auto j=static_cast<size_t>(i);
        outputL[j]=sumL[j].real()+overlapL[j];outputR[j]=sumR[j].real()+overlapR[j];
        overlapL[j]=sumL[j+hop].real();overlapR[j]=sumR[j+hop].real();
    }
    head=(head+1)%maximumParts;
}
} // namespace orbit
