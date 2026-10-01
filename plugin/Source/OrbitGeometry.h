#pragma once
#include <algorithm>
#include <array>
#include <cmath>
namespace orbit {
// Elliptical-grid map: the full parameter square maps bijectively to a disk.
inline std::array<float,2> squareToDisc(float x,float y) noexcept {
    x=std::clamp(x,-1.0f,1.0f);y=std::clamp(y,-1.0f,1.0f);
    return {x*std::sqrt(1-y*y*.5f),y*std::sqrt(1-x*x*.5f)};
}
inline std::array<float,2> discToSquare(float x,float y) noexcept {
    const float radius=std::hypot(x,y);if(radius>1){x/=radius;y/=radius;}
    constexpr double k=2.8284271247461903;
    const double xx=x,yy=y,a=2+xx*xx-yy*yy,b=2-xx*xx+yy*yy;
    const auto root=[](double value){return std::sqrt(std::max(0.0,value));};
    return {std::clamp(static_cast<float>((root(a+k*xx)-root(a-k*xx))*.5),-1.0f,1.0f),
            std::clamp(static_cast<float>((root(b+k*yy)-root(b-k*yy))*.5),-1.0f,1.0f)};
}
}
