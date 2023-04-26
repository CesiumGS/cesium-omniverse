#include "cesium/omniverse/FabricAttributesBuilder.h"

#include "cesium/omniverse/UsdUtil.h"

namespace cesium::omniverse {
void FabricAttributesBuilder::addAttribute(const carb::flatcache::Type& type, const carb::flatcache::TokenC& name) {
    assert(_size < MAX_ATTRIBUTES);
    _attributes[_size++] = carb::flatcache::AttrNameAndType{type, name};
}

void FabricAttributesBuilder::createAttributes(const carb::flatcache::Path& path) {
    // Somewhat annoyingly, stageInProgress.createAttributes takes an std::array instead of a gsl::span. This is fine if
    // you know exactly which set of attributes to create at compile time but we don't. For example, not all prims will
    // have texture coordinates or materials. This class allows attributes to be added dynamically up to a hardcoded maximum
    // count (MAX_ATTRIBUTES) and avoids heap allocations. The downside is that we need this ugly if/else chain below.

    auto stageInProgress = UsdUtil::getFabricStageInProgress();

    // clang-format off
    if (_size == 0) stageInProgress.createAttributes<0>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 0>*>(_attributes.data()));
    else if (_size == 1) stageInProgress.createAttributes<1>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 1>*>(_attributes.data()));
    else if (_size == 2) stageInProgress.createAttributes<2>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 2>*>(_attributes.data()));
    else if (_size == 3) stageInProgress.createAttributes<3>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 3>*>(_attributes.data()));
    else if (_size == 4) stageInProgress.createAttributes<4>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 4>*>(_attributes.data()));
    else if (_size == 5) stageInProgress.createAttributes<5>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 5>*>(_attributes.data()));
    else if (_size == 6) stageInProgress.createAttributes<6>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 6>*>(_attributes.data()));
    else if (_size == 7) stageInProgress.createAttributes<7>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 7>*>(_attributes.data()));
    else if (_size == 8) stageInProgress.createAttributes<8>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 8>*>(_attributes.data()));
    else if (_size == 9) stageInProgress.createAttributes<9>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 9>*>(_attributes.data()));
    else if (_size == 10) stageInProgress.createAttributes<10>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 10>*>(_attributes.data()));
    else if (_size == 11) stageInProgress.createAttributes<11>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 11>*>(_attributes.data()));
    else if (_size == 12) stageInProgress.createAttributes<12>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 12>*>(_attributes.data()));
    else if (_size == 13) stageInProgress.createAttributes<13>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 13>*>(_attributes.data()));
    else if (_size == 14) stageInProgress.createAttributes<14>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 14>*>(_attributes.data()));
    else if (_size == 15) stageInProgress.createAttributes<15>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 15>*>(_attributes.data()));
    else if (_size == 16) stageInProgress.createAttributes<16>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 16>*>(_attributes.data()));
    else if (_size == 17) stageInProgress.createAttributes<17>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 17>*>(_attributes.data()));
    else if (_size == 18) stageInProgress.createAttributes<18>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 18>*>(_attributes.data()));
    else if (_size == 19) stageInProgress.createAttributes<19>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 19>*>(_attributes.data()));
    else if (_size == 20) stageInProgress.createAttributes<20>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 20>*>(_attributes.data()));
    else if (_size == 21) stageInProgress.createAttributes<21>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 21>*>(_attributes.data()));
    else if (_size == 22) stageInProgress.createAttributes<22>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 22>*>(_attributes.data()));
    else if (_size == 23) stageInProgress.createAttributes<23>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 23>*>(_attributes.data()));
    else if (_size == 24) stageInProgress.createAttributes<24>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 24>*>(_attributes.data()));
    else if (_size == 25) stageInProgress.createAttributes<25>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 25>*>(_attributes.data()));
    else if (_size == 26) stageInProgress.createAttributes<26>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 26>*>(_attributes.data()));
    else if (_size == 27) stageInProgress.createAttributes<27>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 27>*>(_attributes.data()));
    else if (_size == 28) stageInProgress.createAttributes<28>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 28>*>(_attributes.data()));
    else if (_size == 29) stageInProgress.createAttributes<29>(path, *reinterpret_cast<const std::array<carb::flatcache::AttrNameAndType, 29>*>(_attributes.data()));
    // clang-format on
}
} // namespace cesium::omniverse
