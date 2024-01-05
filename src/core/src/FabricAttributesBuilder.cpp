#include "cesium/omniverse/FabricAttributesBuilder.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <omni/fabric/SimStageWithHistory.h>

namespace cesium::omniverse {

void FabricAttributesBuilder::addAttribute(const omni::fabric::Type& type, const omni::fabric::Token& name) {
    assert(_size < MAX_ATTRIBUTES);
    _attributes[_size++] = omni::fabric::AttrNameAndType(type, name);
}

void FabricAttributesBuilder::createAttributes(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path) const {
    // Somewhat annoyingly, fabricStage.createAttributes takes an std::array instead of a gsl::span. This is fine if
    // you know exactly which set of attributes to create at compile time but we don't. For example, not all prims will
    // have texture coordinates or materials. This class allows attributes to be added dynamically up to a hardcoded maximum
    // count (MAX_ATTRIBUTES) and avoids heap allocations. The downside is that we need this ugly if/else chain below.

    // clang-format off
    if (_size == 0) fabricStage.createAttributes<0>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 0>*>(_attributes.data()));
    else if (_size == 1) fabricStage.createAttributes<1>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 1>*>(_attributes.data()));
    else if (_size == 2) fabricStage.createAttributes<2>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 2>*>(_attributes.data()));
    else if (_size == 3) fabricStage.createAttributes<3>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 3>*>(_attributes.data()));
    else if (_size == 4) fabricStage.createAttributes<4>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 4>*>(_attributes.data()));
    else if (_size == 5) fabricStage.createAttributes<5>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 5>*>(_attributes.data()));
    else if (_size == 6) fabricStage.createAttributes<6>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 6>*>(_attributes.data()));
    else if (_size == 7) fabricStage.createAttributes<7>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 7>*>(_attributes.data()));
    else if (_size == 8) fabricStage.createAttributes<8>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 8>*>(_attributes.data()));
    else if (_size == 9) fabricStage.createAttributes<9>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 9>*>(_attributes.data()));
    else if (_size == 10) fabricStage.createAttributes<10>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 10>*>(_attributes.data()));
    else if (_size == 11) fabricStage.createAttributes<11>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 11>*>(_attributes.data()));
    else if (_size == 12) fabricStage.createAttributes<12>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 12>*>(_attributes.data()));
    else if (_size == 13) fabricStage.createAttributes<13>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 13>*>(_attributes.data()));
    else if (_size == 14) fabricStage.createAttributes<14>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 14>*>(_attributes.data()));
    else if (_size == 15) fabricStage.createAttributes<15>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 15>*>(_attributes.data()));
    else if (_size == 16) fabricStage.createAttributes<16>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 16>*>(_attributes.data()));
    else if (_size == 17) fabricStage.createAttributes<17>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 17>*>(_attributes.data()));
    else if (_size == 18) fabricStage.createAttributes<18>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 18>*>(_attributes.data()));
    else if (_size == 19) fabricStage.createAttributes<19>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 19>*>(_attributes.data()));
    else if (_size == 20) fabricStage.createAttributes<20>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 20>*>(_attributes.data()));
    else if (_size == 21) fabricStage.createAttributes<21>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 21>*>(_attributes.data()));
    else if (_size == 22) fabricStage.createAttributes<22>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 22>*>(_attributes.data()));
    else if (_size == 23) fabricStage.createAttributes<23>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 23>*>(_attributes.data()));
    else if (_size == 24) fabricStage.createAttributes<24>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 24>*>(_attributes.data()));
    else if (_size == 25) fabricStage.createAttributes<25>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 25>*>(_attributes.data()));
    else if (_size == 26) fabricStage.createAttributes<26>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 26>*>(_attributes.data()));
    else if (_size == 27) fabricStage.createAttributes<27>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 27>*>(_attributes.data()));
    else if (_size == 28) fabricStage.createAttributes<28>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 28>*>(_attributes.data()));
    else if (_size == 29) fabricStage.createAttributes<29>(path, *reinterpret_cast<const std::array<omni::fabric::AttrNameAndType, 29>*>(_attributes.data()));
    // clang-format on
}
} // namespace cesium::omniverse
