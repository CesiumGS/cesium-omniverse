#include "cesium/omniverse/Tokens.h"

#include <spdlog/fmt/fmt.h>

// clang-format off
namespace pxr {

#ifdef CESIUM_OMNI_MSVC
__pragma(warning(push))
__pragma(warning(disable: 4003))
#endif

TF_DEFINE_PUBLIC_TOKENS(
    UsdTokens,
    USD_TOKENS);

#ifdef CESIUM_OMNI_MSVC
__pragma(warning(pop))
#endif

}

namespace cesium::omniverse::FabricTokens {
FABRIC_DEFINE_TOKENS(USD_TOKENS);

namespace {
    std::vector<omni::fabric::Token> propertyTokens;
}

const omni::fabric::TokenC getPropertyToken(uint64_t index) {
    // TODO: need a mutex?
    propertyTokens.resize(index);

    auto& propertyToken = propertyTokens[index];

    if (propertyToken.asTokenC() == omni::fabric::kUninitializedToken) {
        const auto propertyTokenStr = fmt::format("property_token_{}", index);
        propertyToken = omni::fabric::Token(propertyTokenStr.c_str());
    }

    return propertyToken.asTokenC();
}
}
// clang-format on
