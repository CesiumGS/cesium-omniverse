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
// clang-format on

namespace cesium::omniverse::FabricTokens {
FABRIC_DEFINE_TOKENS(USD_TOKENS);

namespace {
std::mutex tokenMutex;
std::vector<omni::fabric::Token> primvars_st_tokens;
std::vector<omni::fabric::Token> imagery_layer_tokens;
std::vector<omni::fabric::Token> inputs_imagery_layer_tokens;
std::vector<omni::fabric::Token> inputs_polygon_imagery_layer_tokens;
std::vector<omni::fabric::Token> feature_id_tokens;

const omni::fabric::TokenC getToken(std::vector<omni::fabric::Token>& tokens, uint64_t index, std::string_view prefix) {
    const auto lock = std::scoped_lock<std::mutex>(tokenMutex);

    const auto size = index + 1;
    if (size > tokens.size()) {
        tokens.resize(size);
    }

    auto& token = tokens[index];

    if (token.asTokenC() == omni::fabric::kUninitializedToken) {
        const auto tokenStr = fmt::format("{}_{}", prefix, index);
        token = omni::fabric::Token(tokenStr.c_str());
    }

    return token.asTokenC();
}
} // namespace

const omni::fabric::TokenC primvars_st_n(uint64_t index) {
    return getToken(primvars_st_tokens, index, "primvars:st");
}

const omni::fabric::TokenC imagery_layer_n(uint64_t index) {
    return getToken(imagery_layer_tokens, index, "imagery_layer");
}

const omni::fabric::TokenC inputs_imagery_layer_n(uint64_t index) {
    return getToken(inputs_imagery_layer_tokens, index, "inputs:imagery_layer");
}

const omni::fabric::TokenC inputs_polygon_imagery_layer_n(uint64_t index) {
    return getToken(inputs_polygon_imagery_layer_tokens, index, "inputs:polygon_imagery_layer");
}

const omni::fabric::TokenC feature_id_n(uint64_t index) {
    return getToken(feature_id_tokens, index, "feature_id");
}

} // namespace cesium::omniverse::FabricTokens
