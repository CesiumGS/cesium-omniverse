#include "cesium/omniverse/UsdTokens.h"

#include <spdlog/fmt/fmt.h>

// clang-format off
PXR_NAMESPACE_OPEN_SCOPE

#ifdef CESIUM_OMNI_MSVC
__pragma(warning(push))
__pragma(warning(disable: 4003))
#endif

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

TF_DEFINE_PUBLIC_TOKENS(
    UsdTokens,
    USD_TOKENS);

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic pop
#endif

#ifdef CESIUM_OMNI_MSVC
__pragma(warning(pop))
#endif

PXR_NAMESPACE_CLOSE_SCOPE
    // clang-format on

    namespace cesium::omniverse::FabricTokens {
    FABRIC_DEFINE_TOKENS(USD_TOKENS);

    namespace {
    std::mutex tokenMutex;
    std::vector<omni::fabric::Token> feature_id_tokens;
    std::vector<omni::fabric::Token> raster_overlay_layer_tokens;
    std::vector<omni::fabric::Token> inputs_raster_overlay_layer_tokens;
    std::vector<omni::fabric::Token> primvars_st_tokens;
    std::vector<omni::fabric::Token> property_tokens;

    const omni::fabric::TokenC
    getToken(std::vector<omni::fabric::Token>& tokens, uint64_t index, const std::string_view& prefix) {
        const auto lock = std::scoped_lock(tokenMutex);

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

    const omni::fabric::TokenC feature_id_n(uint64_t index) {
        return getToken(feature_id_tokens, index, "feature_id");
    }

    const omni::fabric::TokenC raster_overlay_layer_n(uint64_t index) {
        return getToken(raster_overlay_layer_tokens, index, "raster_overlay_layer");
    }

    const omni::fabric::TokenC inputs_raster_overlay_layer_n(uint64_t index) {
        return getToken(inputs_raster_overlay_layer_tokens, index, "inputs:raster_overlay_layer");
    }

    const omni::fabric::TokenC primvars_st_n(uint64_t index) {
        return getToken(primvars_st_tokens, index, "primvars:st");
    }

    const omni::fabric::TokenC property_n(uint64_t index) {
        return getToken(property_tokens, index, "property");
    }

} // namespace cesium::omniverse::FabricTokens
