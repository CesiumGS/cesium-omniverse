#include "cesium/omniverse/UsdTokens.h"

#include <fmt/format.h>

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

TF_DEFINE_PUBLIC_TOKENS(UsdTokens1, USD_TOKENS_1);
TF_DEFINE_PUBLIC_TOKENS(UsdTokens2, USD_TOKENS_2);
TF_DEFINE_PUBLIC_TOKENS(UsdTokens3, USD_TOKENS_3);

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic pop
#endif

#ifdef CESIUM_OMNI_MSVC
__pragma(warning(pop))
#endif

PXR_NAMESPACE_CLOSE_SCOPE
    // clang-format on

    namespace cesium::omniverse::FabricTokens {
    FABRIC_DEFINE_TOKENS_1(USD_TOKENS_1);
    FABRIC_DEFINE_TOKENS_2(USD_TOKENS_2);
    FABRIC_DEFINE_TOKENS_3(USD_TOKENS_3);

    namespace {
    std::mutex tokenMutex;
    std::vector<omni::fabric::Token> feature_id_tokens;
    std::vector<omni::fabric::Token> raster_overlay_tokens;
    std::vector<omni::fabric::Token> inputs_raster_overlay_tokens;
    std::vector<omni::fabric::Token> primvars_st_tokens;
    std::vector<omni::fabric::Token> primvars_st_interpolation_tokens;
    std::vector<omni::fabric::Token> property_tokens;

    const omni::fabric::Token
    getToken(std::vector<omni::fabric::Token>& tokens, uint64_t index, const std::string_view& pattern) {
        const auto lock = std::scoped_lock(tokenMutex);

        const auto size = index + 1;
        if (size > tokens.size()) {
            tokens.resize(size);
        }

        auto& token = tokens[index];

        if (token.isNull()) {
            const auto tokenStr = fmt::format(pattern, index);
            token = omni::fabric::Token::createImmortal(tokenStr.c_str());
        }

        return token;
    }
    } // namespace

    const omni::fabric::Token feature_id_n(uint64_t index) {
        return getToken(feature_id_tokens, index, "feature_id_{}");
    }

    const omni::fabric::Token raster_overlay_n(uint64_t index) {
        return getToken(raster_overlay_tokens, index, "raster_overlay_{}");
    }

    const omni::fabric::Token inputs_raster_overlay_n(uint64_t index) {
        return getToken(inputs_raster_overlay_tokens, index, "inputs:raster_overlay_{}");
    }

    const omni::fabric::Token primvars_st_n(uint64_t index) {
        return getToken(primvars_st_tokens, index, "primvars:st_{}");
    }

    const omni::fabric::Token primvars_st_interpolation_n(uint64_t index) {
        return getToken(primvars_st_interpolation_tokens, index, "primvars:st_{}:interpolation");
    }

    const omni::fabric::Token property_n(uint64_t index) {
        return getToken(property_tokens, index, "property_{}");
    }

} // namespace cesium::omniverse::FabricTokens
