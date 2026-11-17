#include "cesium/omniverse/UsdTokens.h"

#include <fmt/format.h>

// clang-format off
PXR_NAMESPACE_OPEN_SCOPE

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4003)
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
#pragma warning(pop)
#endif

PXR_NAMESPACE_CLOSE_SCOPE

// clang-format on
