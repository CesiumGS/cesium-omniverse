#include "cesium/omniverse/Tokens.h"

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
}
// clang-format on
