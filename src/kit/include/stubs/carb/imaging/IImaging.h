// Stub header for carb/imaging/IImaging.h
// Provides minimal declarations needed for omni.ui ImageProvider
// This file exists because rtx_plugins (which contains the real header) is not publicly available.
#pragma once

namespace carb::imaging {

// Forward declaration for IMetadata interface (used as pointer parameter in ImageProvider)
class IMetadata;

// DisplayWindowRect structure used by ImageProvider
struct DisplayWindowRect {
    float left;
    float top;
    float right;
    float bottom;
};

} // namespace carb::imaging
