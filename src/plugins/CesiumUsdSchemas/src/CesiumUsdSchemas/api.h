#ifndef CESIUM_API_H
#define CESIUM_API_H

#include <pxr/base/arch/export.h>

#if defined(PXR_STATIC)
#   define CESIUM_API
#   define CESIUM_API_TEMPLATE_CLASS(...)
#   define CESIUM_API_TEMPLATE_STRUCT(...)
#   define CESIUM_LOCAL
#else
#   if defined(CESIUM_EXPORTS)
#       define CESIUM_API ARCH_EXPORT
#       define CESIUM_API_TEMPLATE_CLASS(...) ARCH_EXPORT_TEMPLATE(class, __VA_ARGS__)
#       define CESIUM_API_TEMPLATE_STRUCT(...) ARCH_EXPORT_TEMPLATE(struct, __VA_ARGS__)
#   else
#       define CESIUM_API ARCH_IMPORT
#       define CESIUM_API_TEMPLATE_CLASS(...) ARCH_IMPORT_TEMPLATE(class, __VA_ARGS__)
#       define CESIUM_API_TEMPLATE_STRUCT(...) ARCH_IMPORT_TEMPLATE(struct, __VA_ARGS__)
#   endif
#   define CESIUM_LOCAL ARCH_HIDDEN
#endif

#endif
