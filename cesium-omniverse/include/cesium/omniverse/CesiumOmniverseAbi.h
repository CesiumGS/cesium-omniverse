// clang-format off
#pragma once

#ifdef __cplusplus
#    define CESIUM_OMNI_EXPORT_C extern "C"
#elif !defined(CESIUM_OMNI_EXPORTS)
#    define CESIUM_OMNI_EXPORT_C extern
#else
#    define CESIUM_OMNI_EXPORT_C
#endif

#if defined(_WIN32)
#    define CESIUM_OMNI_ABI __cdecl
#else
#    define CESIUM_OMNI_ABI
#endif

#ifdef CESIUM_OMNI_EXPORTS
#    if defined(CESIUM_OMNI_STATIC)
#        define CESIUM_OMNI_EXPORT_CPP
#    elif defined(_WIN32)
#        define CESIUM_OMNI_EXPORT_CPP __declspec(dllexport)
#    elif defined(__linux__)
#        define CESIUM_OMNI_EXPORT_CPP __attribute__((visibility("default")))
#    endif
#    define CESIUM_OMNI_EXPORT_C_FUNCTION(ReturnType) CESIUM_OMNI_EXPORT_C CESIUM_OMNI_EXPORT_CPP ReturnType CESIUM_OMNI_ABI
#    define CESIUM_OMNI_EXPORT_CPP_FUNCTION(ReturnType) CESIUM_OMNI_EXPORT_CPP ReturnType CESIUM_OMNI_ABI
#    define CESIUM_OMNI_EXPORT_CPP_CLASS CESIUM_OMNI_EXPORT_CPP
#else
#    if defined(CESIUM_OMNI_STATIC)
#        define CESIUM_OMNI_EXPORT_CPP
#    elif defined(_WIN32)
#        define CESIUM_OMNI_EXPORT_CPP __declspec(dllimport)
#    else
#        define CESIUM_OMNI_EXPORT_CPP
#    endif
#    define CESIUM_OMNI_EXPORT_C_FUNCTION(ReturnType) CESIUM_OMNI_EXPORT_C CESIUM_OMNI_EXPORT_CPP ReturnType CESIUM_OMNI_ABI
#    define CESIUM_OMNI_EXPORT_CPP_FUNCTION(ReturnType) CESIUM_OMNI_EXPORT_CPP ReturnType CESIUM_OMNI_ABI
#    define CESIUM_OMNI_EXPORT_CPP_CLASS CESIUM_OMNI_EXPORT_CPP
#endif

#ifdef __cplusplus
#    define CESIUM_OMNI_DEFAULT(Val) = (Val)
#    if __cplusplus >= 201703L
#        define CESIUM_OMNI_NOEXCEPT noexcept
#        define CESIUM_OMNI_CALLBACK_NOEXCEPT noexcept
#    elif __cplusplus >= 201103L || _MSC_VER >= 1900
#        define CESIUM_OMNI_NOEXCEPT noexcept
#        define CESIUM_OMNI_CALLBACK_NOEXCEPT
#    else
#        define CESIUM_OMNI_NOEXCEPT throw()
#        define CESIUM_OMNI_CALLBACK_NOEXCEPT
#    endif
#else
#    define CESIUM_OMNI_DEFAULT(Val)
#    define CESIUM_OMNI_NOEXCEPT
#    define CESIUM_OMNI_CALLBACK_NOEXCEPT
#endif

#if !defined(CESIUM_OMNI_EXPORTS) && defined(__cplusplus) && __cplusplus >= 201402L
#    define CESIUM_OMNI_DEPRECATED(x) [[deprecated(x)]]
#else
#    define CESIUM_OMNI_DEPRECATED(x)
#endif
// clang-format on
