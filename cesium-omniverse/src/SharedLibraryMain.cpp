
#include "Globals.h"

#include <filesystem>
#include <iostream>

namespace Cesium {

#ifdef CESIUM_OMNI_WINDOWS
#include <windows.h>

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            TCHAR s[MAX_PATH + 1];
            GetModuleFileName(hinstDLL, s, _countof(s));
            SharedLibraryPath = std::filesystem::path(s);
            break;
        case DLL_THREAD_ATTACH:
        case DLL_THREAD_DETACH:
        case DLL_PROCESS_DETACH:
            break;
    }
    return TRUE;
}
#endif
} // namespace Cesium
