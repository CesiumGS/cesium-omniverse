#pragma once

#include "cesium/omniverse/HttpAssetAccessor.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumAsync/AsyncSystem.h>
#include <pxr/usd/sdf/path.h>

#include <unordered_set>

namespace cesium::omniverse {

class CesiumIonSession;

class SessionRegistry {
  public:
    SessionRegistry(const SessionRegistry&) = delete;
    SessionRegistry(SessionRegistry&&) = delete;

    static SessionRegistry& getInstance() {
        static SessionRegistry instance;
        return instance;
    }

    SessionRegistry& operator=(const SessionRegistry&) = delete;
    SessionRegistry& operator=(SessionRegistry) = delete;

    void addSession(
        CesiumAsync::AsyncSystem& asyncSystem,
        const std::shared_ptr<HttpAssetAccessor>& httpAssetAccessor,
        const pxr::SdfPath& ionServerPath);
    std::vector<pxr::SdfPath> getAllSessionPaths();
    std::shared_ptr<CesiumIonSession> getSession(const pxr::SdfPath& ionServerPath);
    void removeSession(const pxr::SdfPath& ionServerPath);
    bool sessionExists(const pxr::SdfPath& ionServerPath);

    void clear();

  protected:
    SessionRegistry() = default;
    ~SessionRegistry() = default;

  private:
    std::map<pxr::SdfPath, std::shared_ptr<CesiumIonSession>> _sessions{};
};

} // namespace cesium::omniverse
