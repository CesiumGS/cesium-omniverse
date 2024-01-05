#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/CesiumOmniverse.h"

#include <CesiumUtility/CreditSystem.h>
#include <carb/BindingsPythonUtils.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/usd/sdf/path.h>

// Needs to go after carb
#include "pyboost11.h"

namespace pybind11::detail {

PYBOOST11_TYPE_CASTER(PXR_NS::GfMatrix4d, _("Matrix4d"));

}

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

CARB_BINDINGS("cesium.omniverse.python")

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic pop
#endif

DISABLE_PYBIND11_DYNAMIC_CAST(cesium::omniverse::ICesiumOmniverseInterface)

struct ViewportPythonBinding {
    PXR_NS::GfMatrix4d viewMatrix;
    PXR_NS::GfMatrix4d projMatrix;
    double width;
    double height;
};

PYBIND11_MODULE(CesiumOmniversePythonBindings, m) {

    using namespace cesium::omniverse;

    m.doc() = "pybind11 cesium.omniverse bindings";

    // clang-format off
    carb::defineInterfaceClass<ICesiumOmniverseInterface>(
        m, "ICesiumOmniverseInterface", "acquire_cesium_omniverse_interface", "release_cesium_omniverse_interface")
        .def("on_startup", &ICesiumOmniverseInterface::onStartup)
        .def("on_shutdown", &ICesiumOmniverseInterface::onShutdown)
        .def("reload_tileset", &ICesiumOmniverseInterface::reloadTileset)
        .def("on_update_frame", [](ICesiumOmniverseInterface& interface, const std::vector<ViewportPythonBinding>& viewports) {
            return interface.onUpdateFrame(reinterpret_cast<const ViewportApi*>(viewports.data()), viewports.size());
        })
        .def("on_stage_change", &ICesiumOmniverseInterface::onUsdStageChanged)
        .def("connect_to_ion", &ICesiumOmniverseInterface::connectToIon)
        .def("get_session", &ICesiumOmniverseInterface::getSession)
        .def("get_server_path", &ICesiumOmniverseInterface::getServerPath)
        .def("get_all_sessions", &ICesiumOmniverseInterface::getAllSessions)
        .def("get_all_server_paths", &ICesiumOmniverseInterface::getAllServerPaths)
        .def("get_set_default_token_result", &ICesiumOmniverseInterface::getSetDefaultTokenResult)
        .def("is_default_token_set", &ICesiumOmniverseInterface::isDefaultTokenSet)
        .def("create_token", &ICesiumOmniverseInterface::createToken)
        .def("select_token", &ICesiumOmniverseInterface::selectToken)
        .def("specify_token", &ICesiumOmniverseInterface::specifyToken)
        .def("get_asset_troubleshooting_details", &ICesiumOmniverseInterface::getAssetTroubleshootingDetails)
        .def("get_asset_token_troubleshooting_details", &ICesiumOmniverseInterface::getAssetTokenTroubleshootingDetails)
        .def("get_default_token_troubleshooting_details", &ICesiumOmniverseInterface::getDefaultTokenTroubleshootingDetails)
        .def("update_troubleshooting_details", py::overload_cast<const char*, int64_t, uint64_t, uint64_t>(&ICesiumOmniverseInterface::updateTroubleshootingDetails))
        .def("update_troubleshooting_details", py::overload_cast<const char*, int64_t, int64_t, uint64_t, uint64_t>(&ICesiumOmniverseInterface::updateTroubleshootingDetails))
        .def("print_fabric_stage", &ICesiumOmniverseInterface::printFabricStage)
        .def("get_render_statistics", &ICesiumOmniverseInterface::getRenderStatistics)
        .def("credits_available", &ICesiumOmniverseInterface::creditsAvailable)
        .def("get_credits", &ICesiumOmniverseInterface::getCredits)
        .def("credits_start_next_frame", &ICesiumOmniverseInterface::creditsStartNextFrame)
        .def("is_tracing_enabled", &ICesiumOmniverseInterface::isTracingEnabled);
    // clang-format on

    py::class_<CesiumIonSession, std::shared_ptr<CesiumIonSession>>(m, "CesiumIonSession")
        .def("is_connected", &CesiumIonSession::isConnected)
        .def("is_connecting", &CesiumIonSession::isConnecting)
        .def("is_resuming", &CesiumIonSession::isResuming)
        .def("is_profile_loaded", &CesiumIonSession::isProfileLoaded)
        .def("is_loading_profile", &CesiumIonSession::isLoadingProfile)
        .def("is_asset_list_loaded", &CesiumIonSession::isAssetListLoaded)
        .def("is_loading_asset_list", &CesiumIonSession::isLoadingAssetList)
        .def("is_token_list_loaded", &CesiumIonSession::isTokenListLoaded)
        .def("is_loading_token_list", &CesiumIonSession::isLoadingTokenList)
        .def("get_authorize_url", &CesiumIonSession::getAuthorizeUrl)
        .def("get_connection", &CesiumIonSession::getConnection)
        .def("get_profile", &CesiumIonSession::getProfile)
        .def("get_assets", &CesiumIonSession::getAssets)
        .def("get_tokens", &CesiumIonSession::getTokens)
        .def("refresh_tokens", &CesiumIonSession::refreshTokens)
        .def("refresh_profile", &CesiumIonSession::refreshProfile)
        .def("refresh_assets", &CesiumIonSession::refreshAssets)
        .def("disconnect", &CesiumIonSession::disconnect);

    py::class_<SetDefaultTokenResult>(m, "SetDefaultTokenResult")
        .def_readonly("code", &SetDefaultTokenResult::code)
        .def_readonly("message", &SetDefaultTokenResult::message);

    py::class_<CesiumIonClient::Assets>(m, "Assets")
        .def_readonly("link", &CesiumIonClient::Assets::link)
        .def_readonly("items", &CesiumIonClient::Assets::items);

    py::class_<CesiumIonClient::Asset>(m, "Asset")
        .def_readonly("asset_id", &CesiumIonClient::Asset::id)
        .def_readonly("name", &CesiumIonClient::Asset::name)
        .def_readonly("description", &CesiumIonClient::Asset::description)
        .def_readonly("attribution", &CesiumIonClient::Asset::attribution)
        .def_readonly("asset_type", &CesiumIonClient::Asset::type)
        .def_readonly("bytes", &CesiumIonClient::Asset::bytes)
        .def_readonly("date_added", &CesiumIonClient::Asset::dateAdded)
        .def_readonly("status", &CesiumIonClient::Asset::status)
        .def_readonly("percent_complete", &CesiumIonClient::Asset::percentComplete);

    py::class_<CesiumIonClient::Connection>(m, "Connection")
        // Wrap non-static member function in lambda. May be able to use py::overload_cast<> in C++ 20
        .def("get_api_uri", [](CesiumIonClient::Connection& connection) { return connection.getApiUrl(); })
        .def("get_access_token", &CesiumIonClient::Connection::getAccessToken);

    py::class_<CesiumIonClient::Profile>(m, "Profile")
        .def_readonly("id", &CesiumIonClient::Profile::id)
        .def_readonly("username", &CesiumIonClient::Profile::username);

    py::class_<CesiumIonClient::Token>(m, "Token")
        .def_readonly("id", &CesiumIonClient::Token::id)
        .def_readonly("name", &CesiumIonClient::Token::name)
        .def_readonly("token", &CesiumIonClient::Token::token)
        .def_readonly("is_default", &CesiumIonClient::Token::isDefault);

    py::class_<TokenTroubleshootingDetails>(m, "TokenTroubleshootingDetails")
        .def_readonly("token", &TokenTroubleshootingDetails::token)
        .def_readonly("is_valid", &TokenTroubleshootingDetails::isValid)
        .def_readonly("allows_access_to_asset", &TokenTroubleshootingDetails::allowsAccessToAsset)
        .def_readonly("associated_with_user_account", &TokenTroubleshootingDetails::associatedWithUserAccount)
        .def_readonly("show_details", &TokenTroubleshootingDetails::showDetails);

    py::class_<AssetTroubleshootingDetails>(m, "AssetTroubleshootingDetails")
        .def_readonly("asset_id", &AssetTroubleshootingDetails::assetId)
        .def_readonly("asset_exists_in_user_account", &AssetTroubleshootingDetails::assetExistsInUserAccount);

    py::class_<RenderStatistics>(m, "RenderStatistics")
        .def_readonly("materials_capacity", &RenderStatistics::materialsCapacity)
        .def_readonly("materials_loaded", &RenderStatistics::materialsLoaded)
        .def_readonly("geometries_capacity", &RenderStatistics::geometriesCapacity)
        .def_readonly("geometries_loaded", &RenderStatistics::geometriesLoaded)
        .def_readonly("geometries_rendered", &RenderStatistics::geometriesRendered)
        .def_readonly("triangles_loaded", &RenderStatistics::trianglesLoaded)
        .def_readonly("triangles_rendered", &RenderStatistics::trianglesRendered)
        .def_readonly("tileset_cached_bytes", &RenderStatistics::tilesetCachedBytes)
        .def_readonly("tiles_visited", &RenderStatistics::tilesVisited)
        .def_readonly("culled_tiles_visited", &RenderStatistics::culledTilesVisited)
        .def_readonly("tiles_rendered", &RenderStatistics::tilesRendered)
        .def_readonly("tiles_culled", &RenderStatistics::tilesCulled)
        .def_readonly("max_depth_visited", &RenderStatistics::maxDepthVisited)
        .def_readonly("tiles_loading_worker", &RenderStatistics::tilesLoadingWorker)
        .def_readonly("tiles_loading_main", &RenderStatistics::tilesLoadingMain)
        .def_readonly("tiles_loaded", &RenderStatistics::tilesLoaded);

    py::class_<ViewportPythonBinding>(m, "Viewport")
        .def(py::init())
        .def_readwrite("viewMatrix", &ViewportPythonBinding::viewMatrix)
        .def_readwrite("projMatrix", &ViewportPythonBinding::projMatrix)
        .def_readwrite("width", &ViewportPythonBinding::width)
        .def_readwrite("height", &ViewportPythonBinding::height);
}
