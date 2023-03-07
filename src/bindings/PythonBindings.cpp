#include "cesium/omniverse/CesiumOmniverse.h"

#include <Cesium3DTilesSelection/CreditSystem.h>
#include <carb/BindingsPythonUtils.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec4d.h>

// Needs to go after carb
#include "pyboost11.h"

#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/TokenTroubleshooter.h"

namespace pybind11 {
namespace detail {

PYBOOST11_TYPE_CASTER(pxr::GfMatrix4d, _("Matrix4d"));

} // end namespace detail
} // end namespace pybind11

CARB_BINDINGS("cesium.omniverse.python")
DISABLE_PYBIND11_DYNAMIC_CAST(cesium::omniverse::ICesiumOmniverseInterface)

PYBIND11_MODULE(CesiumOmniversePythonBindings, m) {

    using namespace cesium::omniverse;

    m.doc() = "pybind11 cesium.omniverse bindings";

    // clang-format off
    carb::defineInterfaceClass<ICesiumOmniverseInterface>(
        m, "ICesiumOmniverseInterface", "acquire_cesium_omniverse_interface", "release_cesium_omniverse_interface")
        .def("on_startup", &ICesiumOmniverseInterface::onStartup)
        .def("on_shutdown", &ICesiumOmniverseInterface::onShutdown)
        .def("add_tileset_url", &ICesiumOmniverseInterface::addTilesetUrl)
        .def("add_tileset_ion", py::overload_cast<const char*, int64_t>(&ICesiumOmniverseInterface::addTilesetIon))
        .def("add_tileset_ion", py::overload_cast<const char*, int64_t, const char*>(&ICesiumOmniverseInterface::addTilesetIon))
        .def("add_ion_imagery", py::overload_cast<const char*, const char*, int64_t>(&ICesiumOmniverseInterface::addImageryIon))
        .def("add_tileset_and_imagery", &ICesiumOmniverseInterface::addTilesetAndImagery)
        .def("get_all_tileset_paths", &ICesiumOmniverseInterface::getAllTilesetPaths)
        .def("is_tileset", &ICesiumOmniverseInterface::isTileset)
        .def("remove_tileset", &ICesiumOmniverseInterface::removeTileset)
        .def("reload_tileset", &ICesiumOmniverseInterface::reloadTileset)
        .def("on_update_frame", &ICesiumOmniverseInterface::onUpdateFrame)
        .def("on_update_ui", &ICesiumOmniverseInterface::onUpdateUi)
        .def("on_stage_change", &ICesiumOmniverseInterface::onStageChange)
        .def("set_georeference_origin", &ICesiumOmniverseInterface::setGeoreferenceOrigin)
        .def("connect_to_ion", &ICesiumOmniverseInterface::connectToIon)
        .def("get_session", &ICesiumOmniverseInterface::getSession)
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
        .def("credits_available", &ICesiumOmniverseInterface::creditsAvailable)
        .def("get_credits", &ICesiumOmniverseInterface::getCredits);
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
        .def("get_api_url", &CesiumIonClient::Connection::getApiUrl)
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
}
