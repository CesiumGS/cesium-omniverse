#include "cesium/omniverse/CesiumOmniverse.h"

#include <carb/BindingsPythonUtils.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec4d.h>

// Needs to go after carb
#include "pyboost11.h"

#include "cesium/omniverse/CesiumIonSession.h"

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

    carb::defineInterfaceClass<ICesiumOmniverseInterface>(
        m, "ICesiumOmniverseInterface", "acquire_cesium_omniverse_interface", "release_cesium_omniverse_interface")
        .def("initialize", &ICesiumOmniverseInterface::initialize)
        .def("finalize", &ICesiumOmniverseInterface::finalize)
        .def("addCesiumData", &ICesiumOmniverseInterface::addCesiumData)
        .def("addTilesetUrl", &ICesiumOmniverseInterface::addTilesetUrl)
        .def("addTilesetIon", &ICesiumOmniverseInterface::addTilesetIon)
        .def("removeTileset", &ICesiumOmniverseInterface::removeTileset)
        .def("addIonRasterOverlay", &ICesiumOmniverseInterface::addIonRasterOverlay)
        .def("updateFrame", &ICesiumOmniverseInterface::updateFrame)
        .def("update_stage", &ICesiumOmniverseInterface::updateStage)
        .def("setGeoreferenceOrigin", &ICesiumOmniverseInterface::setGeoreferenceOrigin)
        .def("connect_to_ion", &ICesiumOmniverseInterface::connectToIon)
        .def("on_ui_update", &ICesiumOmniverseInterface::onUiUpdate)
        .def("get_session", &ICesiumOmniverseInterface::getSession);

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
        .def("disconnect", &CesiumIonSession::disconnect)
        .def("refresh_profile", &CesiumIonSession::refreshProfile);

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
}
