// Copyright 2023 CesiumGS, Inc. and Contributors

#include "cesium/omniverse/CesiumIonSession.h"

#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/SettingsWrapper.h"

#include <CesiumAsync/SharedFuture.h>
#include <CesiumUtility/Uri.h>

#include <utility>

using namespace CesiumAsync;
using namespace CesiumIonClient;

using namespace cesium::omniverse;

CesiumIonSession::CesiumIonSession(
    const CesiumAsync::AsyncSystem& asyncSystem,
    std::shared_ptr<CesiumAsync::IAssetAccessor> pAssetAccessor,
    std::string ionServerUrl,
    std::string ionApiUrl,
    int64_t ionApplicationId)
    : _asyncSystem(asyncSystem)
    , _pAssetAccessor(std::move(pAssetAccessor))
    , _connection(std::nullopt)
    , _profile(std::nullopt)
    , _assets(std::nullopt)
    , _tokens(std::nullopt)
    , _isConnecting(false)
    , _isResuming(false)
    , _isLoadingProfile(false)
    , _isLoadingAssets(false)
    , _isLoadingTokens(false)
    , _loadProfileQueued(false)
    , _loadAssetsQueued(false)
    , _loadTokensQueued(false)
    , _authorizeUrl()
    , _ionServerUrl(std::move(ionServerUrl))
    , _ionApiUrl(std::move(ionApiUrl))
    , _ionApplicationId(ionApplicationId) {}

void CesiumIonSession::connect() {
    if (this->isConnecting() || this->isConnected() || this->isResuming()) {
        return;
    }

    this->_isConnecting = true;

    CesiumAsync::Future<std::optional<std::string>> futureApiUrl =
        !_ionApiUrl.empty()
            ? this->_asyncSystem.createResolvedFuture<std::optional<std::string>>(_ionApiUrl)
            : CesiumIonClient::Connection::getApiUrl(this->_asyncSystem, this->_pAssetAccessor, _ionServerUrl);

    std::move(futureApiUrl)
        .thenInMainThread([this](std::optional<std::string>&& ionApiUrl) {
            CesiumAsync::Promise<bool> promise = this->_asyncSystem.createPromise<bool>();

            if (_ionApiUrl.empty()) {
                _ionApiUrl = ionApiUrl.value();
            }

            // Make request to /appData to learn the server's authentication mode
            return this->ensureAppDataLoaded();
        })
        .thenInMainThread([this](bool loadedAppData) {
            if (!loadedAppData || !this->_appData.has_value()) {
                CesiumAsync::Promise<CesiumIonClient::Connection> promise =
                    this->_asyncSystem.createPromise<CesiumIonClient::Connection>();

                promise.reject(std::runtime_error("Failed to load _appData, can't create connection"));
                return promise.getFuture();
            }

            if (this->_appData->needsOauthAuthentication()) {
                return CesiumIonClient::Connection::authorize(
                    this->_asyncSystem,
                    this->_pAssetAccessor,
                    "Cesium for Omniverse",
                    _ionApplicationId,
                    "/cesium-for-omniverse/oauth2/callback",
                    {"assets:list", "assets:read", "profile:read", "tokens:read", "tokens:write", "geocode"},
                    [this](const std::string& url) {
                        // NOTE: We open the browser in the Python code. Check in the sign in widget's on_update_frame function.
                        this->_authorizeUrl = url;
                    },
                    this->_appData.value(),
                    _ionApiUrl,
                    CesiumUtility::Uri::resolve(_ionServerUrl, "oauth"));
            }

            return this->_asyncSystem.createResolvedFuture<CesiumIonClient::Connection>(CesiumIonClient::Connection(
                this->_asyncSystem, this->_pAssetAccessor, "", this->_appData.value(), _ionServerUrl));
        })
        .thenInMainThread([this](CesiumIonClient::Connection&& connection) {
            this->_isConnecting = false;
            this->_connection = std::move(connection);

            Settings::AccessToken token;
            token.ionApiUrl = _ionApiUrl;
            token.accessToken = this->_connection.value().getAccessToken();
            Settings::setAccessToken(token);

            Broadcast::connectionUpdated();
        })
        .catchInMainThread([this]([[maybe_unused]] std::exception&& e) {
            this->_isConnecting = false;
            this->_connection = std::nullopt;

            Broadcast::connectionUpdated();
        });
}

void CesiumIonSession::resume() {
    if (this->isConnecting() || this->isConnected() || this->isResuming()) {
        return;
    }

    auto tokens = Settings::getAccessTokens();

    if (tokens.size() < 1) {
        // No existing session to resume.
        return;
    }

    std::string accessToken;
    for (const auto& token : tokens) {
        if (token.ionApiUrl == _ionApiUrl) {
            accessToken = token.accessToken;
            break;
        }
    }

    if (accessToken.empty()) {
        // No existing session to resume.
        return;
    }

    this->_isResuming = true;

    // Verify that the connection actually works.
    this->ensureAppDataLoaded()
        .thenInMainThread([this, accessToken](bool loadedAppData) {
            CesiumAsync::Promise<void> promise = this->_asyncSystem.createPromise<void>();

            if (!loadedAppData || !this->_appData.has_value()) {
                promise.reject(std::runtime_error("Failed to obtain _appData, can't resume connection"));
                return promise.getFuture();
            }

            if (this->_appData->needsOauthAuthentication() && accessToken.empty()) {
                // No user access token was stored, so there's no existing session to resume.
                promise.resolve();
                this->_isResuming = false;
                return promise.getFuture();
            }

            std::shared_ptr<CesiumIonClient::Connection> pConnection = std::make_shared<CesiumIonClient::Connection>(
                this->_asyncSystem, this->_pAssetAccessor, accessToken, this->_appData.value(), _ionApiUrl);

            return pConnection->me().thenInMainThread(
                [this, pConnection](CesiumIonClient::Response<CesiumIonClient::Profile>&& response) {
                    if (!response.value.has_value()) {
                        this->_connection.reset();
                    }
                    this->_isResuming = false;
                    Broadcast::connectionUpdated();
                    // logResponseErrors(response);
                    if (response.value.has_value()) {
                        this->_connection = std::move(*pConnection);
                    }
                });
        })
        .catchInMainThread([this]([[maybe_unused]] std::exception&& e) {
            this->_isResuming = false;
            this->_connection.reset();
            Broadcast::connectionUpdated();
        });
}

void CesiumIonSession::disconnect() {
    this->_connection.reset();
    this->_profile.reset();
    this->_assets.reset();
    this->_tokens.reset();

    Settings::removeAccessToken(_ionApiUrl);

    Broadcast::connectionUpdated();
    Broadcast::profileUpdated();
    Broadcast::assetsUpdated();
    Broadcast::tokensUpdated();
}

void CesiumIonSession::tick() {
    this->_asyncSystem.dispatchMainThreadTasks();
}

void CesiumIonSession::refreshProfile() {
    if (!this->_connection || this->_isLoadingProfile) {
        this->_loadProfileQueued = true;
        return;
    }

    this->_isLoadingProfile = true;
    this->_loadProfileQueued = false;

    this->_connection->me()
        .thenInMainThread([this](Response<Profile>&& profile) {
            this->_isLoadingProfile = false;
            this->_profile = std::move(profile.value);
            Broadcast::profileUpdated();
            this->refreshProfileIfNeeded();
        })
        .catchInMainThread([this]([[maybe_unused]] std::exception&& e) {
            this->_isLoadingProfile = false;
            this->_profile = std::nullopt;
            Broadcast::profileUpdated();
            this->refreshProfileIfNeeded();
        });
}

void CesiumIonSession::refreshAssets() {
    if (!this->_connection || this->_isLoadingAssets) {
        return;
    }

    this->_isLoadingAssets = true;
    this->_loadAssetsQueued = false;

    this->_connection->assets()
        .thenInMainThread([this](Response<Assets>&& assets) {
            this->_isLoadingAssets = false;
            this->_assets = std::move(assets.value);
            Broadcast::assetsUpdated();
            this->refreshAssetsIfNeeded();
        })
        .catchInMainThread([this]([[maybe_unused]] std::exception&& e) {
            this->_isLoadingAssets = false;
            this->_assets = std::nullopt;
            Broadcast::assetsUpdated();
            this->refreshAssetsIfNeeded();
        });
}

void CesiumIonSession::refreshTokens() {
    if (!this->_connection || this->_isLoadingTokens) {
        return;
    }

    this->_isLoadingTokens = true;
    this->_loadTokensQueued = false;

    this->_connection->tokens()
        .thenInMainThread([this](Response<TokenList>&& tokens) {
            this->_isLoadingTokens = false;
            this->_tokens = tokens.value ? std::make_optional(std::move(tokens.value.value().items)) : std::nullopt;
            Broadcast::tokensUpdated();
            this->refreshTokensIfNeeded();
        })
        .catchInMainThread([this]([[maybe_unused]] std::exception&& e) {
            this->_isLoadingTokens = false;
            this->_tokens = std::nullopt;
            Broadcast::tokensUpdated();
            this->refreshTokensIfNeeded();
        });
}

const std::optional<CesiumIonClient::Connection>& CesiumIonSession::getConnection() const {
    return this->_connection;
}

const CesiumIonClient::Profile& CesiumIonSession::getProfile() {
    static const CesiumIonClient::Profile empty{};
    if (this->_profile) {
        return *this->_profile;
    } else {
        this->refreshProfile();
        return empty;
    }
}

const CesiumIonClient::Assets& CesiumIonSession::getAssets() {
    static const CesiumIonClient::Assets empty;
    if (this->_assets) {
        return *this->_assets;
    } else {
        this->refreshAssets();
        return empty;
    }
}

const std::vector<CesiumIonClient::Token>& CesiumIonSession::getTokens() {
    static const std::vector<CesiumIonClient::Token> empty;
    if (this->_tokens) {
        return *this->_tokens;
    } else {
        this->refreshTokens();
        return empty;
    }
}

bool CesiumIonSession::refreshProfileIfNeeded() {
    if (this->_loadProfileQueued || !this->_profile.has_value()) {
        this->refreshProfile();
    }
    return this->isProfileLoaded();
}

bool CesiumIonSession::refreshAssetsIfNeeded() {
    if (this->_loadAssetsQueued || !this->_assets.has_value()) {
        this->refreshAssets();
    }
    return this->isAssetListLoaded();
}

bool CesiumIonSession::refreshTokensIfNeeded() {
    if (this->_loadTokensQueued || !this->_tokens.has_value()) {
        this->refreshTokens();
    }
    return this->isTokenListLoaded();
}

Future<Response<Token>> CesiumIonSession::findToken(const std::string& token) const {
    if (!this->_connection) {
        return _asyncSystem.createResolvedFuture(Response<Token>(0, "NOTCONNECTED", "Not connected to Cesium ion."));
    }

    std::optional<std::string> maybeTokenID = Connection::getIdFromToken(token);

    if (!maybeTokenID) {
        return _asyncSystem.createResolvedFuture(Response<Token>(0, "INVALIDTOKEN", "The token is not valid."));
    }

    return this->_connection->token(*maybeTokenID);
}

CesiumAsync::Future<bool> CesiumIonSession::ensureAppDataLoaded() {

    return CesiumIonClient::Connection::appData(this->_asyncSystem, this->_pAssetAccessor, this->_ionApiUrl)
        .thenInMainThread([this](CesiumIonClient::Response<CesiumIonClient::ApplicationData>&& applicationData) {
            CesiumAsync::Promise<bool> promise = this->_asyncSystem.createPromise<bool>();

            this->_appData = applicationData.value;
            if (!applicationData.value.has_value()) {
                promise.resolve(false);
            } else {
                promise.resolve(true);
            }

            return promise.getFuture();
        })
        .catchInMainThread(
            [this]([[maybe_unused]] std::exception&& e) { return this->_asyncSystem.createResolvedFuture(false); });
}