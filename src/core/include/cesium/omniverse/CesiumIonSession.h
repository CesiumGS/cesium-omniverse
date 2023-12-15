#pragma once

#include <CesiumAsync/AsyncSystem.h>
#include <CesiumAsync/IAssetAccessor.h>
#include <CesiumAsync/SharedFuture.h>
#include <CesiumIonClient/Connection.h>
#include <carb/settings/SettingsUtils.h>

#include <memory>
#include <optional>
#include <vector>

namespace cesium::omniverse {

class CesiumIonSession {
  public:
    CesiumIonSession(
        CesiumAsync::AsyncSystem& asyncSystem,
        std::shared_ptr<CesiumAsync::IAssetAccessor> pAssetAccessor,
        std::string ionServerUrl,
        std::string ionApiUrl,
        int64_t ionApplicationId);

    [[nodiscard]] const std::shared_ptr<CesiumAsync::IAssetAccessor>& getAssetAccessor() const {
        return this->_pAssetAccessor;
    }
    [[nodiscard]] const CesiumAsync::AsyncSystem& getAsyncSystem() const {
        return this->_asyncSystem;
    }
    CesiumAsync::AsyncSystem& getAsyncSystem() {
        return this->_asyncSystem;
    }

    [[nodiscard]] bool isConnected() const {
        return this->_connection.has_value();
    }
    [[nodiscard]] bool isConnecting() const {
        return this->_isConnecting;
    }
    [[nodiscard]] bool isResuming() const {
        return this->_isResuming;
    }

    [[nodiscard]] bool isProfileLoaded() const {
        return this->_profile.has_value();
    }
    [[nodiscard]] bool isLoadingProfile() const {
        return this->_isLoadingProfile;
    }

    [[nodiscard]] bool isAssetListLoaded() const {
        return this->_assets.has_value();
    }
    [[nodiscard]] bool isLoadingAssetList() const {
        return this->_isLoadingAssets;
    }

    [[nodiscard]] bool isTokenListLoaded() const {
        return this->_tokens.has_value();
    }
    [[nodiscard]] bool isLoadingTokenList() const {
        return this->_isLoadingTokens;
    }

    void connect();
    void resume();
    void disconnect();

    void tick();

    void refreshProfile();
    void refreshAssets();
    void refreshTokens();

    [[nodiscard]] const std::optional<CesiumIonClient::Connection>& getConnection() const;
    const CesiumIonClient::Profile& getProfile();
    const CesiumIonClient::Assets& getAssets();
    const std::vector<CesiumIonClient::Token>& getTokens();

    [[nodiscard]] const std::string& getAuthorizeUrl() const {
        return this->_authorizeUrl;
    }

    bool refreshProfileIfNeeded();
    bool refreshAssetsIfNeeded();
    bool refreshTokensIfNeeded();

    /**
     * Finds the details of the specified token in the user's account.
     *
     * If this session is not connected, returns std::nullopt.
     *
     * Even if the list of tokens is already loaded, this method does a new query
     * in order get the most up-to-date information about the token.
     *
     * @param token The token.
     * @return The details of the token, or an error response if the token does
     * not exist in the signed-in user account.
     */
    [[nodiscard]] CesiumAsync::Future<CesiumIonClient::Response<CesiumIonClient::Token>>
    findToken(const std::string& token) const;

  private:
    CesiumAsync::AsyncSystem _asyncSystem;
    std::shared_ptr<CesiumAsync::IAssetAccessor> _pAssetAccessor;

    std::optional<CesiumIonClient::Connection> _connection;
    std::optional<CesiumIonClient::Profile> _profile;
    std::optional<CesiumIonClient::Assets> _assets;
    std::optional<std::vector<CesiumIonClient::Token>> _tokens;

    std::optional<CesiumAsync::SharedFuture<CesiumIonClient::Token>> _projectDefaultTokenDetailsFuture;

    bool _isConnecting;
    bool _isResuming;
    bool _isLoadingProfile;
    bool _isLoadingAssets;
    bool _isLoadingTokens;

    bool _loadProfileQueued;
    bool _loadAssetsQueued;
    bool _loadTokensQueued;

    std::string _authorizeUrl;
    std::string _ionServerUrl;
    std::string _ionApiUrl;
    int64_t _ionApplicationId;
};
} // namespace cesium::omniverse
