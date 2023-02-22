#pragma once

#include <CesiumAsync/IAssetAccessor.h>
#include <CesiumAsync/IAssetRequest.h>
#include <CesiumAsync/IAssetResponse.h>
#include <cpr/cpr.h>

#include <filesystem>

namespace cesium::omniverse {
class HttpAssetResponse : public CesiumAsync::IAssetResponse {
  public:
    HttpAssetResponse(cpr::Response&& response)
        : _response{std::move(response)} {
        for (const auto& header : _response.header) {
            _headers.insert({header.first, header.second});
        }
    }

    uint16_t statusCode() const override {
        return uint16_t(_response.status_code);
    }

    std::string contentType() const override {
        auto it = _response.header.find("content-type");
        if (it != _response.header.end()) {
            return it->second;
        }

        return "";
    }

    const CesiumAsync::HttpHeaders& headers() const override {
        return _headers;
    }

    gsl::span<const std::byte> data() const override {
        return {reinterpret_cast<const std::byte*>(_response.text.data()), _response.text.size()};
    }

  private:
    CesiumAsync::HttpHeaders _headers;
    cpr::Response _response;
};

class HttpAssetRequest : public CesiumAsync::IAssetRequest {
  public:
    HttpAssetRequest(
        std::string&& method,
        const std::string& url,
        const std::vector<CesiumAsync::IAssetAccessor::THeader>& headers,
        cpr::Response&& response)
        : _method{std::move(method)}
        , _url{url}
        , _headers{headers.begin(), headers.end()}
        , _response{std::move(response)} {}

    const std::string& method() const override {
        return _method;
    }

    const std::string& url() const override {
        return _url;
    }

    const CesiumAsync::HttpHeaders& headers() const override {
        return _headers;
    }

    const CesiumAsync::IAssetResponse* response() const override {
        return &_response;
    }

  private:
    std::string _method;
    std::string _url;
    CesiumAsync::HttpHeaders _headers;
    HttpAssetResponse _response;
};

class HttpAssetAccessor : public CesiumAsync::IAssetAccessor {
  public:
    HttpAssetAccessor(const std::filesystem::path& certificatePath);

    CesiumAsync::Future<std::shared_ptr<CesiumAsync::IAssetRequest>>
    get(const CesiumAsync::AsyncSystem& asyncSystem,
        const std::string& url,
        const std::vector<THeader>& headers = {}) override;

    CesiumAsync::Future<std::shared_ptr<CesiumAsync::IAssetRequest>> request(
        const CesiumAsync::AsyncSystem& asyncSystem,
        const std::string& verb,
        const std::string& url,
        const std::vector<THeader>& headers = std::vector<THeader>(),
        const gsl::span<const std::byte>& contentPayload = {}) override;

    void tick() noexcept override;

  private:
    std::shared_ptr<cpr::Interceptor> _interceptor;
};
} // namespace cesium::omniverse
