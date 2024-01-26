/* <editor-fold desc="MIT License">

Copyright(c) 2023 Timothy Moore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

</editor-fold> */

#include "cesium/omniverse/UrlAssetAccessor.h"

#include <CesiumAsync/IAssetResponse.h>
#include <CesiumUtility/Tracing.h>
#include <omni/kit/IApp.h>

#include <algorithm>
#include <cstring>

namespace cesium::omniverse {
const auto CURL_BUFFERSIZE = 3145728L; // 3 MiB

class UrlAssetResponse : public CesiumAsync::IAssetResponse {
  public:
    uint16_t statusCode() const override {
        return _statusCode;
    }

    std::string contentType() const override {
        return _contentType;
    }

    const CesiumAsync::HttpHeaders& headers() const override {
        return _headers;
    }

    gsl::span<const std::byte> data() const override {
        return {const_cast<const std::byte*>(_result.data()), _result.size()};
    }

    static size_t headerCallback(char* buffer, size_t size, size_t nitems, void* userData);
    static size_t dataCallback(char* buffer, size_t size, size_t nitems, void* userData);
    void setCallbacks(CURL* curl);
    uint16_t _statusCode = 0;
    std::string _contentType;
    CesiumAsync::HttpHeaders _headers;
    std::vector<std::byte> _result;
};

class UrlAssetRequest : public CesiumAsync::IAssetRequest {
  public:
    UrlAssetRequest(std::string method, std::string url, CesiumAsync::HttpHeaders headers)
        : _method(std::move(method))
        , _url(std::move(url))
        , _headers(std::move(headers)) {}

    UrlAssetRequest(
        std::string method,
        std::string url,
        const std::vector<CesiumAsync::IAssetAccessor::THeader>& headers)
        : _method(std::move(method))
        , _url(std::move(url)) {
        _headers.insert(headers.begin(), headers.end());

        const auto app = carb::getCachedInterface<omni::kit::IApp>();
        const auto& buildInfo = app->getBuildInfo();
        const auto platformInfo = app->getPlatformInfo();

        _headers.insert({"X-Cesium-Client", "Cesium for Omniverse"});
        _headers.insert(
            {"X-Cesium-Client-Version", fmt::format("v{} {}", CESIUM_OMNI_VERSION, CESIUM_OMNI_GIT_HASH_ABBREVIATED)});
        _headers.insert({"X-Cesium-Client-Engine", fmt::format("Kit SDK {}", buildInfo.kitVersion)});
        _headers.insert({"X-Cesium-Client-OS", platformInfo.platform});
    }

    const std::string& method() const override {
        return this->_method;
    }

    const std::string& url() const override {
        return this->_url;
    }

    const CesiumAsync::HttpHeaders& headers() const override {
        return _headers;
    }

    const CesiumAsync::IAssetResponse* response() const override {
        return this->_response.get();
    }

    void setResponse(std::unique_ptr<UrlAssetResponse> response) {
        _response = std::move(response);
    }

  private:
    std::string _method;
    std::string _url;
    CesiumAsync::HttpHeaders _headers;
    std::unique_ptr<UrlAssetResponse> _response;
};

size_t UrlAssetResponse::headerCallback(char* buffer, size_t size, size_t nitems, void* userData) {
    // size is supposed to always be 1, but who knows
    const size_t cnt = size * nitems;
    auto* response = static_cast<UrlAssetResponse*>(userData);
    if (!response) {
        return cnt;
    }
    auto* colon = static_cast<char*>(std::memchr(buffer, ':', nitems));
    if (colon) {
        char* value = colon + 1;
        auto* end = std::find(value, buffer + cnt, '\r');
        while (value < end && *value == ' ') {
            ++value;
        }
        response->_headers.insert({std::string(buffer, colon), std::string(value, end)});
        auto contentTypeItr = response->_headers.find("content-type");
        if (contentTypeItr != response->_headers.end()) {
            response->_contentType = contentTypeItr->second;
        }
    }
    return cnt;
}

size_t UrlAssetResponse::dataCallback(char* buffer, size_t size, size_t nitems, void* userData) {
    const size_t cnt = size * nitems;
    auto* response = static_cast<UrlAssetResponse*>(userData);
    if (!response) {
        return cnt;
    }
    std::transform(buffer, buffer + cnt, std::back_inserter(response->_result), [](char c) {
        return std::byte{static_cast<unsigned char>(c)};
    });
    return cnt;
}
} //namespace cesium::omniverse

extern "C" size_t headerCallback(char* buffer, size_t size, size_t nitems, void* userData) {
    return cesium::omniverse::UrlAssetResponse::headerCallback(buffer, size, nitems, userData);
}

extern "C" size_t dataCallback(char* buffer, size_t size, size_t nitems, void* userData) {
    return cesium::omniverse::UrlAssetResponse::dataCallback(buffer, size, nitems, userData);
}

namespace cesium::omniverse {
void UrlAssetResponse::setCallbacks(CURL* curl) {
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ::dataCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ::headerCallback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, this);
}

UrlAssetAccessor::UrlAssetAccessor(const std::filesystem::path& certificatePath)
    : userAgent("Mozilla/5.0 Cesium for Omniverse")
    , _certificatePath(certificatePath.generic_string()) {
    // XXX Do we need to worry about the thread safety problems with this?
    curl_global_init(CURL_GLOBAL_ALL);
}

UrlAssetAccessor::~UrlAssetAccessor() {
    curl_global_cleanup();
}

curl_slist*
UrlAssetAccessor::setCommonOptions(CURL* curl, const std::string& url, const CesiumAsync::HttpHeaders& headers) {
    curl_easy_setopt(curl, CURLOPT_USERAGENT, userAgent.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    if (!_certificatePath.empty()) {
        curl_easy_setopt(curl, CURLOPT_CAINFO, _certificatePath.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, CURL_BUFFERSIZE);
    curl_easy_setopt(curl, CURLOPT_MAXCONNECTS, 20L);
    curl_easy_setopt(curl, CURLOPT_DNS_CACHE_TIMEOUT, 300L);
    // curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_slist* list = nullptr;
    for (const auto& header : headers) {
        std::string fullHeader = header.first + ":" + header.second;
        list = curl_slist_append(list, fullHeader.c_str());
    }
    if (list) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);
    }
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    return list;
}

CesiumAsync::Future<std::shared_ptr<CesiumAsync::IAssetRequest>> UrlAssetAccessor::get(
    const CesiumAsync::AsyncSystem& asyncSystem,
    const std::string& url,
    const std::vector<CesiumAsync::IAssetAccessor::THeader>& headers) {
    return asyncSystem.createFuture<std::shared_ptr<CesiumAsync::IAssetRequest>>([&](const auto& promise) {
        std::shared_ptr<UrlAssetRequest> request = std::make_shared<UrlAssetRequest>("GET", url, headers);
        asyncSystem.runInWorkerThread([promise, request, this]() {
            CESIUM_TRACE("UrlAssetAccessor::get");
            CurlHandle curl(this);
            curl_slist* list = setCommonOptions(curl(), request->url(), request->headers());
            std::unique_ptr<UrlAssetResponse> response = std::make_unique<UrlAssetResponse>();
            response->setCallbacks(curl());
            CURLcode responseCode = curl_easy_perform(curl());
            curl_slist_free_all(list);
            if (responseCode == 0) {
                long httpResponseCode = 0;
                curl_easy_getinfo(curl(), CURLINFO_RESPONSE_CODE, &httpResponseCode);
                response->_statusCode = static_cast<uint16_t>(httpResponseCode);
                // The response header callback also sets _contentType, so not sure that this is
                // necessary...
                char* ct = nullptr;
                curl_easy_getinfo(curl(), CURLINFO_CONTENT_TYPE, &ct);
                if (ct) {
                    response->_contentType = ct;
                }
                request->setResponse(std::move(response));
                promise.resolve(request);
            } else {
                std::string curlMsg("curl: ");
                curlMsg += curl_easy_strerror(responseCode);
                promise.reject(std::runtime_error(curlMsg));
            }
        });
    });
}

// request() with a verb and argument is essentially a POST

CesiumAsync::Future<std::shared_ptr<CesiumAsync::IAssetRequest>> UrlAssetAccessor::request(
    const CesiumAsync::AsyncSystem& asyncSystem,
    const std::string& verb,
    const std::string& url,
    const std::vector<CesiumAsync::IAssetAccessor::THeader>& headers,
    const gsl::span<const std::byte>& contentPayload) {
    return asyncSystem.createFuture<std::shared_ptr<CesiumAsync::IAssetRequest>>([&](const auto& promise) {
        auto request = std::make_shared<UrlAssetRequest>(verb, url, headers);
        auto payloadCopy = std::make_shared<std::vector<std::byte>>(contentPayload.begin(), contentPayload.end());
        asyncSystem.runInWorkerThread([promise, request, payloadCopy, this]() {
            CESIUM_TRACE("UrlAssetAccessor::request");
            CurlHandle curl(this);

            curl_slist* list = setCommonOptions(curl(), request->url(), request->headers());
            if (payloadCopy->size() > 1UL << 31) {
                curl_easy_setopt(curl(), CURLOPT_POSTFIELDSIZE_LARGE, payloadCopy->size());
            } else {
                curl_easy_setopt(curl(), CURLOPT_POSTFIELDSIZE, payloadCopy->size());
            }
            curl_easy_setopt(curl(), CURLOPT_COPYPOSTFIELDS, reinterpret_cast<const char*>(payloadCopy->data()));
            curl_easy_setopt(curl(), CURLOPT_CUSTOMREQUEST, request->method().c_str());
            std::unique_ptr<UrlAssetResponse> response = std::make_unique<UrlAssetResponse>();
            response->setCallbacks(curl());
            CURLcode responseCode = curl_easy_perform(curl());
            curl_slist_free_all(list);
            if (responseCode == 0) {
                long httpResponseCode = 0;
                curl_easy_getinfo(curl(), CURLINFO_RESPONSE_CODE, &httpResponseCode);
                response->_statusCode = static_cast<uint16_t>(httpResponseCode);
                char* ct = nullptr;
                curl_easy_getinfo(curl(), CURLINFO_CONTENT_TYPE, &ct);
                if (ct) {
                    response->_contentType = ct;
                }
                request->setResponse(std::move(response));
                promise.resolve(request);
            } else {
                std::string curlMsg("curl: ");
                curlMsg += curl_easy_strerror(responseCode);
                promise.reject(std::runtime_error(curlMsg));
            }
        });
    });
}

void UrlAssetAccessor::tick() noexcept {}
} //namespace cesium::omniverse
