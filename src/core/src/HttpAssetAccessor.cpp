#include "cesium/omniverse/HttpAssetAccessor.h"

#include "cesium/omniverse/LoggerSink.h"

#include <omni/kit/IApp.h>
#include <zlib.h>

#include <exception>
#include <stdexcept>

#define CPR_RESERVE_SIZE 10485760 // 10 MiB

namespace cesium::omniverse {
namespace {
std::string decodeGzip(std::string& content) {
    z_stream zs; // z_stream is zlib's control structure
    memset(&zs, 0, sizeof(zs));

    if (inflateInit2(&zs, MAX_WBITS + 16) != Z_OK) {
        return std::move(content);
    }

    zs.next_in = reinterpret_cast<Bytef*>(content.data());
    zs.avail_in = static_cast<uInt>(content.size());

    int ret;
    std::array<char, 32768> outbuffer;
    std::string output;

    // get the decompressed bytes blockwise using repeated calls to inflate
    do {
        zs.next_out = reinterpret_cast<Bytef*>(outbuffer.data());
        zs.avail_out = sizeof(outbuffer);

        ret = inflate(&zs, 0);

        if (output.size() < zs.total_out) {
            std::size_t decompressSoFar = output.size();
            std::size_t addSize = zs.total_out - output.size();
            output.resize(output.size() + addSize);
            std::memcpy(output.data() + decompressSoFar, outbuffer.data(), addSize);
        }

    } while (ret == Z_OK);

    inflateEnd(&zs);

    if (ret != Z_STREAM_END) {
        return std::move(content);
    }

    return output;
}

struct GZipDecompressInterceptor final : public cpr::Interceptor {
  public:
    GZipDecompressInterceptor(const std::filesystem::path& certificatePath)
        : _certificatePath(certificatePath.generic_string()) {}

    cpr::Response intercept(cpr::Session& session) override {
#ifdef CESIUM_OMNI_UNIX
        curl_easy_setopt(session.GetCurlHolder()->handle, CURLOPT_CAINFO, _certificatePath.c_str());
#endif
        curl_easy_setopt(session.GetCurlHolder()->handle, CURLOPT_ACCEPT_ENCODING, nullptr);

        CURLcode curl_error = curl_easy_perform(session.GetCurlHolder()->handle);

        if (curl_error == CURLE_PEER_FAILED_VERIFICATION) {
            long verifyResult;
            curl_easy_getinfo(session.GetCurlHolder()->handle, CURLINFO_SSL_VERIFYRESULT, &verifyResult);
            CESIUM_LOG_WARN("SSL PEER VERIFICATION FAILED: {}", verifyResult);
        }

        auto response = session.Complete(curl_error);
        response.text = decodeGzip(response.text);
        return response;
    }

  private:
    std::string _certificatePath;
};

cpr::Header createCprHeader(const std::vector<CesiumAsync::IAssetAccessor::THeader>& nativeHeaders) {
    cpr::Header cprHeader;

    const auto app = carb::getCachedInterface<omni::kit::IApp>();
    const auto& buildInfo = app->getBuildInfo();
    const auto platformInfo = app->getPlatformInfo();

    cprHeader.insert(nativeHeaders.begin(), nativeHeaders.end());
    cprHeader.insert(std::make_pair("X-Cesium-Client", "Cesium for Omniverse"));
    cprHeader.insert(std::make_pair(
        "X-Cesium-Client-Version", fmt::format("v{} {}", CESIUM_OMNI_VERSION, CESIUM_OMNI_GIT_HASH_ABBREVIATED)));
    cprHeader.insert(std::make_pair("X-Cesium-Client-Engine", fmt::format("Kit SDK {}", buildInfo.kitVersion)));
    cprHeader.insert(std::make_pair("X-Cesium-Client-OS", platformInfo.platform));

    return cprHeader;
}

} // namespace

HttpAssetAccessor::HttpAssetAccessor(const std::filesystem::path& certificatePath) {
    _interceptor = std::make_shared<GZipDecompressInterceptor>(certificatePath);
}

CesiumAsync::Future<std::shared_ptr<CesiumAsync::IAssetRequest>> HttpAssetAccessor::get(
    const CesiumAsync::AsyncSystem& asyncSystem,
    const std::string& url,
    const std::vector<THeader>& headers) {
    auto promise = asyncSystem.createPromise<std::shared_ptr<CesiumAsync::IAssetRequest>>();
    const auto cprHeader = createCprHeader(headers);
    std::shared_ptr<cpr::Session> session = std::make_shared<cpr::Session>();
    session->AddInterceptor(_interceptor);
    session->SetHeader(cprHeader);
    session->SetUrl(cpr::Url(url));
    session->SetReserveSize(CPR_RESERVE_SIZE);
    session->GetCallback([promise, url, headers](cpr::Response&& response) mutable {
        if (response.error) {
            promise.reject(
                std::runtime_error(fmt::format("Request to {} failed with error: {}", url, response.error.message)));
        } else {
            promise.resolve(std::make_shared<HttpAssetRequest>("GET", url, headers, std::move(response)));
        }
    });

    return promise.getFuture();
}

CesiumAsync::Future<std::shared_ptr<CesiumAsync::IAssetRequest>> HttpAssetAccessor::request(
    const CesiumAsync::AsyncSystem& asyncSystem,
    const std::string& verb,
    const std::string& url,
    const std::vector<THeader>& headers,
    const gsl::span<const std::byte>& contentPayload) {
    auto promise = asyncSystem.createPromise<std::shared_ptr<CesiumAsync::IAssetRequest>>();
    const auto cprHeader = createCprHeader(headers);
    std::shared_ptr<cpr::Session> session = std::make_shared<cpr::Session>();
    session->SetHeader(cprHeader);
    session->SetUrl(cpr::Url(url));
    session->SetReserveSize(CPR_RESERVE_SIZE);
#ifdef CESIUM_OMNI_UNIX
    session->AddInterceptor(_interceptor);
#endif
    if (verb == "GET") {
        session->GetCallback([promise, url, headers](cpr::Response&& response) mutable {
            promise.resolve(std::make_shared<HttpAssetRequest>("GET", url, headers, std::move(response)));
        });

        return promise.getFuture();
    } else if (verb == "POST") {
        session->SetBody(cpr::Body{reinterpret_cast<const char*>(contentPayload.data()), contentPayload.size()});
        session->PostCallback([promise, url, headers](cpr::Response&& response) mutable {
            promise.resolve(std::make_shared<HttpAssetRequest>("POST", url, headers, std::move(response)));
        });

        return promise.getFuture();
    }

    return asyncSystem.createResolvedFuture<std::shared_ptr<CesiumAsync::IAssetRequest>>(nullptr);
}

void HttpAssetAccessor::tick() noexcept {}
} // namespace cesium::omniverse
