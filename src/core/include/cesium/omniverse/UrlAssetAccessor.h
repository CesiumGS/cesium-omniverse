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

#pragma once

#include "CesiumAsync/AsyncSystem.h"
#include "CesiumAsync/IAssetAccessor.h"

#include <curl/curl.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

namespace cesium::omniverse {
// A cache that permits reuse of CURL handles. This is extremely important for performance
// because libcurl will keep existing connections open if a curl handle is not destroyed
// ("cleaned up").

struct CurlCache {
    struct CacheEntry {
        CacheEntry()
            : curl(nullptr)
            , free(false) {}
        CacheEntry(CURL* in_curl, bool in_free)
            : curl(in_curl)
            , free(in_free) {}
        CURL* curl;
        bool free;
    };
    std::mutex cacheMutex;
    std::vector<CacheEntry> cache;
    CURL* get() {
        std::lock_guard<std::mutex> lock(cacheMutex);
        for (auto& entry : cache) {
            if (entry.free) {
                entry.free = false;
                return entry.curl;
            }
        }
        cache.emplace_back(curl_easy_init(), false);
        return cache.back().curl;
    }
    void release(CURL* curl) {
        std::lock_guard<std::mutex> lock(cacheMutex);
        for (auto& entry : cache) {
            if (curl == entry.curl) {
                curl_easy_reset(curl);
                entry.free = true;
                return;
            }
        }
        throw std::logic_error("releasing a curl handle that is not in the cache");
    }
};

// Simple implementation of AssetAcessor that can make network and local requests
class UrlAssetAccessor : public CesiumAsync::IAssetAccessor {
  public:
    UrlAssetAccessor(const std::filesystem::path& certificatePath = {});
    ~UrlAssetAccessor() override;

    CesiumAsync::Future<std::shared_ptr<CesiumAsync::IAssetRequest>>
    get(const CesiumAsync::AsyncSystem& asyncSystem,
        const std::string& url,
        const std::vector<CesiumAsync::IAssetAccessor::THeader>& headers) override;

    CesiumAsync::Future<std::shared_ptr<CesiumAsync::IAssetRequest>> request(
        const CesiumAsync::AsyncSystem& asyncSystem,
        const std::string& verb,
        const std::string& url,
        const std::vector<CesiumAsync::IAssetAccessor::THeader>& headers,
        const gsl::span<const std::byte>& contentPayload) override;

    void tick() noexcept override;
    CurlCache curlCache;
    std::string userAgent;

  protected:
    curl_slist* setCommonOptions(CURL* curl, const std::string& url, const CesiumAsync::HttpHeaders& headers);
    std::string _certificatePath;
};

// RAII wrapper for the CurlCache.

class CurlHandle {
  public:
    CurlHandle(UrlAssetAccessor* in_accessor)
        : _accessor(in_accessor)

    {
        _curl = _accessor->curlCache.get();
    }

    ~CurlHandle() {
        _accessor->curlCache.release(_curl);
    }

    CURL* operator()() const {
        return _curl;
    }

  private:
    UrlAssetAccessor* _accessor;
    CURL* _curl;
};
} // namespace cesium::omniverse
