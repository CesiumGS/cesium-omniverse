#pragma once

#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/packageResolver.h>
#include <tbb/concurrent_hash_map.h>

#include <vector>

PXR_NAMESPACE_OPEN_SCOPE
class InMemoryAssetContext {
  public:
    using accessor = tbb::concurrent_hash_map<std::string, std::shared_ptr<ArAsset>>::accessor;
    using const_accessor = tbb::concurrent_hash_map<std::string, std::shared_ptr<ArAsset>>::const_accessor;

    InMemoryAssetContext(const InMemoryAssetContext&) = delete;

    InMemoryAssetContext(InMemoryAssetContext&&) noexcept = delete;

    static InMemoryAssetContext& instance();

    tbb::concurrent_hash_map<std::string, std::shared_ptr<ArAsset>> assets;

  private:
    InMemoryAssetContext() = default;
};

class InMemoryAsset : public ArAsset {
  public:
    InMemoryAsset(const std::vector<std::byte>& buffer);

    size_t GetSize() override;

    std::shared_ptr<const char> GetBuffer() override;

    size_t Read(void* buffer, size_t count, size_t offset) override;

    std::pair<ArchFile*, size_t> GetFileUnsafe() override;

  private:
    std::shared_ptr<char> _buffer;
    std::size_t _bufferSize;
};

class InMemoryAssetResolver : public ArPackageResolver {
  public:
    InMemoryAssetResolver();

    ~InMemoryAssetResolver() noexcept;

    std::string Resolve(const std::string& packagePath, const std::string& path) override;

    std::shared_ptr<ArAsset> OpenAsset(const std::string& packagePath, const std::string& resolvedPath) override;

    void BeginCacheScope(VtValue* cacheScopeData) override;

    void EndCacheScope(VtValue* cacheScopeData) override;
};

PXR_NAMESPACE_CLOSE_SCOPE
