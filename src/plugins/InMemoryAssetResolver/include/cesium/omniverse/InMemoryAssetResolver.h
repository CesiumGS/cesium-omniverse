#pragma once

#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/packageResolver.h>
#include <tbb/concurrent_hash_map.h>

#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

class InMemoryAsset;

class InMemoryAssetContext {
  public:
    using Accessor = tbb::concurrent_hash_map<std::string, std::shared_ptr<InMemoryAsset>>::accessor;
    using ConstAccessor = tbb::concurrent_hash_map<std::string, std::shared_ptr<InMemoryAsset>>::const_accessor;

    InMemoryAssetContext(const InMemoryAssetContext&) = delete;

    InMemoryAssetContext(InMemoryAssetContext&&) noexcept = delete;

    AR_API static InMemoryAssetContext& instance();

    AR_API void add(const std::string& name, std::vector<std::byte>&& buffer);
    AR_API void remove(const std::string& name);
    AR_API std::shared_ptr<InMemoryAsset> find(const std::string& name) const;

  private:
    tbb::concurrent_hash_map<std::string, std::shared_ptr<InMemoryAsset>> _assets;

    InMemoryAssetContext() = default;
};

class InMemoryAsset : public ArAsset {
  public:
    AR_API InMemoryAsset(const std::vector<std::byte>& buffer);

    AR_API size_t GetSize() override;

    AR_API std::shared_ptr<const char> GetBuffer() override;

    AR_API size_t Read(void* buffer, size_t count, size_t offset) override;

    AR_API std::pair<ArchFile*, size_t> GetFileUnsafe() override;

  private:
    std::shared_ptr<char> _buffer;
    size_t _bufferSize;
    size_t _referenceCount;

    friend InMemoryAssetContext;
};

class InMemoryAssetResolver : public ArPackageResolver {
  public:
    InMemoryAssetResolver();

    ~InMemoryAssetResolver() noexcept;

    AR_API std::string Resolve(const std::string& packagePath, const std::string& path) override;

    AR_API std::shared_ptr<ArAsset> OpenAsset(const std::string& packagePath, const std::string& resolvedPath) override;

    AR_API void BeginCacheScope(VtValue* cacheScopeData) override;

    AR_API void EndCacheScope(VtValue* cacheScopeData) override;
};

PXR_NAMESPACE_CLOSE_SCOPE
