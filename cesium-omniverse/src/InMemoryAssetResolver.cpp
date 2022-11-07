#include "InMemoryAssetResolver.h"

#include <pxr/usd/ar/definePackageResolver.h>

PXR_NAMESPACE_OPEN_SCOPE

AR_DEFINE_PACKAGE_RESOLVER(InMemoryAssetResolver, ArPackageResolver)

InMemoryAssetContext& InMemoryAssetContext::instance() {
    static InMemoryAssetContext ctx;
    return ctx;
}

InMemoryAsset::InMemoryAsset(const std::vector<std::byte>& buffer)
    : _buffer{reinterpret_cast<char*>(std::malloc(buffer.size())), [](char* data) { std::free(data); }}
    , _bufferSize{buffer.size()} {
    std::memcpy(_buffer.get(), buffer.data(), buffer.size());
}

size_t InMemoryAsset::GetSize() {
    return _bufferSize;
}

std::shared_ptr<const char> InMemoryAsset::GetBuffer() {
    return _buffer;
}

size_t InMemoryAsset::Read(void* buffer, size_t count, size_t offset) {
    if (offset >= _bufferSize) {
        return 0;
    }

    if (offset + count > _bufferSize) {
        count = _bufferSize - offset;
    }

    std::memcpy(buffer, _buffer.get() + offset, count);
    return count;
}

std::pair<ArchFile*, size_t> InMemoryAsset::GetFileUnsafe() {
    return std::pair<ArchFile*, size_t>(nullptr, 0);
}

InMemoryAssetResolver::InMemoryAssetResolver() {}

InMemoryAssetResolver::~InMemoryAssetResolver() noexcept = default;

std::string InMemoryAssetResolver::Resolve([[maybe_unused]] const std::string& packagePath, const std::string& path) {
    return path;
}

std::shared_ptr<ArAsset>
InMemoryAssetResolver::OpenAsset([[maybe_unused]] const std::string& packagePath, const std::string& resolvedPath) {
    auto& ctx = InMemoryAssetContext::instance();

    {
        InMemoryAssetContext::const_accessor const_accessor;
        auto found = ctx.assets.find(const_accessor, resolvedPath);
        if (found) {
            return const_accessor->second;
        }
    }

    return nullptr;
}

void InMemoryAssetResolver::BeginCacheScope([[maybe_unused]] VtValue* cacheScopeData) {}

void InMemoryAssetResolver::EndCacheScope([[maybe_unused]] VtValue* cacheScopeData) {}

PXR_NAMESPACE_CLOSE_SCOPE
