#include "cesium/omniverse/InMemoryAssetResolver.h"

#include <pxr/usd/ar/definePackageResolver.h>

PXR_NAMESPACE_OPEN_SCOPE

AR_DEFINE_PACKAGE_RESOLVER(InMemoryAssetResolver, ArPackageResolver)

InMemoryAssetContext& InMemoryAssetContext::instance() {
    static InMemoryAssetContext ctx;
    return ctx;
}

void InMemoryAssetContext::add(const std::string& name, std::vector<std::byte>&& buffer) {
    InMemoryAssetContext::Accessor accessor;
    auto found = _assets.find(accessor, name);
    if (found) {
        accessor->second->_referenceCount++;
    } else {
        _assets.insert({name, std::make_shared<InMemoryAsset>(std::move(buffer))});
    }
}

void InMemoryAssetContext::remove(const std::string& name) {
    InMemoryAssetContext::Accessor accessor;
    auto found = _assets.find(accessor, name);
    if (found) {
        auto& referenceCount = accessor->second->_referenceCount;
        referenceCount--;
        if (referenceCount == 0) {
            _assets.erase(name);
        }
    }
}

std::shared_ptr<InMemoryAsset> InMemoryAssetContext::find(const std::string& name) const {
    InMemoryAssetContext::Accessor accessor;
    auto found = _assets.find(accessor, name);
    if (found) {
        return accessor->second;
    }

    return nullptr;
}

InMemoryAsset::InMemoryAsset(const std::vector<std::byte>& buffer)
    : _buffer{reinterpret_cast<char*>(std::malloc(buffer.size())), [](char* data) { std::free(data); }}
    , _bufferSize{buffer.size()}
    , _referenceCount(1) {
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
    const auto& ctx = InMemoryAssetContext::instance();
    return ctx.find(resolvedPath);
}

void InMemoryAssetResolver::BeginCacheScope([[maybe_unused]] VtValue* cacheScopeData) {}

void InMemoryAssetResolver::EndCacheScope([[maybe_unused]] VtValue* cacheScopeData) {}

PXR_NAMESPACE_CLOSE_SCOPE
