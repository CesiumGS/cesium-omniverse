#include "cesium/omniverse/DynamicTextureProviderCache.h"

namespace cesium::omniverse {

void DynamicTextureProviderCache::insert(
    const std::string& name,
    std::unique_ptr<omni::ui::DynamicTextureProvider> dynamicTextureProvider) {
    _items.insert({name, std::make_shared<DynamicTextureProviderCacheItem>(std::move(dynamicTextureProvider))});
}

void DynamicTextureProviderCache::addReference(const std::string& name) {
    DynamicTextureProviderCache::Accessor accessor;
    auto found = _items.find(accessor, name);
    if (found) {
        accessor->second->_referenceCount++;
    }
}

void DynamicTextureProviderCache::removeReference(const std::string& name) {
    bool removeItem = false;

    {
        // Make sure the accessor goes out of scope and the item is unlocked before removing the item
        DynamicTextureProviderCache::Accessor accessor;
        auto found = _items.find(accessor, name);
        if (found) {
            auto& referenceCount = accessor->second->_referenceCount;
            referenceCount--;
            if (referenceCount == 0) {
                removeItem = true;
            }
        }
    }

    if (removeItem) {
        _items.erase(name);
    }
}

bool DynamicTextureProviderCache::contains(const std::string& name) const {
    DynamicTextureProviderCache::ConstAccessor accessor;
    return _items.find(accessor, name);
}

DynamicTextureProviderCacheItem::DynamicTextureProviderCacheItem(
    std::unique_ptr<omni::ui::DynamicTextureProvider> dynamicTextureProvider)
    : _dynamicTextureProvider(std::move(dynamicTextureProvider))
    , _referenceCount(1) {}

} // namespace cesium::omniverse
