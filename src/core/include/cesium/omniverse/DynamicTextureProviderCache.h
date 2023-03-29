#pragma once

#include <omni/ui/ImageProvider/DynamicTextureProvider.h>
#include <tbb/concurrent_hash_map.h>

namespace cesium::omniverse {

class DynamicTextureProviderCacheItem;

class DynamicTextureProviderCache {
  public:
    using Accessor = tbb::concurrent_hash_map<std::string, std::shared_ptr<DynamicTextureProviderCacheItem>>::accessor;
    using ConstAccessor =
        tbb::concurrent_hash_map<std::string, std::shared_ptr<DynamicTextureProviderCacheItem>>::const_accessor;

    DynamicTextureProviderCache(const DynamicTextureProviderCache&) = delete;
    DynamicTextureProviderCache(DynamicTextureProviderCache&&) = delete;

    static DynamicTextureProviderCache& getInstance() {
        static DynamicTextureProviderCache instance;
        return instance;
    }

    DynamicTextureProviderCache& operator=(const DynamicTextureProviderCache&) = delete;
    DynamicTextureProviderCache& operator=(DynamicTextureProviderCache) = delete;

    void insert(const std::string& name, std::unique_ptr<omni::ui::DynamicTextureProvider> dynamicTextureProvider);
    void addReference(const std::string& name);
    void removeReference(const std::string& name);
    bool contains(const std::string& name) const;

  protected:
    DynamicTextureProviderCache() = default;
    ~DynamicTextureProviderCache() = default;

  private:
    tbb::concurrent_hash_map<std::string, std::shared_ptr<DynamicTextureProviderCacheItem>> _items;
};

class DynamicTextureProviderCacheItem {
  public:
    DynamicTextureProviderCacheItem(std::unique_ptr<omni::ui::DynamicTextureProvider> dynamicTextureProvider);

  private:
    std::unique_ptr<omni::ui::DynamicTextureProvider> _dynamicTextureProvider;
    size_t _referenceCount;

    friend DynamicTextureProviderCache;
};

} // namespace cesium::omniverse
