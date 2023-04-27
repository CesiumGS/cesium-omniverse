#pragma once

#include <algorithm>
#include <memory>
#include <queue>

namespace cesium::omniverse {

template <typename T> class ObjectPool {
  public:
    ObjectPool() {}

    virtual ~ObjectPool() = default;

    std::shared_ptr<T> acquire() {
        const auto percentActive = computePercentActive();

        if (percentActive > _doublingThreshold) {
            const auto newCapacity = std::max(_capacity * 2, uint64_t(1));
            setCapacity(newCapacity);
        }

        const auto object = _queue.front();
        _queue.pop();
        setActive(object, true);

        return object;
    }

    void release(std::shared_ptr<T> object) {
        _queue.push(object);
        setActive(object, false);
    }

    uint64_t getCapacity() const {
        return _capacity;
    }

    uint64_t getNumberActive() const {
        return getCapacity() - getNumberInactive();
    }

    uint64_t getNumberInactive() const {
        return _queue.size();
    }

    double computePercentActive() const {
        const auto numberActive = getNumberActive();
        const auto capacity = getCapacity();

        if (capacity == 0) {
            return 1.0;
        }

        return static_cast<double>(numberActive) / static_cast<double>(capacity);
    }

    void setCapacity(uint64_t capacity) {
        const auto oldCapacity = _capacity;
        const auto newCapacity = capacity;

        assert(newCapacity >= oldCapacity);

        if (capacity < newCapacity) {
            // We can't shrink capacity because it would mean destroying objects currently in use
            return;
        }

        const auto count = newCapacity - oldCapacity;

        for (uint64_t i = 0; i < count; i++) {
            _queue.push(createObject(_objectId++));
            _capacity++;
        }
    }

  protected:
    virtual std::shared_ptr<T> createObject(uint64_t objectId) = 0;
    virtual void setActive(std::shared_ptr<T> object, bool active) = 0;

  private:
    std::queue<std::shared_ptr<T>> _queue;
    uint64_t _objectId = 0;
    uint64_t _capacity = 0;
    double _doublingThreshold = 0.75;
};

} // namespace cesium::omniverse
