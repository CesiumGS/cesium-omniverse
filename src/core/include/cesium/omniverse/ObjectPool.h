#pragma once

#include <algorithm>
#include <cassert>
#include <memory>
#include <queue>

namespace cesium::omniverse {

template <typename T> class ObjectPool {
  public:
    ObjectPool() = default;

    virtual ~ObjectPool() = default;

    std::shared_ptr<T> acquire() {
        const auto percentActive = computePercentActive();

        if (percentActive > _doublingThreshold) {
            // Capacity is initially 0, so make sure the new capacity is at least 1
            const auto newCapacity = std::max(_capacity * 2, uint64_t(1));
            setCapacity(newCapacity);
        }

        const auto object = _queue.front();
        _queue.pop_front();
        setActive(object, true);

        return object;
    }

    void release(std::shared_ptr<T> object) {
        _queue.push_back(object);
        setActive(object, false);
    }

    [[nodiscard]] uint64_t getCapacity() const {
        return _capacity;
    }

    [[nodiscard]] uint64_t getNumberActive() const {
        return getCapacity() - getNumberInactive();
    }

    [[nodiscard]] uint64_t getNumberInactive() const {
        return _queue.size();
    }

    [[nodiscard]] bool isEmpty() const {
        return getNumberInactive() == getCapacity();
    }

    [[nodiscard]] double computePercentActive() const {
        const auto numberActive = static_cast<double>(getNumberActive());
        const auto capacity = static_cast<double>(getCapacity());

        if (capacity == 0) {
            return 1.0;
        }

        return numberActive / capacity;
    }

    void setCapacity(uint64_t capacity) {
        const auto oldCapacity = _capacity;
        const auto newCapacity = capacity;

        assert(newCapacity >= oldCapacity);

        if (newCapacity <= oldCapacity) {
            // We can't shrink capacity because it would mean destroying objects currently in use
            return;
        }

        const auto count = newCapacity - oldCapacity;

        for (uint64_t i = 0; i < count; i++) {
            _queue.push_back(createObject(_objectId++));
            _capacity++;
        }
    }

  protected:
    virtual std::shared_ptr<T> createObject(uint64_t objectId) = 0;
    virtual void setActive(std::shared_ptr<T> object, bool active) = 0;

    const std::deque<std::shared_ptr<T>>& getQueue() {
        return _queue;
    }

  private:
    std::deque<std::shared_ptr<T>> _queue;
    uint64_t _objectId = 0;
    uint64_t _capacity = 0;
    double _doublingThreshold = 0.75;
};

} // namespace cesium::omniverse
