#include "doctestUtils.h"

#include <cesium/omniverse/ObjectPool.h>
#include <doctest/doctest.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <queue>

constexpr int MAX_TESTED_POOL_SIZE = 1024; // The max size pool to randomly generate

// ObjectPool is a virtual class so we cannot directly instantiate it for
// testing, and instatiating the classes that implement it (FabricGeometryPool
// and FabricMaterialPool) requires mocking more complicated classes, so we
// create a bare-bones class here.

class MockObject {
  public:
    MockObject(uint64_t objectId) {
        id = objectId;
        active = false;
    };
    uint64_t id;
    bool active;
};

class MockObjectPool final : public cesium::omniverse::ObjectPool<MockObject> {
  protected:
    std::shared_ptr<MockObject> createObject(uint64_t objectId) override {
        return std::make_shared<MockObject>(objectId);
    };
    void setActive(const std::shared_ptr<MockObject> obj, bool active) override {
        obj->active = active;
    };
};

void testRandomSequenceOfCmds(MockObjectPool &opl, int numEvents, bool setCap) {
    // Track the objects we've acquired so we can release them
    std::queue<std::shared_ptr<MockObject>> activeObjects;

    // The total number of acquires performed, which becomes the minimum
    // expected size of the pool
    auto maxActiveCount = opl.getNumberActive();

    // Perform a random sequence of acquires/releases while
    // ensuring we only release what we've acquired
    for (int i = 0; i < numEvents; i++) {
        if (!activeObjects.empty() && rand() % 2 == 0) {
            opl.release(activeObjects.front());
            activeObjects.pop();
        } else {
            activeObjects.push(opl.acquire());
        }
        maxActiveCount = std::max(maxActiveCount, activeObjects.size());

        if (setCap && i == numEvents / 2) {
            // At the halfway point, try resetting the capacity

            // Ensure the new size is GTE, avoiding rollover
            uint64_t guaranteedGTE =
                std::max(opl.getCapacity(), opl.getCapacity() +
                static_cast<uint64_t>(rand() % MAX_TESTED_POOL_SIZE));
            opl.setCapacity(guaranteedGTE);
        }
    }

    auto numActive = activeObjects.size();

    // Ensure our math matches
    CHECK(opl.getNumberActive() == numActive);

    // Make sure there's capacity for all objects
    CHECK(opl.getCapacity() >= numActive + opl.getNumberInactive());
    CHECK(opl.getCapacity() >= maxActiveCount);

    // The percent active is calculated out of the pool's total capacity
    // which must be gte our max observed active count
    CHECK(opl.computePercentActive() <= (float)numActive / (float)maxActiveCount);
}

// ---- Begin tests ----

TEST_SUITE("Test ObjectPool") {
    TEST_CASE("Test initializiation") {
        MockObjectPool opl = MockObjectPool();

        SUBCASE("Initial capacity") {
            CHECK(opl.getCapacity() == 0);
        }
        SUBCASE("Initial active") {
            CHECK(opl.getNumberActive() == 0);
        }
        SUBCASE("Initial inactive") {
            CHECK(opl.getNumberInactive() == 0);
        }
        SUBCASE("Initial percent active") {
            // Initial percent active is assumed to be 100% in parts of the code
            CHECK(opl.computePercentActive() == 1);
        }
    }

    TEST_CASE("Test acquire/release") {

        MockObjectPool opl = MockObjectPool();

        // Generate a random number of actions to perform
        int numEvents;
        std::list<int> randEventCounts;

        fillWithRandomInts(randEventCounts, 0, MAX_TESTED_POOL_SIZE, NUM_TEST_REPETITIONS);

        SUBCASE("Test repeated acquires") {
            DOCTEST_VALUE_PARAMETERIZED_DATA(numEvents, randEventCounts);

            for (int i = 0; i < numEvents; i++) {
                opl.acquire();
            }

            CHECK(opl.getNumberActive() == numEvents);
            CHECK(opl.getCapacity() >= numEvents);
        }

        SUBCASE("Test random acquire/release patterns") {
            DOCTEST_VALUE_PARAMETERIZED_DATA(numEvents, randEventCounts);
            testRandomSequenceOfCmds(opl, numEvents, false);
        }

        SUBCASE("Test random setting capacity") {
            DOCTEST_VALUE_PARAMETERIZED_DATA(numEvents, randEventCounts);
            testRandomSequenceOfCmds(opl, numEvents, true);
        }
    }
}
