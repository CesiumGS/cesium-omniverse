#include "doctestUtils.h"

#include <cesium/omniverse/ObjectPool.h>
#include <doctest/doctest.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <queue>

#define MAX_TESTED_POOL_SIZE 1024 // the max size pool to randomly generate

// ObjectPool is a virtual class so we cannot directly instantiate it for
// testing, and instatiating the classes that implement it (FabricGeometryPool
// and FabricMaterialPool) requires mocking more complicated classes, so we
// create a bare-bones class here.

class mockObj {
  public:
    mockObj(uint64_t objectId) {
        this->id = objectId;
        this->active = false;
    };
    uint64_t id;
    bool active;
};

class mockObjPool final : public cesium::omniverse::ObjectPool<mockObj> {
  protected:
    std::shared_ptr<mockObj> createObject(uint64_t objectId) override {
        return std::make_shared<mockObj>(objectId);
    };
    void setActive(const std::shared_ptr<mockObj> obj, bool active) override {
        obj->active = active;
    };
};

void testRandomSequenceOfCmds(mockObjPool* opl, int numEvents, bool setCap) {

    // track the objects we've acquired so we can release them
    std::queue<std::shared_ptr<mockObj>> activeObjs;

    // the total number of acquires performed, which becomes the minimum
    // expected size of the pool
    auto maxActiveCount = opl->getNumberActive();

    // perform a random sequence of acquires/releases while
    // ensuring we only release what we've acquired
    for (int i = 0; i < numEvents; i++) {
        if (!activeObjs.empty() && rand() % 2 == 0) {
            opl->release(activeObjs.front());
            activeObjs.pop();
        } else {
            activeObjs.push(opl->acquire());
        }
        maxActiveCount = std::max(maxActiveCount, activeObjs.size());

        if (setCap && i == numEvents / 2) {
            // at the halfway point, try resetting the capacity

            // TODO uncomment and place both setCapacity calls in subcases once
            // assert workaround has been found
            // see https://github.com/CesiumGS/cesium-omniverse/issues/342
            // // ensure we don't rollover 0
            // uint64_t oldCap = opl->getCapacity();
            // uint64_t guaranteedSmaller = std::min((uint64_t)0, oldCap - 1);
            // opl->setCapacity(guaranteedSmaller);
            // // verify that we cannot shrink the capacity
            // CHECK(oldCap == opl->getCapacity());

            // ensure the new size is GTE, avoiding rollover
            uint64_t guaranteedGTE =
                std::max(opl->getCapacity(), opl->getCapacity() + (uint64_t)(rand() % MAX_TESTED_POOL_SIZE));
            opl->setCapacity(guaranteedGTE);
        }
    }

    auto numActive = activeObjs.size();

    // ensure our math matches
    CHECK(opl->getNumberActive() == numActive);

    // make sure there's capacity for all objects
    CHECK(opl->getCapacity() >= numActive + opl->getNumberInactive());
    CHECK(opl->getCapacity() >= maxActiveCount);

    // the percent active is calculated out of the pool's total capacity
    // which must be gte our max observed active count
    CHECK(opl->computePercentActive() <= (float)numActive / (float)maxActiveCount);
}

// ---- Begin tests ----

TEST_SUITE("Test ObjectPool") {
    TEST_CASE("test initializiation") {
        mockObjPool opl = mockObjPool();

        SUBCASE("initial capacity") {
            CHECK(opl.getCapacity() == 0);
        }
        SUBCASE("initial active") {
            CHECK(opl.getNumberActive() == 0);
        }
        SUBCASE("initial inactive") {
            CHECK(opl.getNumberInactive() == 0);
        }
        SUBCASE("initial percent active") {
            // initial percent active is assumed to be 100% in parts of the code
            CHECK(opl.computePercentActive() == 1);
        }
    }

    TEST_CASE("test acquire/release") {

        mockObjPool opl = mockObjPool();

        // Generate a random number of actions to perform
        int numEvents;
        std::list<int> randEventCounts;

        fillWithRandomInts(&randEventCounts, 0, MAX_TESTED_POOL_SIZE, NUM_TEST_REPETITIONS);

        SUBCASE("test repeated acquires") {
            DOCTEST_VALUE_PARAMETERIZED_DATA(numEvents, randEventCounts);

            for (int i = 0; i < numEvents; i++) {
                opl.acquire();
            }

            CHECK(opl.getNumberActive() == numEvents);
            CHECK(opl.getCapacity() >= numEvents);
        }

        SUBCASE("test random acquire/release patterns") {
            DOCTEST_VALUE_PARAMETERIZED_DATA(numEvents, randEventCounts);
            testRandomSequenceOfCmds(&opl, numEvents, false);
        }

        SUBCASE("test random setting capacity") {
            DOCTEST_VALUE_PARAMETERIZED_DATA(numEvents, randEventCounts);
            testRandomSequenceOfCmds(&opl, numEvents, true);
        }
    }
}
