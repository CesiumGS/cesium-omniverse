#include "doctestUtils.h"

void fillWithRandomInts(std::list<int>& lst, int min, int max, int n) {

    for (int i = 0; i < n; i++) {
        // The odd order here is to avoid issues with rollover
        int x = (rand() % (max - min)) + min;
        lst.push_back(x);
    }
}
