#include "testUtils.h"

#include <unordered_map>
#include <variant>

#include <yaml-cpp/node/detail/iterator_fwd.h>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/node/type.h>
#include <yaml-cpp/yaml.h>

void fillWithRandomInts(std::list<int>& lst, int min, int max, int n) {

    for (int i = 0; i < n; i++) {
        // The odd order here is to avoid issues with rollover
        int x = (rand() % (max - min)) + min;
        lst.push_back(x);
    }
}

ConfigMap getScenarioConfig(const std::string& scenario, YAML::Node configRoot) {
    ConfigMap sConfig = ConfigMap();

    const auto& defaultConfig = configRoot["scenarios"]["default"];

    for (YAML::const_iterator it = defaultConfig.begin(); it != defaultConfig.end(); it++) {
        sConfig[it->first.as<std::string>()] = it->second;
    }

    const auto&  overrides = configRoot["scenarios"][scenario];

    for (auto it = overrides.begin(); it != overrides.end(); it++) {
        sConfig[it->first.as<std::string>()] = it->second;
    }

    return sConfig;
}
