#include <CesiumGltfReader/GltfReader.h>
#include "testUtils.h"

#include "cesium/omniverse/GltfAccessors.h"
#include "cesium/omniverse/GltfUtil.h"

#include <CesiumGltf/Material.h>
#include <CesiumGltf/MeshPrimitive.h>
#include <CesiumGltf/Model.h>
#include <doctest/doctest.h>
#include <sys/types.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gsl/span>
#include <yaml-cpp/yaml.h>

const std::string ASSET_DIR = "tests/testAssets/gltfs";
const std::string CONFIG_PATH = "tests/configs/gltfConfig.yaml";

// simplifies casting when comparing some material queries to expected output from config
bool operator==(const pxr::GfVec3f& v3, const std::vector<float>& v) {
    return v.size() == 3 && v3[0] == v[0] && v3[1] == v[1] && v3[2] == v[2];
}

TEST_SUITE("gltf utils") {
    TEST_CASE("IndicesAccessor smoke test") {
        uint64_t data;
        std::list<uint64_t> dataContainer = {42, 64, 8675309, 21};

        DOCTEST_VALUE_PARAMETERIZED_DATA(data, dataContainer);

        CHECK(cesium::omniverse::IndicesAccessor(data).size() == data);
    }

    void checkGltfExpectedResults(const std::filesystem::path& gltfFileName, const YAML::Node& expectedResults) {

        // --- Load Gltf ---
        std::ifstream gltfStream(gltfFileName, std::ifstream::binary);
        std::stringstream gltfBuf;
        gltfBuf << gltfStream.rdbuf();
        CesiumGltfReader::GltfReader reader;
        auto gltf = reader.readGltf(gsl::span(reinterpret_cast<const std::byte*>(gltfBuf.str().c_str()), gltfBuf.str().size()));

        if (!gltf.errors.empty()) {
            for (const auto& err : gltf.errors) {
                std::cerr << err;
            }
            throw std::runtime_error("failed to parse model");
        }

        // gltf.model is a std::optional<CesiumGltf::Model>, make sure it exists
        if (!(gltf.model && gltf.model->meshes.size() > 0)) {
            throw std::runtime_error("test model is empty");
        }

        // --- Begin checks ---
        const auto& prim = gltf.model->meshes[0].primitives[0];
        const auto& model = *gltf.model;

        namespace gltfUtil = cesium::omniverse::GltfUtil;

        CHECK(gltfUtil::hasNormals(model, prim, true) == expectedResults["hasNormals"].as<bool>());
        CHECK(gltfUtil::hasTexcoords(model, prim, 0) == expectedResults["hasTexcoords"].as<bool>());
        CHECK(gltfUtil::hasImageryTexcoords(model, prim, 0) == expectedResults["hasImageryTexcoords"].as<bool>());
        CHECK(gltfUtil::hasVertexColors(model, prim, 0) == expectedResults["hasVertexColors"].as<bool>());
        CHECK(gltfUtil::hasMaterial(prim) == expectedResults["hasMaterial"].as<bool>());
        CHECK(gltfUtil::getDoubleSided(model, prim) == expectedResults["doubleSided"].as<bool>());

        // material tests
        if (gltfUtil::hasMaterial(prim)) {
            const auto& mat = gltf.model->materials[0];
            CHECK(gltfUtil::getAlphaMode(mat) == expectedResults["alphaMode"].as<int>());
            CHECK(gltfUtil::getAlphaCutoff(mat) == expectedResults["alphaCutoff"].as<float>());
            CHECK(gltfUtil::getBaseAlpha(mat) == expectedResults["baseAlpha"].as<float>());
            CHECK(gltfUtil::getMetallicFactor(mat) == expectedResults["metallicFactor"].as<float>());
            CHECK(gltfUtil::getRoughnessFactor(mat) == expectedResults["roughnessFactor"].as<float>());
            CHECK(gltfUtil::getBaseColorTextureWrapS(model, mat) == expectedResults["baseColorTextureWrapS"].as<int>());
            CHECK(gltfUtil::getBaseColorTextureWrapT(model, mat) == expectedResults["baseColorTextureWrapT"].as<int>());

            CHECK(gltfUtil::getBaseColorFactor(mat) == expectedResults["baseColorFactor"].as<std::vector<float>>());
            CHECK(gltfUtil::getEmissiveFactor(mat) == expectedResults["emissiveFactor"].as<std::vector<float>>());
        }

        // Accessor smoke tests
        cesium::omniverse::PositionsAccessor positions;
        cesium::omniverse::IndicesAccessor indices;
        CHECK_NOTHROW(positions = gltfUtil::getPositions(model, prim));
        CHECK_NOTHROW(indices = gltfUtil::getIndices(model, prim, positions));
        if (gltfUtil::hasNormals(model, prim, true)) {
            CHECK_NOTHROW(gltfUtil::getNormals(model, prim, positions, indices, false));
        }
        if (gltfUtil::hasVertexColors(model, prim, 0)) {
            CHECK_NOTHROW(gltfUtil::getVertexColors(model, prim, 0));
        }
        if (gltfUtil::hasTexcoords(model, prim, 0)) {
            CHECK_NOTHROW(gltfUtil::getTexcoords(model, prim, 0));
        }
        if (gltfUtil::hasImageryTexcoords(model, prim, 0)) {
            CHECK_NOTHROW(gltfUtil::getImageryTexcoords(model, prim, 0));
        }
        CHECK_NOTHROW(gltfUtil::getExtent(model, prim));

        // Default getter smoke tests
        CHECK_NOTHROW(gltfUtil::getDefaultBaseAlpha());
        CHECK_NOTHROW(gltfUtil::getDefaultBaseColorFactor());
        CHECK_NOTHROW(gltfUtil::getDefaultMetallicFactor());
        CHECK_NOTHROW(gltfUtil::getDefaultRoughnessFactor());
        CHECK_NOTHROW(gltfUtil::getDefaultEmissiveFactor());
        CHECK_NOTHROW(gltfUtil::getDefaultWrapS());
        CHECK_NOTHROW(gltfUtil::getDefaultWrapT());
        CHECK_NOTHROW(gltfUtil::getDefaultAlphaCutoff());
        CHECK_NOTHROW(gltfUtil::getDefaultAlphaMode());
    }

    TEST_CASE("Check helper functions on various models") {

        std::vector<std::string> gltfFiles;

        // get list of gltf test files
        for (auto const& i : std::filesystem::directory_iterator(ASSET_DIR)) {
            std::filesystem::path fname = i.path().filename();
            if (fname.extension() == ".gltf" || fname.extension() == ".glb") {
                gltfFiles.push_back(fname.string());
            }
        }

        // parse test config yaml
        const auto configRoot = YAML::LoadFile(CONFIG_PATH);
        const auto basePath = std::filesystem::path(ASSET_DIR);

        for (auto const& fileName : gltfFiles) {
            // attach filename to any failed checks
            CAPTURE(fileName);

            const auto conf = getScenarioConfig(fileName, configRoot);

            // the / operator concatonates file paths
            checkGltfExpectedResults(basePath / fileName, conf);
        }
    }
}
