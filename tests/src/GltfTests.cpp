#include "testUtils.h"

#include "cesium/omniverse/FabricMaterialInfo.h"
#include "cesium/omniverse/FabricVertexAttributeAccessors.h"
#include "cesium/omniverse/GltfUtil.h"

#include <CesiumGltf/Material.h>
#include <CesiumGltf/MeshPrimitive.h>
#include <CesiumGltf/Model.h>
#include <CesiumGltfReader/GltfReader.h>
#include <doctest/doctest.h>

#include <cstddef>
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

using namespace cesium::omniverse;

const std::string ASSET_DIR = "tests/testAssets/gltfs";
const std::string CONFIG_PATH = "tests/configs/gltfConfig.yaml";

// simplifies casting when comparing some material queries to expected output from config
bool operator==(const glm::dvec3& v3, const std::vector<double>& v) {
    return v.size() == 3 && v3[0] == v[0] && v3[1] == v[1] && v3[2] == v[2];
}

TEST_SUITE("Test GltfUtil") {
    void checkGltfExpectedResults(const std::filesystem::path& gltfFileName, const YAML::Node& expectedResults) {

        // --- Load Gltf ---
        std::ifstream gltfStream(gltfFileName, std::ifstream::binary);
        gltfStream.seekg(0, std::ios::end);
        auto gltfFileLength = gltfStream.tellg();
        gltfStream.seekg(0, std::ios::beg);

        std::vector<std::byte> gltfBuf(static_cast<uint64_t>(gltfFileLength));
        gltfStream.read((char*)&gltfBuf[0], gltfFileLength);

        CesiumGltfReader::GltfReader reader;
        auto gltf = reader.readGltf(
            gsl::span(reinterpret_cast<const std::byte*>(gltfBuf.data()), static_cast<uint64_t>(gltfFileLength)));

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

        CHECK(GltfUtil::hasNormals(model, prim, false) == expectedResults["hasNormals"].as<bool>());
        CHECK(GltfUtil::hasTexcoords(model, prim, 0) == expectedResults["hasTexcoords"].as<bool>());
        CHECK(GltfUtil::hasRasterOverlayTexcoords(model, prim, 0) == expectedResults["hasRasterOverlayTexcoords"].as<bool>());
        CHECK(GltfUtil::hasVertexColors(model, prim, 0) == expectedResults["hasVertexColors"].as<bool>());
        CHECK(GltfUtil::hasMaterial(prim) == expectedResults["hasMaterial"].as<bool>());

        // material tests
        if (GltfUtil::hasMaterial(prim)) {
            const auto& matInfo = GltfUtil::getMaterialInfo(model, prim);
            CHECK(matInfo.alphaCutoff == expectedResults["alphaCutoff"].as<double>());
            CHECK(matInfo.alphaMode == static_cast<FabricAlphaMode>(expectedResults["alphaMode"].as<int32_t>()));
            CHECK(matInfo.baseAlpha == expectedResults["baseAlpha"].as<double>());
            CHECK(matInfo.baseColorFactor == expectedResults["baseColorFactor"].as<std::vector<double>>());
            CHECK(matInfo.emissiveFactor == expectedResults["emissiveFactor"].as<std::vector<double>>());
            CHECK(matInfo.metallicFactor == expectedResults["metallicFactor"].as<double>());
            CHECK(matInfo.roughnessFactor == expectedResults["roughnessFactor"].as<double>());
            CHECK(matInfo.doubleSided == expectedResults["doubleSided"].as<bool>());
            CHECK(matInfo.hasVertexColors == expectedResults["hasVertexColors"].as<bool>());
        }

        // Accessor smoke tests
        PositionsAccessor positions;
        IndicesAccessor indices;
        positions = GltfUtil::getPositions(model, prim);
        CHECK(positions.size() > 0);
        indices = GltfUtil::getIndices(model, prim, positions);
        CHECK(indices.size() > 0);
        if (GltfUtil::hasNormals(model, prim, false)) {
            CHECK(GltfUtil::getNormals(model, prim, positions, indices, false).size() > 0);
        }
        if (GltfUtil::hasVertexColors(model, prim, 0)) {
            CHECK(GltfUtil::getVertexColors(model, prim, 0).size() > 0);
        }
        if (GltfUtil::hasTexcoords(model, prim, 0)) {
            CHECK(GltfUtil::getTexcoords(model, prim, 0).size() > 0);
        }
        if (GltfUtil::hasRasterOverlayTexcoords(model, prim, 0)) {
            CHECK(GltfUtil::getRasterOverlayTexcoords(model, prim, 0).size() > 0);
        }
        CHECK(GltfUtil::getExtent(model, prim) != std::nullopt);
    }

    TEST_CASE("Default getter smoke tests") {
        CHECK_NOTHROW(GltfUtil::getDefaultMaterialInfo());
        CHECK_NOTHROW(GltfUtil::getDefaultTextureInfo());
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
