#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <array>

namespace CesiumGeospatial {
class Cartographic;
}

namespace cesium::omniverse::MathUtil {

enum class EulerAngleOrder {
    XYZ,
    XZY,
    YXZ,
    YZX,
    ZXY,
    ZYX,
};

struct DecomposedEuler {
    glm::dvec3 translation;
    glm::dvec3 rotation;
    glm::dvec3 scale;
};

struct Decomposed {
    glm::dvec3 translation;
    glm::dquat rotation;
    glm::dvec3 scale;
};

DecomposedEuler decomposeEuler(const glm::dmat4& matrix, EulerAngleOrder eulerAngleOrder);
Decomposed decompose(const glm::dmat4& matrix);

glm::dmat4 composeEuler(
    const glm::dvec3& translation,
    const glm::dvec3& rotation,
    const glm::dvec3& scale,
    EulerAngleOrder eulerAngleOrder);
bool equal(const CesiumGeospatial::Cartographic& a, const CesiumGeospatial::Cartographic& b);
bool epsilonEqual(const CesiumGeospatial::Cartographic& a, const CesiumGeospatial::Cartographic& b, double epsilon);
bool epsilonEqual(const glm::dmat4& a, const glm::dmat4& b, double epsilon);
bool epsilonEqual(const glm::dvec3& a, const glm::dvec3& b, double epsilon);

glm::dvec3 getCorner(const std::array<glm::dvec3, 2>& extent, uint64_t index);

std::array<glm::dvec3, 2> transformExtent(const std::array<glm::dvec3, 2>& extent, const glm::dmat4& transform);

} // namespace cesium::omniverse::MathUtil
