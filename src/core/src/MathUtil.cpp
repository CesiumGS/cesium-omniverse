#include "cesium/omniverse/MathUtil.h"

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>

namespace cesium::omniverse::MathUtil {

EulerAngleOrder getReversedEulerAngleOrder(EulerAngleOrder eulerAngleOrder) {
    switch (eulerAngleOrder) {
        case EulerAngleOrder::XYZ:
            return EulerAngleOrder::ZYX;
        case EulerAngleOrder::XZY:
            return EulerAngleOrder::YZX;
        case EulerAngleOrder::YXZ:
            return EulerAngleOrder::ZXY;
        case EulerAngleOrder::YZX:
            return EulerAngleOrder::XZY;
        case EulerAngleOrder::ZXY:
            return EulerAngleOrder::YXZ;
        case EulerAngleOrder::ZYX:
            return EulerAngleOrder::XYZ;
    }

    return EulerAngleOrder::XYZ;
}

DecomposedEuler decomposeEuler(const glm::dmat4& matrix, EulerAngleOrder eulerAngleOrder) {
    glm::dvec3 scale;
    glm::dquat rotation;
    glm::dvec3 translation;
    glm::dvec3 skew;
    glm::dvec4 perspective;

    [[maybe_unused]] const auto decomposable = glm::decompose(matrix, scale, rotation, translation, skew, perspective);
    assert(decomposable);

    const auto rotationMatrix = glm::mat4_cast(rotation);
    glm::dvec3 rotationEuler(0.0);

    switch (eulerAngleOrder) {
        case EulerAngleOrder::XYZ:
            glm::extractEulerAngleXYZ(rotationMatrix, rotationEuler.x, rotationEuler.y, rotationEuler.z);
            break;
        case EulerAngleOrder::XZY:
            glm::extractEulerAngleXZY(rotationMatrix, rotationEuler.x, rotationEuler.z, rotationEuler.y);
            break;
        case EulerAngleOrder::YXZ:
            glm::extractEulerAngleYXZ(rotationMatrix, rotationEuler.y, rotationEuler.x, rotationEuler.z);
            break;
        case EulerAngleOrder::YZX:
            glm::extractEulerAngleYZX(rotationMatrix, rotationEuler.y, rotationEuler.z, rotationEuler.x);
            break;
        case EulerAngleOrder::ZXY:
            glm::extractEulerAngleZXY(rotationMatrix, rotationEuler.z, rotationEuler.x, rotationEuler.y);
            break;
        case EulerAngleOrder::ZYX:
            glm::extractEulerAngleZYX(rotationMatrix, rotationEuler.z, rotationEuler.y, rotationEuler.x);
            break;
    }

    return {translation, rotationEuler, scale};
}

Decomposed decompose(const glm::dmat4& matrix) {
    glm::dvec3 scale;
    glm::dquat rotation;
    glm::dvec3 translation;
    glm::dvec3 skew;
    glm::dvec4 perspective;

    [[maybe_unused]] const auto decomposable = glm::decompose(matrix, scale, rotation, translation, skew, perspective);
    assert(decomposable);

    return {translation, rotation, scale};
}

glm::dmat4 composeEuler(
    const glm::dvec3& translation,
    const glm::dvec3& rotation,
    const glm::dvec3& scale,
    EulerAngleOrder eulerAngleOrder) {
    const auto translationMatrix = glm::translate(glm::dmat4(1.0), translation);
    const auto scaleMatrix = glm::scale(glm::dmat4(1.0), scale);

    auto rotationMatrix = glm::dmat4(1.0);

    switch (eulerAngleOrder) {
        case EulerAngleOrder::XYZ:
            rotationMatrix = glm::eulerAngleXYZ(rotation.x, rotation.y, rotation.z);
            break;
        case EulerAngleOrder::XZY:
            rotationMatrix = glm::eulerAngleXZY(rotation.x, rotation.z, rotation.y);
            break;
        case EulerAngleOrder::YXZ:
            rotationMatrix = glm::eulerAngleYXZ(rotation.y, rotation.x, rotation.z);
            break;
        case EulerAngleOrder::YZX:
            rotationMatrix = glm::eulerAngleYZX(rotation.y, rotation.z, rotation.x);
            break;
        case EulerAngleOrder::ZXY:
            rotationMatrix = glm::eulerAngleZXY(rotation.z, rotation.x, rotation.y);
            break;
        case EulerAngleOrder::ZYX:
            rotationMatrix = glm::eulerAngleZYX(rotation.z, rotation.y, rotation.x);
            break;
    }

    return translationMatrix * rotationMatrix * scaleMatrix;
}

glm::dmat4 compose(const glm::dvec3& translation, const glm::dquat& rotation, const glm::dvec3& scale) {
    const auto translationMatrix = glm::translate(glm::dmat4(1.0), translation);
    const auto rotationMatrix = glm::mat4_cast(rotation);
    const auto scaleMatrix = glm::scale(glm::dmat4(1.0), scale);

    return translationMatrix * rotationMatrix * scaleMatrix;
}

bool equal(const CesiumGeospatial::Cartographic& a, const CesiumGeospatial::Cartographic& b) {
    const auto& aVec = *reinterpret_cast<const glm::dvec3*>(&a);
    const auto& bVec = *reinterpret_cast<const glm::dvec3*>(&b);
    return aVec == bVec;
}

bool epsilonEqual(const CesiumGeospatial::Cartographic& a, const CesiumGeospatial::Cartographic& b, double epsilon) {
    const auto& aVec = *reinterpret_cast<const glm::dvec3*>(&a);
    const auto& bVec = *reinterpret_cast<const glm::dvec3*>(&b);
    return glm::all(glm::epsilonEqual(aVec, bVec, epsilon));
}

bool epsilonEqual(const glm::dmat4& a, const glm::dmat4& b, double epsilon) {
    return glm::all(glm::epsilonEqual(a[0], b[0], epsilon)) && glm::all(glm::epsilonEqual(a[1], b[1], epsilon)) &&
           glm::all(glm::epsilonEqual(a[2], b[2], epsilon)) && glm::all(glm::epsilonEqual(a[3], b[3], epsilon));
}

bool epsilonEqual(const glm::dvec3& a, const glm::dvec3& b, double epsilon) {
    return glm::all(glm::epsilonEqual(a, b, epsilon));
}

bool epsilonEqual(const glm::dquat& a, const glm::dquat& b, double epsilon) {
    return glm::all(glm::epsilonEqual(a, b, epsilon));
}

glm::dvec3 getCorner(const std::array<glm::dvec3, 2>& extent, uint64_t index) {
    return {
        (index & 1) ? extent[1].x : extent[0].x,
        (index & 2) ? extent[1].y : extent[0].y,
        (index & 4) ? extent[1].z : extent[0].z,
    };
}

std::array<glm::dvec3, 2> transformExtent(const std::array<glm::dvec3, 2>& extent, const glm::dmat4& transform) {
    const auto min = std::numeric_limits<double>::lowest();
    const auto max = std::numeric_limits<double>::max();

    glm::dvec3 transformedMin(max);
    glm::dvec3 transformedMax(min);

    for (uint64_t i = 0; i < 8; ++i) {
        const auto position = MathUtil::getCorner(extent, i);
        const auto transformedPosition = glm::dvec3(transform * glm::dvec4(position, 1.0));

        transformedMin = glm::min(transformedMin, transformedPosition);
        transformedMax = glm::max(transformedMax, transformedPosition);
    }

    return {{transformedMin, transformedMax}};
}

} // namespace cesium::omniverse::MathUtil
