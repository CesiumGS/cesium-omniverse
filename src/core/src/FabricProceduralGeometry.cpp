#include "cesium/omniverse/FabricProceduralGeometry.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/random.hpp>
#include <glm/trigonometric.hpp>
#include <omni/fabric/FabricUSD.h>
#include <omni/fabric/IFabric.h>
#include <carb/Framework.h>

// #include <omni/usd/omni.h>
// #include <omni/usd/UsdContextIncludes.h>
// #include <omni/usd/UsdContext.h>

#include "pxr/base/tf/token.h"

#include <pxr/usd/usd/prim.h>
#include <iostream>
#include <stdexcept>
#include <omni/gpucompute/GpuCompute.h>
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/xform.h"
#include "pxr/usd/usdGeom/xformable.h"
#include "pxr/usd/usdGeom/xformCommonAPI.h"
#include "pxr/usd/usdGeom/cube.h"


namespace cesium::omniverse::FabricProceduralGeometry {


constexpr int numPrimsForExperiment = 99;
glm::dvec3 lookatPositionHost{0.0, 0.0, 0.0};
glm::fvec3 lookatUpHost{0.0, 0.0, 0.0};
CudaRunner cudaRunner;
double elapsedTime = 0;

const omni::fabric::Type cudaTestAttributeFabricType(omni::fabric::BaseDataType::eDouble, 1, 0, omni::fabric::AttributeRole::eNone);
omni::fabric::Token getCudaTestAttributeFabricToken() {
    static const auto cudaTestAttributeFabricToken = omni::fabric::Token("cudaTest");
    return cudaTestAttributeFabricToken;
}

const omni::fabric::Type quadOrientationType(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::eNone);
omni::fabric::Token getQuadOrientationAttributeFabricToken() {
    static const auto quadOrientationAttributeFabricToken = omni::fabric::Token("quadOrientation");
    return quadOrientationAttributeFabricToken;
}

const omni::fabric::Type numQuadsFabricType(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
omni::fabric::Token getNumQuadsAttributeFabricToken() {
    static const auto quadOrientationAttributeFabricToken = omni::fabric::Token("numQuads");
    return quadOrientationAttributeFabricToken;
}

//CUDA via CUDA_JIT and string
const char* kernelCode = R"(
extern "C" __global__
void changeValue(double* values, size_t count)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (count <= i) return;

    float oldVal = values[i];
    values[i] = 543.21;
    printf("Changed value of index %llu from %lf to %lf\n", i, oldVal, values[i]);
}
)";

const char* modifyVec3fKernelCode = R"(
struct Vec3f
{
    float x;
    float y;
    float z;
};

extern "C" __global__
void setVec3f(Vec3f* values, size_t count)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (count <= i) return;

    float oldValX = values[i].x;
    values[i].x = 0;
    values[i].y = 1.0f;
    printf("Changed x value of index %llu from %f to %f\n", i, oldValX, values[i][0].x);
}
)";

const char* modifyVec3dKernelCode = R"(
struct Vec3d
{
    double x;
    double y;
    double z;
};

extern "C" __global__
void setVec3d(Vec3d* values, size_t count)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    double oldValX = values[i].x;
    values[i].x = static_cast<double>(i) * 10;
    values[i].y = static_cast<double>(i) * 10;
    printf("Changed x value of index %llu from %lf to %lf\n", i, oldValX, values[i].x);
}
)";

const char* randomizeVec3dKernelCode = R"(
__device__ double rand(int seed)
{
  seed = (1103515245 * seed + 12345) % (1 << 31);
  return ((double)seed) / ((1 << 31) - 1);  // This will return a double between 0 and 1
}

struct Vec3d
{
    double x;
    double y;
    double z;
};

extern "C" __global__
void randomizeVec3d(Vec3d* values, size_t count, int seed)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    unsigned long long state = 42 + i;

    double randomizationAmount = 200.0;
    double oldValX = values[i].x;

    double randX = rand(i + seed) * randomizationAmount - (randomizationAmount / 2.0);
    double randY = rand(i + seed + count) * randomizationAmount - (randomizationAmount / 2.0);
    double randZ = rand(i + seed + count * 2) * randomizationAmount - (randomizationAmount / 2.0);

    values[i].x = randX;
    values[i].y = randY;
    values[i].z = randZ;

    printf("Changed x value of index %llu from %lf to %lf\n", i, oldValX, values[i].x);
}
)";

const char* randomizeDVecKernelCode = R"(
#include <curand_kernel.h>

extern "C" __global__
void randomizeDVec3(double3* values, size_t count, unsigned int seed)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (count <= i) return;

    curandState state;
    curand_init(seed, i, 0, &state);

    double oldVal = values[i].z;
    values[i].x = 0;
    values[i].y = 0;
    values[i].z = curand_uniform_double(&state) * 2000;
    printf("Changed z value of index %zu from %lf to %lf via curand seed %d\n", i, oldVal, values[i].z, seed);
}
)";

const char* modifyVec3fArrayKernelCode = R"(
struct Vec3f
{
    float x;
    float y;
    float z;
};

extern "C" __global__
void changeValue(Vec3f** values, size_t count)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Vec3f* row = values[i];
    float oldValX = row[0].x;
    row[0].x = 0;
    row[0].y = 100.0f;
    printf("Changed x value of index %llu from %f to %f\n", i, oldValX, row[0].x);
}
)";

const char* cudaSimpleHeaderTest = R"(
#include "cudaTestHeader.h"

struct Vec3f
{
    float x;
    float y;
    float z;
};

extern "C" __global__
void runHeaderTest(Vec3f* values, size_t count)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (count <= i) return;

    cudaHello();

    // float oldValX = values[i].x;
    values[i].x = 0;
    values[i].y = 1.0f;
    // printf("Changed x value of index %llu from %f to %f\n", i, oldValX, values[i][0].x);
}
)";

const char* curandHeaderTest = R"(
#include <curand_kernel.h>

struct Vec3d
{
    double x;
    double y;
    double z;
};

extern "C" __global__
void runCurandTest(Vec3d* values, size_t count, unsigned int seed)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (count <= i) return;

    curandState state;
    curand_init(seed, i, 0, &state);

    double oldVal = values[i].z;
    values[i].x = 0;
    values[i].y = 0;
    values[i].z = curand_uniform_double(&state) * 2000;
    printf("Changed z value of index %zu from %lf to %lf via curand seed %d\n", i, oldVal, values[i].z, seed);
}
)";

const char* lookAtKernelCode = R"(
struct fquat
{
    float x;
    float y;
    float z;
    float w;

    __device__ fquat() : x(0), y(0), z(0), w(0) {}
    __device__ fquat(float _x, float _y, float _z, float _w) {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }
};

struct mat3 {
    float3 col0;
    float3 col1;
    float3 col2;
};

__device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(float3 v) {
    float normSquared = v.x * v.x + v.y * v.y + v.z * v.z;
    float inverseSqrtNorm = rsqrtf(normSquared);
    v.x *= inverseSqrtNorm;
    v.y *= inverseSqrtNorm;
    v.z *= inverseSqrtNorm;
    return v;
}

__device__ double3 normalize(double3 v) {
    double norm = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    double inverseNorm = 1.0 / norm;
    v.x *= inverseNorm;
    v.y *= inverseNorm;
    v.z *= inverseNorm;
    return v;
}

__device__ float3 cross(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float3 operator*(const float3 &a, const float &b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ fquat quat_cast(mat3 Result)
{
    float m00 = Result.col0.x, m01 = Result.col1.x, m02 = Result.col2.x,
           m10 = Result.col0.y, m11 = Result.col1.y, m12 = Result.col2.y,
           m20 = Result.col0.z, m21 = Result.col1.z, m22 = Result.col2.z;

    float t = m00 + m11 + m22;
    float w, x, y, z;

    if (t > 0.0) {
        float s = sqrt(t + 1.0f) * 2.0f; // s=4*qw
        w = 0.25f * s;
        x = (m21 - m12) / s;
        y = (m02 - m20) / s;
        z = (m10 - m01) / s;
    } else if ((m00 > m11) && (m00 > m22)) {
        float s = sqrt(1.0f + m00 - m11 - m22) * 2.0f; // s=4*qx
        w = (m21 - m12) / s;
        x = 0.25f * s;
        y = (m01 + m10) / s;
        z = (m02 + m20) / s;
    } else if (m11 > m22) {
        float s = sqrt(1.0 + m11 - m00 - m22) * 2.0f; // s=4*qy
        w = (m02 - m20) / s;
        x = (m01 + m10) / s;
        y = 0.25f * s;
        z = (m12 + m21) / s;
    } else {
        float s = sqrt(1.0f + m22 - m00 - m11) * 2.0f; // s=4*qz
        w = (m10 - m01) / s;
        x = (m02 + m20) / s;
        y = (m12 + m21) / s;
        z = 0.25f * s;
    }

    return fquat(x, y, z, w);
}

__device__ fquat quatLookAtRH(float3 direction, float3 up)
{
    mat3 Result;

    Result.col2 = make_float3(-direction.x, -direction.y, -direction.z);
    float3 Right = cross(up, Result.col2);
    Result.col0 = Right * rsqrtf(max(0.0000f, dot(Right, Right)));

    Result.col1 = cross(Result.col2, Result.col0);

    return quat_cast(Result);
}

extern "C" __global__ void lookAtKernel(fquat* orientation, double3* worldPositions, double3* lookatPosition, size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const float3 up = make_float3(0, 1.0, 0.0);

    double3 lookAtDirectionD = make_double3(lookatPosition->x - worldPositions[i].x, lookatPosition->y - worldPositions[i].y, lookatPosition->z - worldPositions[i].z);
    float3 lookAtDirection = make_float3(static_cast<float>(lookAtDirectionD.x), static_cast<float>(lookAtDirectionD.y), static_cast<float>(lookAtDirectionD.z));
    lookAtDirection = normalize(lookAtDirection);

    fquat result = quatLookAtRH(lookAtDirection, up);
    orientation[i] = result;
}
)";

const char* lookAtMultiquadKernelCode = R"(
struct fquat
{
    float x;
    float y;
    float z;
    float w;

    __device__ fquat() : x(0), y(0), z(0), w(0) {}
    __device__ fquat(float _x, float _y, float _z, float _w) {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }
};

__device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct mat3 {
    float3 col0;
    float3 col1;
    float3 col2;

    __host__ __device__ float3 multiply(float3 vec) const {
        float3 result;
        result.x = dot(col0, vec);
        result.y = dot(col1, vec);
        result.z = dot(col2, vec);
        return result;
    }
};

struct quad {
    float3 lowerLeft;
    float3 upperLeft;
    float3 upperRight;
    float3 lowerRight;

    __device__ float3 getCenter() {
        return make_float3(
            (lowerLeft.x + upperRight.x) * .5f,
            (lowerLeft.y + upperRight.y) * .5f,
            0);
    }
};

__device__ float3 normalize(float3 v) {
    float normSquared = v.x * v.x + v.y * v.y + v.z * v.z;
    float inverseSqrtNorm = rsqrtf(normSquared);
    v.x *= inverseSqrtNorm;
    v.y *= inverseSqrtNorm;
    v.z *= inverseSqrtNorm;
    return v;
}

__device__ double3 normalize(double3 v) {
    double norm = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    double inverseNorm = 1.0 / norm;
    v.x *= inverseNorm;
    v.y *= inverseNorm;
    v.z *= inverseNorm;
    return v;
}

__device__ float3 cross(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float3 operator*(const float3 &a, const float &b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ fquat quat_cast(mat3 Result)
{
    float m00 = Result.col0.x, m01 = Result.col1.x, m02 = Result.col2.x,
           m10 = Result.col0.y, m11 = Result.col1.y, m12 = Result.col2.y,
           m20 = Result.col0.z, m21 = Result.col1.z, m22 = Result.col2.z;

    float t = m00 + m11 + m22;
    float w, x, y, z;

    if (t > 0.0) {
        float s = sqrt(t + 1.0f) * 2.0f; // s=4*qw
        w = 0.25f * s;
        x = (m21 - m12) / s;
        y = (m02 - m20) / s;
        z = (m10 - m01) / s;
    } else if ((m00 > m11) && (m00 > m22)) {
        float s = sqrt(1.0f + m00 - m11 - m22) * 2.0f; // s=4*qx
        w = (m21 - m12) / s;
        x = 0.25f * s;
        y = (m01 + m10) / s;
        z = (m02 + m20) / s;
    } else if (m11 > m22) {
        float s = sqrt(1.0 + m11 - m00 - m22) * 2.0f; // s=4*qy
        w = (m02 - m20) / s;
        x = (m01 + m10) / s;
        y = 0.25f * s;
        z = (m12 + m21) / s;
    } else {
        float s = sqrt(1.0f + m22 - m00 - m11) * 2.0f; // s=4*qz
        w = (m10 - m01) / s;
        x = (m02 + m20) / s;
        y = (m12 + m21) / s;
        z = 0.25f * s;
    }

    return fquat(x, y, z, w);
}

__device__ mat3 matLookAtRH(float3 direction, float3 up)
{
    mat3 Result;

    Result.col2 = make_float3(-direction.x, -direction.y, -direction.z);
    float3 Right = cross(up, Result.col2);
    Result.col0 = Right * rsqrtf(max(0.0000f, dot(Right, Right)));

    Result.col1 = cross(Result.col2, Result.col0);

    return Result;
}

extern "C" __global__ void lookAtMultiquadKernel(quad* quads, double3* lookatPosition, int numQuads) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numQuads) return;

    printf("(kernel) numQuads: %d\n", numQuads);
    printf("Initial quad positions:\n");
    printf("  ll: %f, %f, %f\n", quads[i].lowerLeft);
    printf("  ul: %f, %f, %f\n", quads[i].upperLeft);
    printf("  ur: %f, %f, %f\n", quads[i].upperRight);
    printf("  lr: %f, %f, %f\n", quads[i].lowerRight);

    printf("(kernel) lookAtPosition is (%lf, %lf, %lf)\n", lookatPosition->x, lookatPosition->y, lookatPosition->z);

    const float3 up = make_float3(0, 1.0, 0.0);
    const float3 quadCenter = quads[i].getCenter();
    printf("(kernel) quadCenter is (%f, %f, %f)\n", quadCenter.x, quadCenter.y, quadCenter.z);

    double3 lookAtDirectionD = make_double3(lookatPosition->x - quadCenter.x, lookatPosition->y - quadCenter.y, lookatPosition->z - quadCenter.z);
    float3 lookAtDirection = make_float3(static_cast<float>(lookAtDirectionD.x), static_cast<float>(lookAtDirectionD.y), static_cast<float>(lookAtDirectionD.z));
    lookAtDirection = normalize(lookAtDirection);
    printf("(kernel) lookAtDirection is (%f, %f, %f)\n", lookAtDirection.x, lookAtDirection.y, lookAtDirection.z);

    mat3 rotationMatrix = matLookAtRH(lookAtDirection, up);
    quads[i].lowerLeft = rotationMatrix.multiply(quads[i].lowerLeft);
    quads[i].upperLeft = rotationMatrix.multiply(quads[i].upperLeft);
    quads[i].upperRight = rotationMatrix.multiply(quads[i].upperRight);
    quads[i].lowerRight = rotationMatrix.multiply(quads[i].lowerRight);
}
)";

const char* lookAtMultiquadKernelCode2 = R"(

__device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct mat3 {
    float3 col0;
    float3 col1;
    float3 col2;

    __device__ float3 multiply(const float3 &vec) const {
        float3 result;
        result.x = dot(make_float3(col0.x, col1.x, col2.x), vec);
        result.y = dot(make_float3(col0.y, col1.y, col2.y), vec);
        result.z = dot(make_float3(col0.z, col1.z, col2.z), vec);
        return result;
    }
};

struct quad {
    float3 lowerLeft;
    float3 upperLeft;
    float3 upperRight;
    float3 lowerRight;

    __device__ float3 getCenter() {
        return make_float3(
            (lowerLeft.x + upperRight.x) * .5f,
            (lowerLeft.y + upperRight.y) * .5f,
            0);
    }
};

__device__ float3 normalize(float3 v) {
    float normSquared = v.x * v.x + v.y * v.y + v.z * v.z;
    float inverseSqrtNorm = rsqrtf(normSquared);
    v.x *= inverseSqrtNorm;
    v.y *= inverseSqrtNorm;
    v.z *= inverseSqrtNorm;
    return v;
}

__device__ double3 normalize(double3 v) {
    double norm = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    double inverseNorm = 1.0 / norm;
    v.x *= inverseNorm;
    v.y *= inverseNorm;
    v.z *= inverseNorm;
    return v;
}

__device__ float3 cross(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float3 operator*(const float3 &a, const float &b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ bool almostEquals(float3 a, float3 b) {
    const float epsilon = 0.0000001f;
    if (abs(a.x - b.x) > epsilon) return false;
    if (abs(a.y - b.y) > epsilon) return false;
    if (abs(a.z - b.z) > epsilon) return false;

    return true;
}

__device__ double3 subtractDouble3(double3 a, double3 b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 addFloat3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

extern "C" __global__ void lookAtMultiquadKernel(quad** quads, double3* lookatPosition, float3* lookatUp, int numQuads) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numQuads) return;

    int quadIndex = static_cast<int>(i);

    float3 targetUpN = *lookatUp;
    float3 quadCenter = quads[0][quadIndex].getCenter();
    double3 quadCenterD = make_double3(static_cast<double>(quadCenter.x), static_cast<double>(quadCenter.y) , static_cast<double>(quadCenter.z));
    double3 newQuadForwardDouble = subtractDouble3(*lookatPosition, quadCenterD);
    float3 newQuadForward = make_float3(
        static_cast<float>(newQuadForwardDouble.x),
        static_cast<float>(newQuadForwardDouble.y),
        static_cast<float>(newQuadForwardDouble.z));
    float3 newQuadForwardN = normalize(newQuadForward);
    float3 newQuadRightN;
    float3 newQuadUpN;

    if (almostEquals(newQuadForwardN, targetUpN)) {
        //directly beneath the camera, no op
        printf("directly beneath camera, no op. returning\n");
        return;
    } else {
        newQuadRightN = normalize(cross(newQuadForwardN, targetUpN));
        newQuadUpN = normalize(cross(newQuadRightN, newQuadForward));
    }

    mat3 translationMatrix = {newQuadRightN, newQuadUpN, newQuadForwardN};

    //untransformed quad points are assumed to be in XY plane
    float3 rotatedLL = translationMatrix.multiply(make_float3(-1.0f, -1.0f, 0));
    float3 rotatedUL = translationMatrix.multiply(make_float3(-1.0f, 1.0f, 0));
    float3 rotatedUR = translationMatrix.multiply(make_float3(1.0f, 1.0f, 0));
    float3 rotatedLR = translationMatrix.multiply(make_float3(1.0f, -1.0f, 0));
    float3 newQuadUL = addFloat3(rotatedUL, quadCenter);
    float3 newQuadUR = addFloat3(rotatedUR, quadCenter);
    float3 newQuadLL = addFloat3(rotatedLL, quadCenter);
    float3 newQuadLR = addFloat3(rotatedLR, quadCenter);

    quads[0][quadIndex].upperLeft = newQuadUL;
    quads[0][quadIndex].upperRight = newQuadUR;
    quads[0][quadIndex].lowerLeft = newQuadLL;
    quads[0][quadIndex].lowerRight = newQuadLR;
}
)";

const char* printMultiquadKernel = R"(

struct quad {
    float3 lowerLeft;
    float3 upperLeft;
    float3 upperRight;
    float3 lowerRight;
};

extern "C" __global__ void printMultiquad(quad** quads, int numQuads) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numQuads) return;

    int quadIndex = static_cast<int>(i);

    printf("Quad positions for quad %d:\n", quadIndex);
    printf("  ll: %f, %f, %f\n", quads[0][quadIndex].lowerLeft.x, quads[0][quadIndex].lowerLeft.y, quads[0][quadIndex].lowerLeft.z);
    printf("  ul: %f, %f, %f\n", quads[0][quadIndex].upperLeft.x, quads[0][quadIndex].upperLeft.y, quads[0][quadIndex].upperLeft.z);
    printf("  ur: %f, %f, %f\n", quads[0][quadIndex].upperRight.x, quads[0][quadIndex].upperRight.y, quads[0][quadIndex].upperRight.z);
    printf("  lr: %f, %f, %f\n", quads[0][quadIndex].lowerRight.x, quads[0][quadIndex].lowerRight.y, quads[0][quadIndex].lowerRight.z);
}
)";

const char* printPointsKernel = R"(

extern "C" __global__ void printPoints(float3** points, int numPoints) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) return;

    int pointIndex = static_cast<int>(i);

    printf("point %d: %f, %f, %f\n", pointIndex, points[0][pointIndex].x, points[0][pointIndex].y, points[0][pointIndex].z);
}
)";

int createPrims() {

    // modifyUsdCubePrimWithFabric();
    // modify1000UsdCubePrimsWithFabric();
    // modify1000UsdCubesViaCuda();
    // modify1000UsdQuadsViaCuda();
    // editSingleFabricAttributeViaCuda();
    // createQuadViaFabricAndShiftWithCuda();
    // modifyAllPrimsWithCustomAttrViaCuda();
    // createFabricQuadsModifyViaCuda(numPrimsForExperiment);

    // alterUsdPrimTranslationWithUsd();
    // alterUsdPrimTranslationWithFabric();
    // alterFabricPrimTranslationWithFabric();

    // createQuadMeshViaUsd("/testQuadMesh", 0);
    // setDisplayColor();
    // createQuadsViaFabric(10);

    // createQuadsViaFabric(80000, 1000.f);
    // createMultiquadViaFabric();
    createMultiquadMeshViaFabric2(500);
    // createSingleQuad(pxr::GfVec3f(3.f, -3.f, 0), 2);
    // createSingleQuad(pxr::GfVec3f(3.f, 3.f, -3.0f), 2);

    return 0;
}

int alterPrims(double cameraPositionX, double cameraPositionY, double cameraPositionZ,
    float cameraUpX, float cameraUpY, float cameraUpZ) {
    // printf("camera position is %lf, %lf, %lf\n", cameraPositionX, cameraPositionY, cameraPositionZ);

    //DEBUG
    // cameraPositionX = 4;
    // cameraPositionY = 0;
    // cameraPositionZ = 0;

    //trick clang

    auto cameraPositionf = glm::fvec3(
        static_cast<float>(cameraPositionX),
        static_cast<float>(cameraPositionY),
        static_cast<float>(cameraPositionZ));

    // repositionAllPrimsWithCustomAttrViaFabric(200);
    // repositionAllPrimsWithCustomAttrViaCuda(200);
    // modifyAllPrimsWithCustomAttrViaCuda();
    // randomizePrimWorldPositionsWithCustomAttrViaCuda();
    // rotateAllPrimsWithCustomAttrViaFabric();
    // billboardAllPrimsWithCustomAttrViaFabric();
    // billboardAllPrimsWithCustomAttrViaCuda();
    // billboardMultiquadWithCustomAttrViaFabric();
    // billboardQuad(glm::fvec3{10.f, 0, 0});
    // billboardMultiQuadCpu(cameraPositionf, glm::fvec3(cameraUpX, cameraUpY, cameraUpZ));
    billboardMultiQuadCuda(cameraPositionf, glm::fvec3(cameraUpX, cameraUpY, cameraUpZ));
    // billboardMultiquadWithCustomAttrViaCuda();
    // printMultiquadWithCuda();
    // printMultiquadPointsWithCuda();
    // printPositionsWithFabric();
    // runSimpleCudaHeaderTest();
    // runCurandHeaderTest();
    // exportToUsd();
    // randomizeDVec3ViaCuda();
    return 0;
}

void modifyUsdCubePrimWithFabric() {
    //Linker error getting UsdContext using omni::usd
    // auto context = omni::usd::UsdContext::getContext();
    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();
    omni::fabric::StageReaderWriter stageReaderWriter = UsdUtil::getFabricStageReaderWriter();
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    long id = Context::instance().getStageId();
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};

    //create a cube in USD and set its size.
    pxr::UsdPrim prim = usdStagePtr->DefinePrim(pxr::SdfPath("/TestCube"), pxr::TfToken("Cube"));
    auto sizeUsdToken = pxr::TfToken("size");
    prim.CreateAttribute(sizeUsdToken, pxr::SdfValueTypeNames->Double).Set(3.0);
    auto customAttrUsdToken = pxr::TfToken("customTestAttr");
    prim.CreateAttribute(customAttrUsdToken, pxr::SdfValueTypeNames->Double).Set(123.45);

    auto gprim = pxr::UsdGeomGprim(prim);
    auto displayColorPrimvar = gprim.CreateDisplayColorPrimvar(UsdTokens::constant);
    auto displayColor = pxr::VtArray<pxr::GfVec3f>(1);
    displayColor[0] = pxr::GfVec3f(0, 1.0f, 1.0f);
    displayColorPrimvar.Set(displayColor);
    // prim.CreateAttribute(pxr::TfToken("primvars:displayColor"), pxr::SdfValueTypeNames->).Set(displayColor);

    //prefetch it to Fabric’s cache.
    omni::fabric::Path primPath("/TestCube");
    iStageReaderWriter->prefetchPrim(usdStageId, primPath);

    //use Fabric to modify the cube’s dimensions
    auto sizeFabricToken = omni::fabric::Token("size");
    auto customAttrFabricToken = omni::fabric::Token("customTestAttr");

    double& size = *stageReaderWriter.getAttribute<double>(primPath, sizeFabricToken);
    double sizeTarget = 30;
    size = sizeTarget;

    double& customAttr = *stageReaderWriter.getAttribute<double>(primPath, customAttrFabricToken);
    double customAttrTarget = 987.654;
    customAttr = customAttrTarget;

    //write our changes back to USD.
    //NOTE: does not work
    const auto iFabricUsd = carb::getCachedInterface<omni::fabric::IFabricUsd>();
    auto fabricId = omni::fabric::FabricId();
    iFabricUsd->exportUsdPrimData(fabricId);

    //check that Fabric correctly modified the USD stage.
    pxr::UsdAttribute verifyAttr = prim.GetAttribute(customAttrUsdToken);
    double value;
    verifyAttr.Get(&value);
    if (value == customAttrTarget) {
        std::cout << "modified stage" << std::endl;
    } else {
        std::cout << "did not modify stage" << std::endl;
    }
}

void modify1000UsdCubePrimsWithFabric() {
    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();

    //use USD to make a thousand cubes
    const size_t cubeCount = 1000;
    auto customAttrUsdToken = pxr::TfToken("cudaTest");
    for (size_t i = 0; i != cubeCount; i++)
    {
        pxr::SdfPath path("/cube_" + std::to_string(i));
        pxr::UsdPrim prim = usdStagePtr->DefinePrim(path, pxr::TfToken("Cube"));
        prim.CreateAttribute(pxr::TfToken("size"), pxr::SdfValueTypeNames->Double).Set(3.3);
        prim.CreateAttribute(customAttrUsdToken, pxr::SdfValueTypeNames->Double).Set(123.45);
    }

    //call prefetchPrim to get the data into Fabric.
    long id = Context::instance().getStageId();
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    for (size_t i = 0; i != cubeCount; i++)
    {
        omni::fabric::Path path(("/cube_" + std::to_string(i)).c_str());
        iStageReaderWriter->prefetchPrim(usdStageId, path);
    }

    //tell Fabric which prims to change. Select all prims of type Cube.
    omni::fabric::AttrNameAndType cubeTag(
        omni::fabric::Type(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName),
        omni::fabric::Token("Cube"));

    const auto stageReaderWriterId = iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(id)});
    auto fabricReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::PrimBucketList cubeBuckets = fabricReaderWriter.findPrims({ cubeTag });

    // Fabric is free to store the 1000 cubes in as many buckets as it likes...iterate over the buckets
    int counter = 0;
    auto fabricSizeToken = omni::fabric::Token("size");
    for (size_t bucket = 0; bucket != cubeBuckets.bucketCount(); bucket++)
    {
        auto sizes = fabricReaderWriter.getAttributeArray<double>(cubeBuckets, bucket, fabricSizeToken);
        for (double& size : sizes)
        {
            size = 77.7;
        }

        auto testValues = fabricReaderWriter.getAttributeArray<double>(cubeBuckets, bucket, getCudaTestAttributeFabricToken());
        for (double& testValue : testValues)
        {
            testValue = 54.321;
        }

        counter++;
    }
    std::cout << "Modified " << counter << " prims" << std::endl;
}

void modify1000UsdCubesViaCuda() {
    const int cubeCount = 1000;

    const auto usdStagePtr = Context::instance().getStage();
    const auto cudaTestAttrUsdToken = pxr::TfToken("cudaTest");
    for (size_t i = 0; i != cubeCount; i++)
    {
        pxr::SdfPath path("/cube_" + std::to_string(i));
        pxr::UsdPrim prim = usdStagePtr->DefinePrim(path, pxr::TfToken("Cube"));
        //note that "size" is not altered by CUDA kernel
        prim.CreateAttribute(pxr::TfToken("size"), pxr::SdfValueTypeNames->Double).Set(17.3);
        prim.CreateAttribute(cudaTestAttrUsdToken, pxr::SdfValueTypeNames->Double).Set(12.3);
    }

    auto id = Context::instance().getStageId();
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    for (size_t i = 0; i != cubeCount; i++)
    {
        omni::fabric::Path path(("/cube_" + std::to_string(i)).c_str());
        // This guarantees that subsequent gets of the prim from the cache will succeed
        iStageReaderWriter->prefetchPrim(usdStageId, path);
    }

    modifyAllPrimsWithCustomAttrViaCuda();
}

//a way to compile without using nvrtc
//note: did not work, but might now work with DLLs correctly handled
CUfunction compileKernel(const char *kernelSource, const char *kernelName) {

    CUfunction kernel_func;
    CUmodule module;

    // Set up JIT compilation options and compile the module
    // CUjit_option jitOptions[] = { //NOLINT
    //     CU_JIT_TARGET_FROM_CUCONTEXT,
    //     CU_JIT_OPTIMIZATION_LEVEL,
    // };

    // void *jitOptVals[] = { //NOLINT
    //     nullptr, // Target is picked from CUcontext
    //     (void *)3 // Optimization level 3
    // };

    const unsigned int jitNumOptions = 2;
    CUjit_option jitOptions[jitNumOptions]; //NOLINT
    void*        jitOptVals[jitNumOptions]; //NOLINT

    // Set up JIT compilation options to print verbose log
    jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    int jitLogBufferSize = 1024;
    jitOptVals[0] = (void *)(size_t)jitLogBufferSize; //NOLINT

    char* jitLogBuffer = new char[jitLogBufferSize];
    jitOptions[1] = CU_JIT_INFO_LOG_BUFFER; //NOLINT
    jitOptVals[1] = jitLogBuffer; //NOLINT

    // Compile the module
    CUresult result = cuModuleLoadDataEx(&module, kernelSource, 2, jitOptions, jitOptVals);
    //attempt to use cuModuleLoadData instead of cuModuleLoadDataEx to see if it solves the CUDA_ERROR_INVALID_IMAGE (200) issue. (It did not.)
    //CUresult result = cuModuleLoadData(&module, kernelSource);

    std::cout << "JIT compilation log:\n" << jitLogBuffer << std::endl;
    delete[] jitLogBuffer;

    if (result != CUDA_SUCCESS) {
        const char *error;
        cuGetErrorString(result, &error);
        std::cerr << "Could not compile module: " << error << std::endl;
        return nullptr;
    }

    // Get kernel function
    result = cuModuleGetFunction(&kernel_func, module, kernelName);
    if (result != CUDA_SUCCESS) {
        const char *error;
        cuGetErrorString(result, &error);
        std::cerr << "Could not get kernel function: " << error << std::endl;
        return nullptr;
    }

    return kernel_func;
}


CUfunction compileKernel2(const char *kernelSource, const char *kernelName) {
    nvrtcProgram prog;
    nvrtcResult result = nvrtcCreateProgram(&prog, kernelSource, kernelName, 0, nullptr, nullptr);
    if (result != NVRTC_SUCCESS) {
        std::cout << "Failed to create the program." << std::endl;
    }

    // Compile the program
    auto compileResult = nvrtcCompileProgram(prog, 0, nullptr);
    if (compileResult != NVRTC_SUCCESS) {
        std::cout << "Failed to compile the program." << std::endl;
    }

    // Get compilation log
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    char* log = new char[logSize];
    nvrtcGetProgramLog(prog, log);
    std::cout << "Compilation log: \n" << log << std::endl;

    // Get the PTX code
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char *ptx = new char[ptxSize];
    result = nvrtcGetPTX(prog, ptx);
    if (result != NVRTC_SUCCESS) {
        std::cout << "Failed to get the PTX code." << std::endl;
    }

    // Load the PTX code into a CUDA module
    CUmodule module;
    CUresult moduleLoadResult = cuModuleLoadData(&module, ptx);
    if (moduleLoadResult != CUDA_SUCCESS) {
        std::cout << "Failed to load the module." << std::endl;
    }

    // Get the kernel function from the module
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, kernelName);

    delete[] log;
    delete[] ptx;

    return kernel;
}

bool checkCudaCompatibility() {
    int runtimeVer, driverVer;

    // Get CUDA Runtime version
    if (cudaRuntimeGetVersion(&runtimeVer) != cudaSuccess) {
        std::cerr << "Failed to get CUDA Runtime version" << std::endl;
        return false;
    }

    // Get CUDA driver version
    if (cudaDriverGetVersion(&driverVer) != cudaSuccess) {
        std::cerr << "Failed to get CUDA driver version" << std::endl;
        return false;
    }

    std::cout << "CUDA Runtime Version: " << runtimeVer / 1000 << "." << (runtimeVer % 100) / 10 << std::endl;
    std::cout << "CUDA Driver Version: " << driverVer / 1000 << "." << (driverVer % 100) / 10 << std::endl;

    // Check compatibility
    if (runtimeVer > driverVer) {
        std::cerr << "CUDA Toolkit version is greater than the driver version. "
                  << "Please update the CUDA driver." << std::endl;
        return false;
    }

    return true;
}

void createQuadMeshViaFabric() {
    createQuadsViaFabric(1);
}

void createQuadViaFabricAndShiftWithCuda() {
    const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = Context::instance().getStageId();

    const auto stageReaderWriterId =
        iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(usdStageId)});
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    omni::fabric::Path fabricPath = omni::fabric::Path("/fabricMeshQuad");
    stageReaderWriter.createPrim(fabricPath);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.createAttributes(fabricPath);

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexCounts, 2);
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexIndices, 6);
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::points, 4);

    stageReaderWriter.createAttribute(fabricPath, getCudaTestAttributeFabricToken(), cudaTestAttributeFabricType);
    auto customAttrWriter = stageReaderWriter.getAttribute<float>(fabricPath, getCudaTestAttributeFabricToken());
    *customAttrWriter = 17.3f;

    auto pointsFabric = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    float extentScalar = 50;
    pointsFabric[0] = pxr::GfVec3f(-extentScalar, -extentScalar, 0);
    pointsFabric[1] = pxr::GfVec3f(-extentScalar, extentScalar, 0);
    pointsFabric[2] = pxr::GfVec3f(extentScalar, extentScalar, 0);
    pointsFabric[3] = pxr::GfVec3f(extentScalar, -extentScalar, 0);

    auto faceVertexCountsFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexCounts);
    faceVertexCountsFabric[0] = 3;
    faceVertexCountsFabric[1] = 3;

    auto faceVertexIndicesFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexIndices);
    faceVertexIndicesFabric[0] = 0;
    faceVertexIndicesFabric[1] = 1;
    faceVertexIndicesFabric[2] = 2;
    faceVertexIndicesFabric[3] = 0;
    faceVertexIndicesFabric[4] = 2;
    faceVertexIndicesFabric[5] = 3;

    auto extent = pxr::GfRange3d(pxr::GfVec3d(-extentScalar, -extentScalar, 0), pxr::GfVec3d(extentScalar, extentScalar, 0));
    auto extentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::extent);
    *extentFabric = extent;

    auto worldExtentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::_worldExtent);
    *worldExtentFabric = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));

    auto worldVisibilityFabric = stageReaderWriter.getAttributeWr<bool>(fabricPath, FabricTokens::_worldVisibility);
    *worldVisibilityFabric = true;

    auto worldPositionFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3d>(fabricPath, FabricTokens::_worldPosition);
    *worldPositionFabric = pxr::GfVec3d(0, 0, 0);

    auto worldOrientationFabric = stageReaderWriter.getAttributeWr<pxr::GfQuatf>(fabricPath, FabricTokens::_worldOrientation);
    *worldOrientationFabric = pxr::GfQuatf(1.f, 0, 0, 0);

    auto worldScaleFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::_worldScale);
    *worldScaleFabric = pxr::GfVec3f(1.f, 1.f, 1.f);


    // modify with CUDA
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
    }

    int major, minor;
    result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not get CUDA major version." << std::endl;
    }
    result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not get CUDA minor version." << std::endl;
    }

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
    }

    //scale with CUDA to be oblong
    //CUDA via CUDA_JIT and string
    //gets include error
    // const char* kernelCode = R"(
    // #include <pxr/base/gf/vec3f.h>

    // extern "C" __global__
    // void addTenToXComponentKernel(pxr::GfVec3f* data, size_t size)
    // {
    //     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //     if (idx < size) {
    //         data[idx].Set(data[idx].GetX() + 10.0f, data[idx].GetY(), data[idx].GetZ());
    //     }
    // }
    // )";

    const char* addToXKernelCode = R"(
    struct Triplet
    {
        float x;
        float y;
        float z;
    };

    extern "C" __global__
    void addTenToXComponentKernel(Triplet* data, size_t size)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) {
            data[idx].x += 10.0f;
            printf("Element %zu: x = %f\n", idx, data[idx].x);
        }
    }
    )";

    // CUfunction kernel = compileKernel(scaleCubes, "scaleCubes");
    CUfunction kernel = compileKernel2(addToXKernelCode, "addTenToXComponentKernel");

    struct Triplet
    {
        float x;
        float y;
        float z;
    };



    // // auto pointsFabric = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    // //auto pointsFabricGpu = stageReaderWriter.getAttributeWrGpu<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    // auto pointsFabricGpu = stageReaderWriter.getAttributeGpu<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    // auto* directData = pointsFabricGpu->data();
    // //direct float data access of GfVec3f
    // auto* triplets = reinterpret_cast<Triplet*>(directData);
    // size_t elemCount = pointsFabric.size();
    // void *args[] = { &triplets, &elemCount }; //NOLINT
    // int blockSize, minGridSize;
    // cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, nullptr, 0, 0);
    // auto err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
    // // REQUIRE(!err);
    // if (err) {
    //     std::cout << "error" << std::endl;
    // }

    //omni::fabric::AttrNameAndType quadTag(omni::fabric::BaseDataType::eToken, FabricTokens::points);
    omni::fabric::AttrNameAndType quadMeshTag(omni::fabric::Type(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName), omni::fabric::Token("Mesh"));
    omni::fabric::PrimBucketList quadBuckets = stageReaderWriter.findPrims({ quadMeshTag });
    auto bucketCount = quadBuckets.bucketCount();
    printf("Found %llu buckets\n", bucketCount);

    // omni::fabric::PrimBucketList quadBuckets = stageReaderWriter.findPrims({ quadTag });

    // //iterate over buckets but pass the vector for the whole bucket to the GPU.
    for (size_t bucket = 0; bucket != quadBuckets.bucketCount(); bucket++)
    {

        // Step 2: Get the device pointer to the data
        auto pointsD = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f>(quadBuckets, bucket, FabricTokens::points);
        pxr::GfVec3f* ptr = pointsD.data();
        //direct data access
        auto directData = ptr->data();
        auto* triplets = reinterpret_cast<Triplet*>(directData);
        size_t elemCount = pointsD.size();
        //size_t dataSize = elemCount * 3 * sizeof(float); // Assuming each point has x, y, and z components
        void *args[] = { &triplets, &elemCount }; //NOLINT
        int blockSize, minGridSize;
        cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, nullptr, 0, 0);
        auto err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        // REQUIRE(!err);
        if (err) {
            std::cout << "error" << std::endl;
        }
    }

    result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }
}

void addOneMillionCPU() {
    int N = 1<<20;

    auto* a = new float[N];
    auto* b = new float[N];

    for (int i = 0; i < 1000000; i++) {
        a[i] = 1.f;
        b[i] = 2.f;
    }

    addArrays(N, a, b);

    auto maxError = 0.f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, b[i] - 3.f);
    }
    std::cout << "max error: " << maxError << std::endl;

    cudaFree(a);
    cudaFree(b);
}

void addOneMillionCuda() {

    const char *addKernelCode = R"(
    extern "C" __global__
    void add(int n, float *x, float *y)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = index; i < n; i += stride) {
            y[i] = x[i] + y[i];
            // printf("threadIdx.x: %i, blockIdx.x: %i, blockDim.x: %i, gridDim.x: %i, y[%i]: %f\n", threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, index, y[i]);
        }
    }
    )";

  int N = 1<<20;
  N++;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  nvrtcProgram prog;
  nvrtcCreateProgram(&prog, addKernelCode, "add_program", 0, nullptr, nullptr);

  // Compile the program
  nvrtcResult res = nvrtcCompileProgram(prog, 0, nullptr);
  if (res != NVRTC_SUCCESS) {
    std::cout << "error compiling NVRTC program" << std::endl;
  }

  // Get the PTX (assembly code for the GPU) from the compilation
  size_t ptxSize;
  nvrtcGetPTXSize(prog, &ptxSize);
  char* ptx = new char[ptxSize];
  nvrtcGetPTX(prog, ptx);

  // Load the generated PTX and get a handle to the kernel.
  CUmodule module;
  CUfunction function;
  cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
  cuModuleGetFunction(&function, module, "add");

  // Set kernel parameters and launch the kernel.
  void *args[] = { &N, &x, &y }; //NOLINT
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  cuLaunchKernel(function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

}

__global__ void addArrays(int n, float* x, float* y) {
    for (auto i = 0; i < n; i++) {
        y[i] += x[i];
    }
}

void editSingleFabricAttributeViaCuda() {
   const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = Context::instance().getStageId();

    const auto stageReaderWriterId =
        iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(usdStageId)});
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    omni::fabric::Path fabricPath = omni::fabric::Path("/fabricMeshQuad");
    stageReaderWriter.createPrim(fabricPath);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.createAttributes(fabricPath);

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexCounts, 2);
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexIndices, 6);
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::points, 4);

    stageReaderWriter.createAttribute(fabricPath, getCudaTestAttributeFabricToken() , omni::fabric::BaseDataType::eFloat);
    auto customAttrWriter = stageReaderWriter.getAttribute<float>(fabricPath, getCudaTestAttributeFabricToken());
    *customAttrWriter = 12.3f;

    auto pointsFabric = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    float extentScalar = 50;
    pointsFabric[0] = pxr::GfVec3f(-extentScalar, -extentScalar, 0);
    pointsFabric[1] = pxr::GfVec3f(-extentScalar, extentScalar, 0);
    pointsFabric[2] = pxr::GfVec3f(extentScalar, extentScalar, 0);
    pointsFabric[3] = pxr::GfVec3f(extentScalar, -extentScalar, 0);

    auto faceVertexCountsFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexCounts);
    faceVertexCountsFabric[0] = 3;
    faceVertexCountsFabric[1] = 3;

    auto faceVertexIndicesFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexIndices);
    faceVertexIndicesFabric[0] = 0;
    faceVertexIndicesFabric[1] = 1;
    faceVertexIndicesFabric[2] = 2;
    faceVertexIndicesFabric[3] = 0;
    faceVertexIndicesFabric[4] = 2;
    faceVertexIndicesFabric[5] = 3;

    auto extent = pxr::GfRange3d(pxr::GfVec3d(-extentScalar, -extentScalar, 0), pxr::GfVec3d(extentScalar, extentScalar, 0));
    auto extentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::extent);
    *extentFabric = extent;

    auto worldExtentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::_worldExtent);
    *worldExtentFabric = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));

    auto worldVisibilityFabric = stageReaderWriter.getAttributeWr<bool>(fabricPath, FabricTokens::_worldVisibility);
    *worldVisibilityFabric = true;

    auto worldPositionFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3d>(fabricPath, FabricTokens::_worldPosition);
    *worldPositionFabric = pxr::GfVec3d(0, 0, 0);

    auto worldOrientationFabric = stageReaderWriter.getAttributeWr<pxr::GfQuatf>(fabricPath, FabricTokens::_worldOrientation);
    *worldOrientationFabric = pxr::GfQuatf(1.f, 0, 0, 0);

    auto worldScaleFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::_worldScale);
    *worldScaleFabric = pxr::GfVec3f(1.f, 1.f, 1.f);

    //select and bucket the Cube prims
    // omni::fabric::AttrNameAndType quadTag(
    //     omni::fabric::Type(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::ePrimTypeName),
    //     omni::fabric::Token("cudaTest"));
    omni::fabric::AttrNameAndType quadMeshTag(omni::fabric::Type(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName), omni::fabric::Token("Mesh"));
    omni::fabric::PrimBucketList cubeBuckets = stageReaderWriter.findPrims({ quadMeshTag });

    auto isCudaCompatible = checkCudaCompatibility();
    if (!isCudaCompatible) {
        std::cout << "error: CUDA drives and toolkit versions are not compatible." << std::endl;
    }

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
    }

    int major, minor;
    result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not get CUDA major version." << std::endl;
    }
    result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not get CUDA minor version." << std::endl;
    }

    std::cout << "Compute capability: " << major << "." << minor << std::endl;

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
    }

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, kernelCode, "changeValue", 0, nullptr, nullptr);

    // Compile the program
    nvrtcResult res = nvrtcCompileProgram(prog, 0, nullptr);
    if (res != NVRTC_SUCCESS) {
        std::cout << "error compiling NVRTC program" << std::endl;
        return;
    }

    // Get the PTX (assembly code for the GPU) from the compilation
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    // Load the generated PTX and get a handle to the kernel.
    CUmodule module;
    CUfunction function;
    cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    cuModuleGetFunction(&function, module, "changeValue");

    auto bucketCount = cubeBuckets.bucketCount();
    printf("Num buckets: %llu", bucketCount);

    for (size_t bucket = 0; bucket != cubeBuckets.bucketCount(); bucket++)
    {
        gsl::span<double> sizesD = stageReaderWriter.getAttributeArrayGpu<double>(cubeBuckets, bucket, getCudaTestAttributeFabricToken());

        double* ptr = sizesD.data();
        size_t elemCount = sizesD.size();
        void *args[] = { &ptr, &elemCount }; //NOLINT
        int blockSize, minGridSize;
        cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, nullptr, 0, 0);
        //CUresult err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, 0);
        auto err = cuLaunchKernel(function, minGridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        // REQUIRE(!err);
        if (err) {
            std::cout << "error" << std::endl;
        }
    }

    result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }
}


void createQuadsViaFabric(int numQuads, float maxCenterRandomization) {
    const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    const auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(cesium::omniverse::Context::instance().getStageId())};
    const auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    const std::string primPathStub{"/quadMesh_"};

    for (int i = 0; i < numQuads; i++) {
        const auto fabricPath = omni::fabric::Path((primPathStub + std::to_string(i)).c_str());
        stageReaderWriter.createPrim(fabricPath);

        FabricAttributesBuilder attributes;
        attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
        attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
        attributes.addAttribute(FabricTypes::points, FabricTokens::points);
        attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
        attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
        attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
        attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
        attributes.addAttribute(FabricTypes::primvars_displayColor, FabricTokens::primvars_displayColor);
        attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
        attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
        // attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
        attributes.createAttributes(fabricPath);

        stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::points, 4);
        auto pointsFabric = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
        auto extentScalar = glm::linearRand(1.f, 8.f);
        pointsFabric[0] = pxr::GfVec3f(-extentScalar, -extentScalar, 0);
        pointsFabric[1] = pxr::GfVec3f(-extentScalar, extentScalar, 0);
        pointsFabric[2] = pxr::GfVec3f(extentScalar, extentScalar, 0);
        pointsFabric[3] = pxr::GfVec3f(extentScalar, -extentScalar, 0);

        stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexCounts, 2);
        auto faceVertexCountsFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexCounts);
        faceVertexCountsFabric[0] = 3;
        faceVertexCountsFabric[1] = 3;

        stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexIndices, 6);
        auto faceVertexIndicesFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexIndices);
        faceVertexIndicesFabric[0] = 0;
        faceVertexIndicesFabric[1] = 1;
        faceVertexIndicesFabric[2] = 2;
        faceVertexIndicesFabric[3] = 0;
        faceVertexIndicesFabric[4] = 2;
        faceVertexIndicesFabric[5] = 3;

        auto extent = pxr::GfRange3d(pxr::GfVec3d(-extentScalar, -extentScalar, 0), pxr::GfVec3d(extentScalar, extentScalar, 0));
        auto extentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::extent);
        *extentFabric = extent;

        auto worldExtentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::_worldExtent);
        *worldExtentFabric = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));

        auto worldVisibilityFabric = stageReaderWriter.getAttributeWr<bool>(fabricPath, FabricTokens::_worldVisibility);
        *worldVisibilityFabric = true;

        auto center = pxr::GfVec3d{
            glm::linearRand(-maxCenterRandomization, maxCenterRandomization),
            glm::linearRand(-maxCenterRandomization, maxCenterRandomization),
            glm::linearRand(-maxCenterRandomization, maxCenterRandomization)
        };

        auto worldPositionFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3d>(fabricPath, FabricTokens::_worldPosition);
        *worldPositionFabric = pxr::GfVec3d(0.0, 0.0, 0.0) + center;
        //DEBUG
        // *worldPositionFabric = pxr::GfVec3d(300.0, 300.0, 0.0);

        auto worldOrientationFabric = stageReaderWriter.getAttributeWr<pxr::GfQuatf>(fabricPath, FabricTokens::_worldOrientation);
        //*worldOrientationFabric = pxr::GfQuatf(1.f, 0, 0, 0);
        *worldOrientationFabric = pxr::GfQuatf(0.f, 0, 0, 0);

        // auto worldScaleFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::_worldScale);
        // *worldScaleFabric = pxr::GfVec3f(1.f, 1.f, 1.f);

        stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::primvars_displayColor, 1);
        auto displayColors = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::primvars_displayColor);
        displayColors[0] = pxr::GfVec3f(0.8f, 0.8f, 0.8f);

        //create a custom attribute for testing
        stageReaderWriter.createAttribute(fabricPath, getCudaTestAttributeFabricToken(), cudaTestAttributeFabricType);

        auto testAttribute = stageReaderWriter.getAttributeWr<double>(fabricPath, getCudaTestAttributeFabricToken());
        *testAttribute = 123.45;
    }
}

void createMultiquadViaFabric() {
    const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    const auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(cesium::omniverse::Context::instance().getStageId())};
    const auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    const std::string primPathStub{"/multiquadMesh"};
    const auto fabricPath = omni::fabric::Path((primPathStub + std::to_string(0)).c_str());
    stageReaderWriter.createPrim(fabricPath);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::primvars_displayColor, FabricTokens::primvars_displayColor);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    // attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.createAttributes(fabricPath);

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::points, 16);
    auto pointsFabric = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    // auto extentScalar = glm::linearRand(1.f, 8.f);
    const float unit = 1.f;
    pxr::GfVec3f quadShift{-unit * 2.f, unit * 2.f, 0};
    pointsFabric[0] = pxr::GfVec3f{-unit, -unit, 0} + quadShift;
    pointsFabric[1] = pxr::GfVec3f{-unit, unit, 0} + quadShift;
    pointsFabric[2] = pxr::GfVec3f{unit, unit, 0} + quadShift;
    pointsFabric[3] = pxr::GfVec3f{unit, -unit, 0} + quadShift;
    quadShift.Set(unit * 2.f, unit * 2.f, 0);
    pointsFabric[4] = pxr::GfVec3f{-unit, -unit, 0} + quadShift;
    pointsFabric[5] = pxr::GfVec3f{-unit, unit, 0} + quadShift;
    pointsFabric[6] = pxr::GfVec3f{unit, unit, 0} + quadShift;
    pointsFabric[7] = pxr::GfVec3f{unit, -unit, 0} + quadShift;
    quadShift.Set(unit * 2.f, unit * -2.f, 0);
    pointsFabric[8] = pxr::GfVec3f{-unit, -unit, 0} + quadShift;
    pointsFabric[9] = pxr::GfVec3f{-unit, unit, 0} + quadShift;
    pointsFabric[10] = pxr::GfVec3f{unit, unit, 0} + quadShift;
    pointsFabric[11] = pxr::GfVec3f{unit, -unit, 0} + quadShift;
    quadShift.Set(unit * -2.f, unit * -2.f, 0);
    pointsFabric[12] = pxr::GfVec3f{-unit, -unit, 0} + quadShift;
    pointsFabric[13] = pxr::GfVec3f{-unit, unit, 0} + quadShift;
    pointsFabric[14] = pxr::GfVec3f{unit, unit, 0} + quadShift;
    pointsFabric[15] = pxr::GfVec3f{unit, -unit, 0} + quadShift;

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexCounts, 8);
    auto faceVertexCountsFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexCounts);
    faceVertexCountsFabric[0] = 3;
    faceVertexCountsFabric[1] = 3;
    faceVertexCountsFabric[2] = 3;
    faceVertexCountsFabric[3] = 3;
    faceVertexCountsFabric[4] = 3;
    faceVertexCountsFabric[5] = 3;
    faceVertexCountsFabric[6] = 3;
    faceVertexCountsFabric[7] = 3;

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexIndices, 24);
    auto faceVertexIndicesFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexIndices);
    faceVertexIndicesFabric[0] = 0;
    faceVertexIndicesFabric[1] = 1;
    faceVertexIndicesFabric[2] = 2;
    faceVertexIndicesFabric[3] = 0;
    faceVertexIndicesFabric[4] = 2;
    faceVertexIndicesFabric[5] = 3;

    faceVertexIndicesFabric[0 + 6] = 0 + 4;
    faceVertexIndicesFabric[1 + 6] = 1 + 4;
    faceVertexIndicesFabric[2 + 6] = 2 + 4;
    faceVertexIndicesFabric[3 + 6] = 0 + 4;
    faceVertexIndicesFabric[4 + 6] = 2 + 4;
    faceVertexIndicesFabric[5 + 6] = 3 + 4;

    faceVertexIndicesFabric[0 + 12] = 0 + 8;
    faceVertexIndicesFabric[1 + 12] = 1 + 8;
    faceVertexIndicesFabric[2 + 12] = 2 + 8;
    faceVertexIndicesFabric[3 + 12] = 0 + 8;
    faceVertexIndicesFabric[4 + 12] = 2 + 8;
    faceVertexIndicesFabric[5 + 12] = 3 + 8;

    faceVertexIndicesFabric[0 + 18] = 0 + 12;
    faceVertexIndicesFabric[1 + 18] = 1 + 12;
    faceVertexIndicesFabric[2 + 18] = 2 + 12;
    faceVertexIndicesFabric[3 + 18] = 0 + 12;
    faceVertexIndicesFabric[4 + 18] = 2 + 12;
    faceVertexIndicesFabric[5 + 18] = 3 + 12;


    auto extent = pxr::GfRange3d(pxr::GfVec3d(-unit * 3.f, -unit * 3.f, 0), pxr::GfVec3d(unit * 3.f, unit * 3.f, 0));
    auto extentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::extent);
    *extentFabric = extent;

    auto worldExtentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::_worldExtent);
    *worldExtentFabric = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));

    auto worldVisibilityFabric = stageReaderWriter.getAttributeWr<bool>(fabricPath, FabricTokens::_worldVisibility);
    *worldVisibilityFabric = true;

    // auto center = pxr::GfVec3d{
    //     glm::linearRand(-maxCenterRandomization, maxCenterRandomization),
    //     glm::linearRand(-maxCenterRandomization, maxCenterRandomization),
    //     glm::linearRand(-maxCenterRandomization, maxCenterRandomization)
    // };

    auto worldPositionFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3d>(fabricPath, FabricTokens::_worldPosition);
    *worldPositionFabric = pxr::GfVec3d(0.0, 0.0, 0.0);// + center;
    //DEBUG
    // *worldPositionFabric = pxr::GfVec3d(300.0, 300.0, 0.0);

    auto worldOrientationFabric = stageReaderWriter.getAttributeWr<pxr::GfQuatf>(fabricPath, FabricTokens::_worldOrientation);
    //*worldOrientationFabric = pxr::GfQuatf(1.f, 0, 0, 0);
    *worldOrientationFabric = pxr::GfQuatf(0.f, 0, 0, 0);

    // auto worldScaleFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::_worldScale);
    // *worldScaleFabric = pxr::GfVec3f(1.f, 1.f, 1.f);

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::primvars_displayColor, 1);
    auto displayColors = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::primvars_displayColor);
    displayColors[0] = pxr::GfVec3f(0.8f, 0.8f, 0.8f);

    //create a custom attribute for testing
    stageReaderWriter.createAttribute(fabricPath, getCudaTestAttributeFabricToken(), cudaTestAttributeFabricType);

    auto testAttribute = stageReaderWriter.getAttributeWr<double>(fabricPath, getCudaTestAttributeFabricToken());
    *testAttribute = 123.45;
}

void createMultiquadMeshViaFabric2(size_t size) {
    const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    const auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(cesium::omniverse::Context::instance().getStageId())};
    const auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    const std::string primPathStub{"/multiquadMesh"};
    const auto fabricPath = omni::fabric::Path((primPathStub + std::to_string(0)).c_str());
    stageReaderWriter.createPrim(fabricPath);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::primvars_displayColor, FabricTokens::primvars_displayColor);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    // attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.createAttributes(fabricPath);

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::points, static_cast<size_t>(size * size * 4));
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexCounts, size * size * 2);
    auto pointsFabric =
        stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    auto faceVertexCountsFabric =
        stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexCounts);
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexIndices, size * size * 6);
    auto faceVertexIndicesFabric =
        stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexIndices);


    const float unit = 1.f;
    auto floatSize = static_cast<float>(size);
    float extentVal = floatSize / 2 * unit * 2 + (floatSize / 2 - 1) * unit * 2;

    size_t vertIndex = 0;
    size_t vertexCountsIndex = 0;
    size_t faceVertexIndex = 0;
    size_t quadCounter = 0;
    for (size_t rowNum = 0; rowNum < size; rowNum++) {
        for (size_t colNum = 0; colNum < size; colNum++) {
            float xPos = -extentVal + static_cast<float>(colNum) * unit * 4;
            float yPos = -extentVal + static_cast<float>(rowNum) * unit * 4;
            //verts
            pxr::GfVec3f quadShift{xPos, yPos, 0};
            pointsFabric[vertIndex++] = pxr::GfVec3f{-unit, -unit, 0} + quadShift;
            pointsFabric[vertIndex++] = pxr::GfVec3f{-unit, unit, 0} + quadShift;
            pointsFabric[vertIndex++] = pxr::GfVec3f{unit, unit, 0} + quadShift;
            pointsFabric[vertIndex++] = pxr::GfVec3f{unit, -unit, 0} + quadShift;

            //vert counts
            faceVertexCountsFabric[vertexCountsIndex++] = 3;
            faceVertexCountsFabric[vertexCountsIndex++] = 3;

            //vert indices
            faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(quadCounter * 4);
            faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(quadCounter * 4);
            faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(quadCounter * 4);
            faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(quadCounter * 4);
            faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(quadCounter * 4);
            faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(quadCounter * 4);
            quadCounter++;
        }
    }

    auto outerExtent = extentVal + unit;
    auto extent = pxr::GfRange3d(pxr::GfVec3d(-outerExtent, -outerExtent, 0), pxr::GfVec3d(outerExtent, outerExtent, 0));
    auto extentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::extent);
    *extentFabric = extent;

    auto worldExtentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::_worldExtent);
    *worldExtentFabric = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));

    auto worldVisibilityFabric = stageReaderWriter.getAttributeWr<bool>(fabricPath, FabricTokens::_worldVisibility);
    *worldVisibilityFabric = true;

    // auto center = pxr::GfVec3d{
    //     glm::linearRand(-maxCenterRandomization, maxCenterRandomization),
    //     glm::linearRand(-maxCenterRandomization, maxCenterRandomization),
    //     glm::linearRand(-maxCenterRandomization, maxCenterRandomization)
    // };

    auto worldPositionFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3d>(fabricPath, FabricTokens::_worldPosition);
    *worldPositionFabric = pxr::GfVec3d(0.0, 0.0, 0.0);// + center;
    //DEBUG
    // *worldPositionFabric = pxr::GfVec3d(300.0, 300.0, 0.0);

    auto worldOrientationFabric = stageReaderWriter.getAttributeWr<pxr::GfQuatf>(fabricPath, FabricTokens::_worldOrientation);
    //*worldOrientationFabric = pxr::GfQuatf(1.f, 0, 0, 0);
    *worldOrientationFabric = pxr::GfQuatf(0.f, 0, 0, 0);

    // auto worldScaleFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::_worldScale);
    // *worldScaleFabric = pxr::GfVec3f(1.f, 1.f, 1.f);

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::primvars_displayColor, 1);
    auto displayColors = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::primvars_displayColor);
    displayColors[0] = pxr::GfVec3f(0.8f, 0.8f, 0.8f);

    //create a custom attribute for testing
    stageReaderWriter.createAttribute(fabricPath, getCudaTestAttributeFabricToken(), cudaTestAttributeFabricType);
    auto testAttribute = stageReaderWriter.getAttributeWr<double>(fabricPath, getCudaTestAttributeFabricToken());
    *testAttribute = 123.45;

    //record number of quads (for testing purposes)
    stageReaderWriter.createAttribute(fabricPath, getNumQuadsAttributeFabricToken(), numQuadsFabricType);
    auto numQuadsAttribute = stageReaderWriter.getAttributeWr<int>(fabricPath, getNumQuadsAttributeFabricToken());
    *numQuadsAttribute = static_cast<int>(size * size);

    //create one quaternion for each quad
    stageReaderWriter.createAttribute(fabricPath, getQuadOrientationAttributeFabricToken(), quadOrientationType);
    stageReaderWriter.setArrayAttributeSize(fabricPath, getQuadOrientationAttributeFabricToken(), size * 4);
}


void modifyAllPrimsWithCustomAttrViaCuda() {

    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    omni::fabric::AttrNameAndType cudaTestAttrTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    omni::fabric::PrimBucketList buckets = stageReaderWriter.findPrims({cudaTestAttrTag});

    if (buckets.bucketCount() == 0 ) {
        std::cout << "No prims found, returning" << std::endl;
    }

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
    }

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
    }

    nvrtcProgram prog;
    //nvrtcCreateProgram(&prog, kernelCode, "changeValue", 0, nullptr, nullptr);
    nvrtcCreateProgram(&prog, modifyVec3fArrayKernelCode, "changeValue", 0, nullptr, nullptr);

    // Compile the program
    nvrtcResult res = nvrtcCompileProgram(prog, 0, nullptr);
    if (res != NVRTC_SUCCESS) {
        std::cout << "Error compiling NVRTC program:" << std::endl;

        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(prog, log);
        std::cout << "   Compilation log: \n" << log << std::endl;

        return;
    }

    // Get the PTX (assembly code for the GPU) from the compilation
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    // Load the generated PTX and get a handle to the kernel.
    CUmodule module;
    CUfunction function;
    cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    cuModuleGetFunction(&function, module, "changeValue");

    auto bucketCount = buckets.bucketCount();
    printf("Num buckets: %llu\n", bucketCount);

    //iterate over buckets but pass the vector for the whole bucket to the GPU.
    // int primCount = 0;
    //test on CPU
    // for (size_t bucket = 0; bucket != buckets.bucketCount(); bucket++)
    // {
    //     auto values = stageReaderWriter.getAttributeArray<pxr::GfVec3f*>(buckets, bucket, FabricTokens::primvars_displayColor);
    //     auto elementCount = values.size();
    //     for (size_t i = 0; i < elementCount; i++) {
    //         printf("displayColors of element %llu are %f, %f, %f\n", i, values[i][0][0], values[i][0][1], values[i][0][2]);
    //         values[i][0][0] = 0;
    //         values[i][0][1] = 1.f;
    //     }
    // }

    struct Vec3f
    {
        float x;
        float y;
        float z;
    };

    for (size_t bucket = 0; bucket != buckets.bucketCount(); bucket++)
    {
        //gsl::span<double> values = stageReaderWriter.getAttributeArrayGpu<double>(buckets, bucket, getCudaTestAttributeFabricToken());
        auto values = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f*>(buckets, bucket, FabricTokens::primvars_displayColor);

        auto ptr = reinterpret_cast<Vec3f**>(values.data());
        size_t elemCount = values.size();
        void *args[] = { &ptr, &elemCount }; //NOLINT
        int blockSize = 32 * 4;
        int numBlocks = (static_cast<int>(elemCount) + blockSize - 1) / blockSize;
        // alternatively, CUDA can calculate these for you
        // int blockSize, minGridSize;
        // cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, nullptr, 0, 0);
        auto err = cuLaunchKernel(function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        if (err) {
            std::cout << "error" << std::endl;
        }
        // primCount += static_cast<int>(elemCount);
    }

    // std::cout << "modified " << primCount << " quads" << std::endl;

    result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }

    delete[] ptx;
}

void repositionAllPrimsWithCustomAttrViaCuda(double spacing) {

    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    omni::fabric::AttrNameAndType cudaTestAttrTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    omni::fabric::PrimBucketList buckets = stageReaderWriter.findPrims({cudaTestAttrTag});

    if (buckets.bucketCount() == 0 ) {
        std::cout << "No prims found, returning" << std::endl;
    }

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
    }

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
    }

    nvrtcProgram prog;
    //nvrtcCreateProgram(&prog, kernelCode, "changeValue", 0, nullptr, nullptr);
    nvrtcCreateProgram(&prog, modifyVec3dKernelCode, "changeValue", 0, nullptr, nullptr);

    // Compile the program
    nvrtcResult res = nvrtcCompileProgram(prog, 0, nullptr);
    if (res != NVRTC_SUCCESS) {
        std::cout << "Error compiling NVRTC program:" << std::endl;

        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(prog, log);
        std::cout << "   Compilation log: \n" << log << std::endl;

        return;
    }

    // Get the PTX (assembly code for the GPU) from the compilation
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    // Load the generated PTX and get a handle to the kernel.
    CUmodule module;
    CUfunction function;
    cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    cuModuleGetFunction(&function, module, "setVec3d");

    auto bucketCount = buckets.bucketCount();
    printf("Num buckets: %llu\n", bucketCount);

    //iterate over buckets but pass the vector for the whole bucket to the GPU.
    //int primCount = 0;
    //test on CPU
    // for (size_t bucket = 0; bucket != buckets.bucketCount(); bucket++)
    // {
    //     auto values = stageReaderWriter.getAttributeArray<pxr::GfVec3f*>(buckets, bucket, FabricTokens::primvars_displayColor);
    //     auto elementCount = values.size();
    //     for (size_t i = 0; i < elementCount; i++) {
    //         printf("displayColors of element %llu are %f, %f, %f\n", i, values[i][0][0], values[i][0][1], values[i][0][2]);
    //         values[i][0][0] = 0;
    //         values[i][0][1] = 1.f;
    //     }
    // }

    struct Vec3d
    {
        double x;
        double y;
        double z;
    };

    //DEVELOP
    std::cout << spacing << std::endl;

    for (size_t bucket = 0; bucket != buckets.bucketCount(); bucket++)
    {
        //gsl::span<double> values = stageReaderWriter.getAttributeArrayGpu<double>(buckets, bucket, getCudaTestAttributeFabricToken());
        auto values = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3d>(buckets, bucket, FabricTokens::_worldPosition);

        auto ptr = reinterpret_cast<Vec3d*>(values.data());
        size_t elemCount = values.size();
        void *args[] = { &ptr, &elemCount }; //NOLINT
        int blockSize = 32 * 4;
        int numBlocks = (static_cast<int>(elemCount) + blockSize - 1) / blockSize;
        // alternatively, CUDA can calculate these for you
        // int blockSize, minGridSize;
        // cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, nullptr, 0, 0);
        auto err = cuLaunchKernel(function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        if (err) {
            std::cout << "error" << std::endl;
        }
        // primCount += static_cast<int>(elemCount);
    }

    // std::cout << "modified " << primCount << " quads" << std::endl;

    result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }

    delete[] ptx;
}

void randomizePrimWorldPositionsWithCustomAttrViaCuda() {

    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    omni::fabric::AttrNameAndType cudaTestAttrTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    omni::fabric::PrimBucketList buckets = stageReaderWriter.findPrims({cudaTestAttrTag});

    if (buckets.bucketCount() == 0 ) {
        std::cout << "No prims found, returning" << std::endl;
    }

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
    }

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
    }

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog,                 // prog
                       randomizeVec3dKernelCode,  // buffer
                       nullptr,               // name
                       0,                     // numHeaders
                       nullptr,               // headers
                       nullptr);              // includeNames

    // Compile the program
    nvrtcResult res = nvrtcCompileProgram(prog, 0, nullptr);
     if (res != NVRTC_SUCCESS) {
        std::cout << "Error compiling NVRTC program:" << std::endl;

        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(prog, log);
        std::cout << "   Compilation log: \n" << log << std::endl;

        return;
    }

    // Get the PTX (assembly code for the GPU) from the compilation
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    // Load the generated PTX and get a handle to the kernel.
    CUmodule module;
    CUfunction function;
    cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    cuModuleGetFunction(&function, module, "randomizeVec3d");

    auto bucketCount = buckets.bucketCount();
    printf("Num buckets: %llu\n", bucketCount);

    //iterate over buckets but pass the vector for the whole bucket to the GPU.
    int primCount = 0;
    //test on CPU
    // for (size_t bucket = 0; bucket != buckets.bucketCount(); bucket++)
    // {
    //     auto values = stageReaderWriter.getAttributeArray<pxr::GfVec3f*>(buckets, bucket, FabricTokens::primvars_displayColor);
    //     auto elementCount = values.size();
    //     for (size_t i = 0; i < elementCount; i++) {
    //         printf("displayColors of element %llu are %f, %f, %f\n", i, values[i][0][0], values[i][0][1], values[i][0][2]);
    //         values[i][0][0] = 0;
    //         values[i][0][1] = 1.f;
    //     }
    // }

    struct Vec3d
    {
        double x;
        double y;
        double z;
    };

    for (size_t bucket = 0; bucket != buckets.bucketCount(); bucket++)
    {
        //gsl::span<double> values = stageReaderWriter.getAttributeArrayGpu<double>(buckets, bucket, getCudaTestAttributeFabricToken());
        auto values = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3d>(buckets, bucket, FabricTokens::_worldPosition);

        auto ptr = reinterpret_cast<Vec3d*>(values.data());
        size_t elemCount = values.size();
        auto seed = rand();
        void *args[] = { &ptr, &elemCount, &seed }; //NOLINT
        int blockSize = 32 * 4;
        int numBlocks = (static_cast<int>(elemCount) + blockSize - 1) / blockSize;
        // alternatively, CUDA can calculate these for you
        // int blockSize, minGridSize;
        // cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, nullptr, 0, 0);
        auto err = cuLaunchKernel(function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        if (err) {
            std::cout << "error" << std::endl;
        }
        primCount += static_cast<int>(elemCount);
    }

    std::cout << "modified " << primCount << " quads" << std::endl;

    result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }

    delete[] ptx;
}


void createFabricQuadsModifyViaCuda(int numQuads) {
    createQuadsViaFabric(numQuads);
    modifyAllPrimsWithCustomAttrViaCuda();
}

void modify1000UsdQuadsViaCuda() {
    const size_t quadCount = 100;

    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();
    for (size_t i = 0; i != quadCount; i++)
    {
        createQuadMeshViaUsd(("/quad_" + std::to_string(i)).c_str(), 300.f);

        //NOTE: this function has a runtime error. You can define primitives (Cube, Sphere) and an Xform
        //However, a Mesh will lead to runtime errors
        //test code below to see if something like an xformable can be used instead
        // pxr::SdfPath path("/quad_" + std::to_string(i));
        //pxr::UsdPrim prim = usdStagePtr->DefinePrim(path, pxr::TfToken("Mesh"));
        // auto prim = pxr::UsdGeomMesh::Define(usdStagePtr, path);

        // float centerRandomization{200.f};
        // pxr::GfVec3f center{
        // glm::linearRand(-centerRandomization, centerRandomization),
        // glm::linearRand(-centerRandomization, centerRandomization),
        // glm::linearRand(-centerRandomization, centerRandomization)
        // };

        // auto pointsAttr = prim.GetPointsAttr();
        // auto points = pxr::VtArray<pxr::GfVec3f>{4};
        // float quadSize{50.f};
        // points[0] = pxr::GfVec3f{-quadSize, -quadSize, 0} + center;
        // points[1] = pxr::GfVec3f{-quadSize, quadSize, 0} + center;
        // points[2] = pxr::GfVec3f{quadSize, quadSize, 0} + center;
        // points[3] = pxr::GfVec3f{quadSize, -quadSize, 0} + center;
        // pointsAttr.Set(points);

        // auto faceVertexIndicesAttr = prim.GetFaceVertexIndicesAttr();
        // pxr::VtArray<int> faceVertIndices{0, 1, 2, 0, 2, 3};
        // faceVertexIndicesAttr.Set(faceVertIndices);

        // auto faceVertexCountsAttr = prim.GetFaceVertexCountsAttr();
        // faceVertexIndicesAttr.Set(pxr::VtArray<int>{3, 3});

        // auto extentAttr = prim.GetExtentAttr();
        // extentAttr.Set(pxr::VtArray<pxr::GfVec3f>{
        //     pxr::GfVec3f(-quadSize, -quadSize, 0),
        //     pxr::GfVec3f(-quadSize, -quadSize, 0)
        //     });
    }

    //Alter cudaTest attribute on all quads with Cuda
    long id = Context::instance().getStageId();
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    for (size_t i = 0; i != quadCount; i++)
    {
        omni::fabric::Path path(("/quad_" + std::to_string(i)).c_str());
        iStageReaderWriter->prefetchPrim(usdStageId, path);
    }

    // //select and bucket the Cube prims
    // omni::fabric::AttrNameAndType meshTag(
    //     omni::fabric::Type(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName),
    //     omni::fabric::Token("meshTag"));
    // const auto stageReaderWriterId = iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(id)});
    // auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    // omni::fabric::PrimBucketList quadBuckets = stageReaderWriter.findPrims({ meshTag });

    // auto bucketCount = quadBuckets.bucketCount();
    // if (bucketCount == 0) {
    //     std::cout << "Found no prims to modify" << std::endl;
    // }
    // printf("Num buckets: %llu", bucketCount);

    // auto isCudaCompatible = checkCudaCompatibility();
    // if (!isCudaCompatible) {
    //     std::cout << "error: CUDA drives and toolkit versions are not compatible." << std::endl;
    // }

    // CUresult result = cuInit(0);
    // if (result != CUDA_SUCCESS) {
    //     std::cout << "error: CUDA did not init." << std::endl;
    // }

    // CUdevice device;
    // result = cuDeviceGet(&device, 0);
    // if (result != CUDA_SUCCESS) {
    //     std::cout << "error: CUDA did not get a device." << std::endl;
    // }

    // int major, minor;
    // result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    // if (result != CUDA_SUCCESS) {
    //     std::cout << "error: could not get CUDA major version." << std::endl;
    // }
    // result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    // if (result != CUDA_SUCCESS) {
    //     std::cout << "error: could not get CUDA minor version." << std::endl;
    // }

    // std::cout << "Compute capability: " << major << "." << minor << std::endl;

    // CUcontext context;
    // result = cuCtxCreate(&context, 0, device);
    // if (result != CUDA_SUCCESS) {
    //     std::cout << "error: could not create CUDA context." << std::endl;
    // }

    // //CUDA via CUDA_JIT and string
    // const char *kernelCode = R"(
    // extern "C" __global__
    // void changeValue(double* values, size_t count)
    // {
    //     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    //     if (count <= i) return;

    //     float oldVal = values[i];
    //     values[i] = 543.21;
    //     printf("Changed value of index %llu from %lf to %lf\n", i, oldVal, values[i]);
    // }
    // )";

    // nvrtcProgram prog;
    // nvrtcCreateProgram(&prog, kernelCode, "changeValue", 0, nullptr, nullptr);

    // // Compile the program
    // nvrtcResult res = nvrtcCompileProgram(prog, 0, nullptr);
    // if (res != NVRTC_SUCCESS) {
    //     std::cout << "error compiling NVRTC program" << std::endl;
    //     return;
    // }

    // // Get the PTX (assembly code for the GPU) from the compilation
    // size_t ptxSize;
    // nvrtcGetPTXSize(prog, &ptxSize);
    // char* ptx = new char[ptxSize];
    // nvrtcGetPTX(prog, ptx);

    // // Load the generated PTX and get a handle to the kernel.
    // CUmodule module;
    // CUfunction function;
    // cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    // cuModuleGetFunction(&function, module, "changeValue");

    // //iterate over buckets but pass the vector for the whole bucket to the GPU.
    // int alterationCount = 0;
    // for (size_t bucket = 0; bucket != quadBuckets.bucketCount(); bucket++)
    // {
    //     gsl::span<double> sizesD = stageReaderWriter.getAttributeArrayGpu<double>(quadBuckets, bucket, omni::fabric::Token("cudaTest"));

    //     double* ptr = sizesD.data();
    //     size_t elemCount = sizesD.size();
    //     void *args[] = { &ptr, &elemCount }; //NOLINT
    //     int blockSize, minGridSize;
    //     cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, nullptr, 0, 0);
    //     //CUresult err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, 0);
    //     auto err = cuLaunchKernel(function, minGridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
    //     // REQUIRE(!err);
    //     if (err) {
    //         std::cout << "error" << std::endl;
    //     }

    //     alterationCount += static_cast<int>(elemCount);
    // }
    // std::cout << "Altered " << alterationCount << " prims" << std::endl;

    // result = cuCtxDestroy(context);
    // if (result != CUDA_SUCCESS) {
    //     std::cout << "error: could not destroy CUDA context." << std::endl;
    // }
}

void createQuadMeshViaUsd(const char* pathString, float maxCenterRandomization) {
    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();
    pxr::SdfPath path(pathString);

    auto mesh = pxr::UsdGeomMesh::Define(usdStagePtr, path);

    //set points
    auto pointsAttr = mesh.GetPointsAttr();
    //if you want to know the type name
    // {
    //     pxr::SdfValueTypeName typeName = pointsAttr.GetTypeName();
    //     auto attrToken = typeName.GetAsToken();
    //     std::cout << "Type name: " << attrToken.GetString() << std::endl;
    // }
    pxr::VtArray<pxr::GfVec3f> points{4};
    float quadScalar = 50.f;
    pxr::GfVec3f center{
        glm::linearRand(-maxCenterRandomization, maxCenterRandomization),
        glm::linearRand(-maxCenterRandomization, maxCenterRandomization),
        0};
    points[0] = pxr::GfVec3f(-quadScalar, -quadScalar, 0) + center;
    points[1] = pxr::GfVec3f(-quadScalar, quadScalar, 0) + center;
    points[2] = pxr::GfVec3f(quadScalar, quadScalar, 0) + center;
    points[3] = pxr::GfVec3f(quadScalar, -quadScalar, 0) + center;
    pointsAttr.Set(points);

    auto faceVertexCountsAttr = mesh.GetFaceVertexCountsAttr();
    pxr::VtArray<int> faceVertexCounts{3, 3};
    faceVertexCountsAttr.Set(faceVertexCounts);

    auto faceVertexIndicesAttr = mesh.GetFaceVertexIndicesAttr();
    pxr::VtArray<int> faceVertexIndices{0, 1, 2, 0, 2, 3};
    faceVertexIndicesAttr.Set(faceVertexIndices);

    auto extentAttr = mesh.GetExtentAttr();
    pxr::VtArray<pxr::GfVec3f> extent{2};
    extent[0] = pxr::GfVec3f{-quadScalar, -quadScalar, 0};
    extent[1] = pxr::GfVec3f{quadScalar, quadScalar, 0};
    extentAttr.Set(extent);

    // auto worldPositionFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3d>(fabricPath, FabricTokens::_worldPosition);
    // *worldPositionFabric = pxr::GfVec3d(0, 0, 0);

    // auto worldOrientationFabric = stageReaderWriter.getAttributeWr<pxr::GfQuatf>(fabricPath, FabricTokens::_worldOrientation);
    // *worldOrientationFabric = pxr::GfQuatf(1.f, 0, 0, 0);

    // auto worldScaleFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::_worldScale);
    // *worldScaleFabric = pxr::GfVec3f(1.f, 1.f, 1.f);

    auto customAttrUsdToken = pxr::TfToken("cudaTest");
    //can only set a custom attr on the prim, not on an object defined by the USD schema
    auto prim = mesh.GetPrim();
    prim.CreateAttribute(customAttrUsdToken, pxr::SdfValueTypeNames->Double).Set(12.3);

    // mesh.CreateDisplayColorPrimvar(UsdTokens::constant);
    // pxr::VtArray<pxr::GfVec3f> displayColor(1);
    // displayColor[0] = pxr::GfVec3f(1.0f, 0.0f, 0.0f);
    // mesh.GetDisplayColorPrimvar().Set(displayColor);
}

void alterUsdPrimTranslationWithFabric() {
    auto usdStagePtr = Context::instance().getStage();

    const size_t cubeCount = 10;
    auto customAttrUsdToken = pxr::TfToken("cudaTest");
    for (size_t i = 0; i != cubeCount; i++)
    {
        pxr::SdfPath path("/cube_" + std::to_string(i));
        auto cube = pxr::UsdGeomCube::Define(usdStagePtr, path);
        cube.GetPrim().CreateAttribute(customAttrUsdToken, pxr::SdfValueTypeNames->Double).Set(12.3);
        auto xform = pxr::UsdGeomXformCommonAPI::Get(usdStagePtr, path);
        xform.SetTranslate(pxr::GfVec3d(0, 0, 0));
    }

    //call prefetchPrim to get the data into Fabric.
    long id = Context::instance().getStageId();
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    for (size_t i = 0; i != cubeCount; i++)
    {
        omni::fabric::Path path(("/cube_" + std::to_string(i)).c_str());
        iStageReaderWriter->prefetchPrim(usdStageId, path);
    }

    //get all USD prims with the custom attr
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    // edit translations
    auto token = omni::fabric::Token("_worldPosition");
    auto numBuckets = bucketList.bucketCount();
    for (size_t bucketNum = 0; bucketNum < numBuckets; bucketNum++) {
        auto values = stageReaderWriter.getAttributeArray<pxr::GfVec3d>(bucketList, bucketNum, token);
        auto numElements = values.size();
        for (unsigned long long i = 0; i < numElements; i++) {
            values[i].Set(static_cast<double>(1), static_cast<double>(2), static_cast<double>(3));
        }
    }

    // edit cudaTest attr
    for (size_t bucketNum = 0; bucketNum < numBuckets; bucketNum++) {
        gsl::span<double> values = stageReaderWriter.getAttributeArray<double>(bucketList, bucketNum, getCudaTestAttributeFabricToken());
        const auto numElements = values.size();
        for (unsigned long long i = 0; i < numElements; i++) {
            values[i] = 543.21;
        }
    }
}

void repositionAllPrimsWithCustomAttrViaFabric(double spacing) {

    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    // edit translations
    auto token = omni::fabric::Token("_worldPosition");
    auto numBuckets = bucketList.bucketCount();
    for (size_t bucketNum = 0; bucketNum < numBuckets; bucketNum++) {
        auto values = stageReaderWriter.getAttributeArray<pxr::GfVec3d>(bucketList, bucketNum, token);
        auto numElements = values.size();
        for (unsigned long long i = 0; i < numElements; i++) {
            auto coord = static_cast<double>(i) * spacing;
            values[i].Set(coord, coord, coord);
        }
    }

    // edit cudaTest attr
    for (size_t bucketNum = 0; bucketNum < numBuckets; bucketNum++) {
        gsl::span<double> values = stageReaderWriter.getAttributeArray<double>(bucketList, bucketNum, getCudaTestAttributeFabricToken());
        const auto numElements = values.size();
        for (unsigned long long i = 0; i < numElements; i++) {
            values[i] = 543.21;
        }
    }
}


void alterUsdPrimTranslationWithUsd() {
    auto usdStagePtr = Context::instance().getStage();

    const size_t cubeCount = 10;
    for (size_t i = 0; i != cubeCount; i++)
    {
        pxr::SdfPath path("/cube_" + std::to_string(i));

        //parenting to an Xform could work
        // pxr::UsdGeomXform xform = pxr::UsdGeomXform::Define(usdStagePtr, path);
        // pxr::UsdPrim prim = usdStagePtr->DefinePrim(xform.GetPath().AppendChild(pxr::TfToken("CubePrim")), pxr::TfToken("Cube"));

        pxr::UsdPrim prim = usdStagePtr->DefinePrim(path, pxr::TfToken("Cube"));
        if (prim.IsA<pxr::UsdGeomXformable>()) {
            pxr::UsdGeomXformable xformable(prim);
            // Add an xformOp to the Xformable prim to define the transform
            xformable.AddTranslateOp().Set(pxr::GfVec3d(3. * static_cast<double>(i), 0, 0));
        }
    }

    //expand coords along diagonal
    for (size_t i = 0; i != cubeCount; i++)
    {
        // This crashes at runtime
        // auto prim = usdStagePtr->GetPrimAtPath(path);
        // pxr::UsdGeomXformable xformable(prim);
        // auto coord = static_cast<double>(i);
        // xformable.AddTranslateOp().Set(pxr::GfVec3d(coord, coord, coord));

        pxr::SdfPath path("/cube_" + std::to_string(i));
        auto xform = pxr::UsdGeomXformCommonAPI::Get(usdStagePtr, path);
        auto coord = static_cast<double>(i);
        xform.SetTranslate(pxr::GfVec3d(coord, coord, coord));
    }
}

void setDisplayColor() {
    auto usdStagePtr = Context::instance().getStage();
    pxr::SdfPath cubePath("/testPrim");
    pxr::UsdPrim cubePrim = usdStagePtr->GetPrimAtPath(cubePath);

    if (!cubePrim.IsValid()) {
        std::cout << "must be a /testPrim prim in the stage" << std::endl;
        return;
    }

    pxr::UsdGeomMesh cubeMesh(cubePrim);

    // // Create the displayColor attribute if it doesn't already exist
    // if (!cubeMesh.GetDisplayColorPrimvar())
    // {
    //     cubeMesh.CreateDisplayColorPrimvar();
    // }

    // cubeMesh.CreateDisplayColorPrimvar().Set(pxr::VtArray<pxr::GfVec3f>{pxr::GfVec3f(0.0f, 1.0, 0.0)});


    // Create the displayColor attribute if it doesn't already exist
    if (!cubeMesh.GetDisplayColorAttr())
    {
        cubeMesh.CreateDisplayColorAttr();
    }

    // Set the display color to red
    cubeMesh.GetDisplayColorAttr().Set(pxr::VtArray<pxr::GfVec3f>{pxr::GfVec3f(0.0, 1.0, 0.0)});

}

void createQuadMeshWithDisplayColor() {

}

void rotateAllPrimsWithCustomAttrViaFabric() {
    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    // edit rotations
    auto token = omni::fabric::Token("_worldOrientation");
    auto numBuckets = bucketList.bucketCount();
    for (size_t bucketNum = 0; bucketNum < numBuckets; bucketNum++) {
        auto values = stageReaderWriter.getAttributeArray<pxr::GfQuatf>(bucketList, bucketNum, token);
        auto numElements = values.size();
        for (unsigned long long i = 0; i < numElements; i++) {
            pxr::GfQuatf quat = values[i];
            auto glmQuat = convertToGlm(quat);
            auto angle = static_cast<float>(glm::radians(1.f));
            glm::fvec3 axis(1, 0, 0); //
            glm::fquat rotation = glm::angleAxis(angle, axis);
            glmQuat = rotation * glmQuat;
            auto rotatedQuat = convertToGf(glmQuat);
            values[i] = rotatedQuat;
        }
    }
}

// glm::dquat convertToGlm(const pxr::GfQuatd& quat) {
//     return {
//         quat.GetReal(),
//         quat.GetImaginary()[0],
//         quat.GetImaginary()[1],
//         quat.GetImaginary()[2]};
// }

glm::fquat convertToGlm(const pxr::GfQuatf& quat) {
    return {
        quat.GetReal(),
        quat.GetImaginary()[0],
        quat.GetImaginary()[1],
        quat.GetImaginary()[2]};
}

pxr::GfQuatd convertToGf(const glm::dquat& quat) {
    return {quat.w, pxr::GfVec3d(quat.x, quat.y, quat.z)};
}

pxr::GfQuatf convertToGf(const glm::fquat& quat) {
    return {quat.w, pxr::GfVec3f(quat.x, quat.y, quat.z)};
}

void billboardAllPrimsWithCustomAttrViaFabric() {
    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    // edit rotations
    auto token = omni::fabric::Token("_worldOrientation");
    auto worldPositionsTokens = omni::fabric::Token("_worldPosition");
    auto numBuckets = bucketList.bucketCount();

    glm::fvec3 lookatPosition{0.0, 0.0, 0.0};

    for (size_t bucketNum = 0; bucketNum < numBuckets; bucketNum++) {
        auto orientations = stageReaderWriter.getAttributeArray<pxr::GfQuatf>(bucketList, bucketNum, token);
        auto worldPositions = stageReaderWriter.getAttributeArray<pxr::GfVec3d>(bucketList, bucketNum, worldPositionsTokens);
        auto numElements = orientations.size();
        for (unsigned long long i = 0; i < numElements; i++) {
            // pxr::GfQuatf quat = values[i];
            // auto glmQuat = convertToGlm(quat);
            auto worldPositionGfVec3f = pxr::GfVec3f(worldPositions[i]);
            auto worldPositionGlm = usdToGlmVector(worldPositionGfVec3f);
            glm::fvec3 direction = lookatPosition - worldPositionGlm;
            direction = glm::normalize(direction);
            glm::fquat newQuat = glm::quatLookAt(direction, glm::fvec3{0, 1.f, 0});
            auto rotatedQuat = convertToGf(newQuat);
            orientations[i] = rotatedQuat;
        }
    }
}

void billboardAllPrimsWithCustomAttrViaCuda() {
    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    // edit rotations
    auto token = omni::fabric::Token("_worldOrientation");
    auto worldPositionsTokens = omni::fabric::Token("_worldPosition");

    // auto numBuckets = bucketList.bucketCount();
    // printf("Num buckets: %llu\n", numBuckets);

    if (bucketList.bucketCount() == 0 ) {
        std::cout << "No prims found, returning" << std::endl;
    }

    cudaRunner.init(lookAtKernelCode, "lookAt");

    //iterate over buckets but pass the vector for the whole bucket to the GPU.
    // int primCount = 0;

    CUresult err;
    CUdeviceptr lookatPositionDevice;

    err = cuMemAlloc(&lookatPositionDevice, sizeof(glm::dvec3));
    if (err != CUDA_SUCCESS) {
        const char *errName;
        const char *errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemAlloc failed: %s: %s\n", errName, errStr);
        return;
    }

    err = cuMemcpyHtoD(lookatPositionDevice, &lookatPositionHost, sizeof(glm::dvec3));
    if (err != CUDA_SUCCESS) {
        const char *errName;
        const char *errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemcpyHtoD failed: %s: %s\n", errName, errStr);
        return;
    }



    for (size_t bucket = 0; bucket != bucketList.bucketCount(); bucket++)
    {
        auto worldPositions = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3d>(bucketList, bucket, FabricTokens::_worldPosition);
        auto worldOrientations = stageReaderWriter.getAttributeArrayGpu<pxr::GfQuatf>(bucketList, bucket, FabricTokens::_worldOrientation);
        auto worldPositionsPtr = reinterpret_cast<glm::dvec3*>(worldPositions.data());
        auto worldOrientationsPtr = reinterpret_cast<glm::fquat*>(worldOrientations.data());
        size_t elemCount = worldPositions.size();
        void *args[] = { &worldOrientationsPtr, &worldPositionsPtr, &lookatPositionDevice, &elemCount}; //NOLINT

        cudaRunner.runKernel(args, elemCount);

        // primCount += static_cast<int>(elemCount);
    }

    // std::cout << "modified " << primCount << " quads" << std::endl;

    err = cuMemFree(lookatPositionDevice);
    if (err != CUDA_SUCCESS) {
        const char *errName;
        const char *errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemFree failed: %s: %s\n", errName, errStr);
        return;
    }

    // lookatPositionHost.x += 10.0;

}

void billboardMultiquadWithCustomAttrViaFabric() {
    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    auto numBuckets = bucketList.bucketCount();
    printf("Num buckets: %llu\n", numBuckets);

    if (bucketList.bucketCount() == 0 ) {
        std::cout << "No prims found, returning" << std::endl;
        throw std::runtime_error("Bucketlist is empty");
    }

    auto lookatPosition = make_double3(0, 0, 0);

    for (size_t bucketNum = 0; bucketNum != bucketList.bucketCount(); bucketNum++)
    {
        auto positions = stageReaderWriter.getAttributeArray<pxr::GfVec3f*>(bucketList, bucketNum, FabricTokens::points);
        auto numQuadsSpan = stageReaderWriter.getAttributeArray<int>(bucketList, bucketNum, getNumQuadsAttributeFabricToken());
        int numQuads = numQuadsSpan[0];
        auto sizeofQuad = sizeof(quad);
        std::cout << "sizeofQuad: " << sizeofQuad << std::endl;
        auto sizeOfGfVec3f = sizeof(pxr::GfVec3f);
        std::cout << "sizeofGfVec3f " << sizeOfGfVec3f << std::endl;
        std::cout << "sizeofFloat3 " << sizeof(float3) << std::endl;

        auto quadsPtr = reinterpret_cast<quad*>(positions[0]->data());
        // std::cout << "(host) numQuads: " << numQuads << std::endl;

        for (int quadNum = 0; quadNum < numQuads; quadNum++) {
            printf("quad %d lowerLeft: %f, %f, %f\n", quadNum,
                quadsPtr[quadNum].lowerLeft.x,
                quadsPtr[quadNum].lowerLeft.y,
                quadsPtr[quadNum].lowerLeft.z);
        }

        int elemCount = numQuads;
        if (elemCount == 0) {
            throw std::runtime_error("Fabric did not retrieve any elements");
        }
        std::cout << elemCount << std::endl;

        lookatMultiquad(quadsPtr, &lookatPosition, numQuads);
    }
}

void billboardMultiquadWithCustomAttrViaCuda() {
    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    auto numBuckets = bucketList.bucketCount();
    printf("Num buckets: %llu\n", numBuckets);

    if (bucketList.bucketCount() == 0 ) {
        std::cout << "No prims found, returning" << std::endl;
        throw std::runtime_error("Bucketlist is empty");
    }

    cudaRunner.init(lookAtMultiquadKernelCode, "lookAtMultiquadKernel");

    //iterate over buckets but pass the vector for the whole bucket to the GPU.
    // int primCount = 0;

    CUresult err;
    CUdeviceptr lookatPositionDevice;

    err = cuMemAlloc(&lookatPositionDevice, sizeof(glm::dvec3));
    if (err != CUDA_SUCCESS) {
        const char *errName;
        const char *errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemAlloc failed: %s: %s\n", errName, errStr);
        return;
    }

    err = cuMemcpyHtoD(lookatPositionDevice, &lookatPositionHost, sizeof(glm::dvec3));
    if (err != CUDA_SUCCESS) {
        const char *errName;
        const char *errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemcpyHtoD failed: %s: %s\n", errName, errStr);
        return;
    }

    struct vec3 {
        float x;
        float y;
        float z;
    };

    struct quad {
        vec3 lowerLeft;
        vec3 upperLeft;
        vec3 upperRight;
        vec3 lowerRight;
    };

    for (size_t bucketNum = 0; bucketNum != bucketList.bucketCount(); bucketNum++)
    {
        auto positions = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f*>(bucketList, bucketNum, FabricTokens::points);
        auto numQuadsSpan = stageReaderWriter.getAttributeArray<int>(bucketList, bucketNum, getNumQuadsAttributeFabricToken());
        int numQuads = numQuadsSpan[0];
        auto quadsPtr = reinterpret_cast<quad*>(positions.data());
        std::cout << "(host) numQuads: " << numQuads << std::endl;

        // for (int quadNum = 0; quadNum < numQuads; quadNum++) {
        //     printf("quad %d lowerLeft: %f, %f, %f\n", quadNum,
        //         quadsPtr[quadNum].lowerLeft.x,
        //         quadsPtr[quadNum].lowerLeft.y,
        //         quadsPtr[quadNum].lowerLeft.z);
        // }

        int elemCount = numQuads;
        if (elemCount == 0) {
            throw std::runtime_error("Fabric did not retrieve any elements");
        }
        std::cout << elemCount << std::endl;
        void *args[] = { &quadsPtr, &lookatPositionDevice, &elemCount}; //NOLINT

        cudaRunner.runKernel(args, static_cast<size_t>(elemCount));

        // primCount += static_cast<int>(elemCount);
    }

    // std::cout << "modified " << primCount << " quads" << std::endl;

    err = cuMemFree(lookatPositionDevice);
    if (err != CUDA_SUCCESS) {
        const char *errName;
        const char *errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemFree failed: %s: %s\n", errName, errStr);
        return;
    }

    // lookatPositionHost.x += 10.0;

}

glm::fvec3 usdToGlmVector(const pxr::GfVec3f& vector) {
    return {vector[0], vector[1], vector[2]};
}

void runSimpleCudaHeaderTest() {

    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    omni::fabric::AttrNameAndType cudaTestAttrTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    omni::fabric::PrimBucketList buckets = stageReaderWriter.findPrims({cudaTestAttrTag});

    if (buckets.bucketCount() == 0 ) {
        std::cout << "No prims found, returning" << std::endl;
    }

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
    }

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
    }

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog,                 // prog
                       cudaSimpleHeaderTest,  // buffer
                       nullptr,               // name
                       0,                     // numHeaders
                       nullptr,               // headers
                       nullptr);              // includeNames

    // Compile the program
    nvrtcResult res = nvrtcCompileProgram(prog, 0, nullptr);
     if (res != NVRTC_SUCCESS) {
        std::cout << "Error compiling NVRTC program:" << std::endl;

        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(prog, log);
        std::cout << "   Compilation log: \n" << log << std::endl;

        return;
    }

    // Get the PTX (assembly code for the GPU) from the compilation
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    // Load the generated PTX and get a handle to the kernel.
    CUmodule module;
    CUfunction function;
    cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    cuModuleGetFunction(&function, module, "runHeaderTest");

    auto bucketCount = buckets.bucketCount();
    printf("Num buckets: %llu\n", bucketCount);

    int primCount = 0;

    struct Vec3d
    {
        double x;
        double y;
        double z;
    };

    for (size_t bucket = 0; bucket != buckets.bucketCount(); bucket++)
    {
        gsl::span<double> sizesD = stageReaderWriter.getAttributeArrayGpu<double>(buckets, bucket, getCudaTestAttributeFabricToken());

        double* ptr = sizesD.data();
        size_t elemCount = sizesD.size();
        void *args[] = { &ptr, &elemCount }; //NOLINT
        int blockSize, minGridSize;
        cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, nullptr, 0, 0);
        //CUresult err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, 0);
        auto err = cuLaunchKernel(function, minGridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        // REQUIRE(!err);
        if (err) {
            std::cout << "error" << std::endl;
        }
    }

    std::cout << "modified " << primCount << " quads" << std::endl;

    result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }

    delete[] ptx;
}

void runCurandHeaderTest() {

    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    omni::fabric::AttrNameAndType cudaTestAttrTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    omni::fabric::PrimBucketList buckets = stageReaderWriter.findPrims({cudaTestAttrTag});

    if (buckets.bucketCount() == 0 ) {
        std::cout << "No prims found, returning" << std::endl;
    }

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
    }

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
    }

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog,                 // prog
                       curandHeaderTest,  // buffer
                       nullptr,               // name
                       0,                     // numHeaders
                       nullptr,               // headers
                       nullptr);              // includeNames

    // Compile the program
    std::string compileOptions = "--include-path=extern/nvidia/_build/target-deps/cuda/cuda/include";
    char *compileParams[1]; //NOLINT
    compileParams[0] = reinterpret_cast<char *>(malloc(sizeof(char) * (compileOptions.length() + 1))); //NOLINT
    sprintf_s(compileParams[0], sizeof(char) * (compileOptions.length() + 1),
        "%s", compileOptions.c_str());

    nvrtcResult res = nvrtcCompileProgram(prog, 1, compileParams);
     if (res != NVRTC_SUCCESS) {
        std::cout << "Error compiling NVRTC program:" << std::endl;

        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(prog, log);
        std::cout << "   Compilation log: \n" << log << std::endl;

        return;
    }

    // Get the PTX (assembly code for the GPU) from the compilation
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    // Load the generated PTX and get a handle to the kernel.
    CUmodule module;
    CUfunction function;
    cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    cuModuleGetFunction(&function, module, "runCurandTest");

    auto bucketCount = buckets.bucketCount();
    printf("Num buckets: %llu\n", bucketCount);

    //iterate over buckets but pass the vector for the whole bucket to the GPU.
    int primCount = 0;
    //test on CPU
    // for (size_t bucket = 0; bucket != buckets.bucketCount(); bucket++)
    // {
    //     auto values = stageReaderWriter.getAttributeArray<pxr::GfVec3f*>(buckets, bucket, FabricTokens::primvars_displayColor);
    //     auto elementCount = values.size();
    //     for (size_t i = 0; i < elementCount; i++) {
    //         printf("displayColors of element %llu are %f, %f, %f\n", i, values[i][0][0], values[i][0][1], values[i][0][2]);
    //         values[i][0][0] = 0;
    //         values[i][0][1] = 1.f;
    //     }
    // }

    struct Vec3d
    {
        double x;
        double y;
        double z;
    };

    for (size_t bucket = 0; bucket != buckets.bucketCount(); bucket++)
    {
        //gsl::span<double> values = stageReaderWriter.getAttributeArrayGpu<double>(buckets, bucket, getCudaTestAttributeFabricToken());
        auto values = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3d>(buckets, bucket, FabricTokens::_worldPosition);

        auto ptr = reinterpret_cast<Vec3d*>(values.data());
        size_t elemCount = values.size();
        auto seed = rand();
        void *args[] = { &ptr, &elemCount, &seed }; //NOLINT
        int blockSize = 32 * 4;
        int numBlocks = (static_cast<int>(elemCount) + blockSize - 1) / blockSize;
        // alternatively, CUDA can calculate these for you
        // int blockSize, minGridSize;
        // cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, nullptr, 0, 0);
        auto err = cuLaunchKernel(function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        if (err) {
            std::cout << "error" << std::endl;
        }
        primCount += static_cast<int>(elemCount);
    }

    std::cout << "modified " << primCount << " quads" << std::endl;

    result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }

    delete[] ptx;
}

void randomizeDVec3ViaCuda() {

    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    omni::fabric::AttrNameAndType cudaTestAttrTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    omni::fabric::PrimBucketList buckets = stageReaderWriter.findPrims({cudaTestAttrTag});

    if (buckets.bucketCount() == 0 ) {
        std::cout << "No prims found, returning" << std::endl;
    }

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
    }

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
    }

    nvrtcProgram prog;
    //nvrtcCreateProgram(&prog, kernelCode, "changeValue", 0, nullptr, nullptr);
    nvrtcCreateProgram(&prog, randomizeDVecKernelCode, "changeValue", 0, nullptr, nullptr);

    // Compile the program
    std::string compileOptions = "--include-path=extern/nvidia/_build/target-deps/cuda/cuda/include";
    char *compileParams[1]; //NOLINT
    compileParams[0] = reinterpret_cast<char *>(malloc(sizeof(char) * (compileOptions.length() + 1))); //NOLINT
    sprintf_s(compileParams[0], sizeof(char) * (compileOptions.length() + 1),
        "%s", compileOptions.c_str());

    nvrtcResult res = nvrtcCompileProgram(prog, 1, compileParams);    if (res != NVRTC_SUCCESS) {
        std::cout << "Error compiling NVRTC program:" << std::endl;

        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(prog, log);
        std::cout << "   Compilation log: \n" << log << std::endl;

        return;
    }

    // Get the PTX (assembly code for the GPU) from the compilation
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    // Load the generated PTX and get a handle to the kernel.
    CUmodule module;
    CUfunction function;
    cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    cuModuleGetFunction(&function, module, "randomizeDVec3");

    auto bucketCount = buckets.bucketCount();
    printf("Num buckets: %llu\n", bucketCount);

    //iterate over buckets but pass the vector for the whole bucket to the GPU.
    int primCount = 0;

    for (size_t bucket = 0; bucket != buckets.bucketCount(); bucket++)
    {
        //gsl::span<double> values = stageReaderWriter.getAttributeArrayGpu<double>(buckets, bucket, getCudaTestAttributeFabricToken());
        auto values = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3d>(buckets, bucket, FabricTokens::_worldPosition);

        auto ptr = reinterpret_cast<glm::dvec3*>(values.data());
        size_t elemCount = values.size();
        auto seed = rand();
        void *args[] = { &ptr, &elemCount, &seed }; //NOLINT
        int blockSize = 32 * 4;
        int numBlocks = (static_cast<int>(elemCount) + blockSize - 1) / blockSize;
        // alternatively, CUDA can calculate these for you
        // int blockSize, minGridSize;
        // cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, nullptr, 0, 0);
        auto err = cuLaunchKernel(function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        if (err) {
            std::cout << "error" << std::endl;
        }
        primCount += static_cast<int>(elemCount);
    }

    std::cout << "modified " << primCount << " quads" << std::endl;

    result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }

    delete[] ptx;
}

// no effect
void exportToUsd() {
    const auto iFabricUsd = carb::getCachedInterface<omni::fabric::IFabricUsd>();
    auto fabricId = omni::fabric::FabricId();
    iFabricUsd->exportUsdPrimData(fabricId);
    // auto usdStageRefPtr = Context::instance().getStage();
    // iFabricUsd->exportUsdPrimDataToStage(fabricId, usdStageRefPtr, 0, 0);
}

void CudaRunner::init(const char* kernelCodeDEBUG, const char* kernelFunctionName) {
    if (_initted) return;

    _kernelCode = kernelCodeDEBUG;
    _kernelFunctionName = kernelFunctionName;

    CUresult result;

    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
        throw std::runtime_error("ERROR");
    }

    // CUdevice device;
    result = cuDeviceGet(&_device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
        throw std::runtime_error("ERROR");
    }

    // CUcontext context;
    result = cuCtxCreate(&_context, 0, _device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
        throw std::runtime_error("ERROR");
    }
    auto threadId = std::this_thread::get_id();
    std::cout << "Initial thread ID: " << threadId << std::endl;


    // nvrtcProgram program;
    nvrtcCreateProgram(&_program, _kernelCode, _kernelFunctionName, 0, nullptr, nullptr);

    nvrtcResult res = nvrtcCompileProgram(_program, 0, nullptr);
    if (res != NVRTC_SUCCESS) {
        std::cout << "Error compiling NVRTC program:" << std::endl;

        size_t logSize;
        nvrtcGetProgramLogSize(_program, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(_program, log);
        std::cout << "   Compilation log: \n" << log << std::endl;
        delete[] log;

        throw std::runtime_error("ERROR");
    }

    // Get the PTX (assembly code for the GPU)
    size_t ptxSize;
    nvrtcGetPTXSize(_program, &ptxSize);
    _ptx = new char[ptxSize];
    nvrtcGetPTX(_program, _ptx);

    // CUmodule module;
    // CUfunction function;
    cuModuleLoadDataEx(&_module, _ptx, 0, nullptr, nullptr);
    auto cudaRes = cuModuleGetFunction(&_function, _module, _kernelFunctionName);
    if (cudaRes != CUDA_SUCCESS) {
        const char *errName = nullptr;
        const char *errString = nullptr;
        cuGetErrorName(cudaRes, &errName);
        cuGetErrorString(cudaRes, &errString);
        std::cout << "Error getting function: " << errName << ": " << errString << std::endl;

        std::ostringstream errMsg;
        errMsg << "Error getting function: " << errName << ": " << errString;
        throw std::runtime_error(errMsg.str());
    }

    _initted = true;
}

int animatePrims(float deltaTime, double cameraPositionX, double cameraPositionY, double cameraPositionZ,
        float cameraUpX, float cameraUpY, float cameraUpZ) {
    // std::cout << "animating " << deltaTime << std::endl;
    // const float speed = 1.5f;
    // const float radius = 600.f;
    alterPrims(cameraPositionX, cameraPositionY, cameraPositionZ, cameraUpX, cameraUpY, cameraUpZ); // TODO: no dummy vars
    elapsedTime += deltaTime;
    lookatPositionHost.x = cameraPositionX;
    lookatPositionHost.y = cameraPositionY;
    lookatPositionHost.z = cameraPositionZ;
    lookatUpHost.x = cameraUpX;
    lookatUpHost.y = cameraUpY;
    lookatUpHost.z = cameraUpZ;

    // lookatPositionHost.x = sin(elapsedTime * speed) * radius;
    // // std::cout << "x: " << lookatPositionHost.x << std::endl;
    // lookatPositionHost.y = sin(elapsedTime * speed * .93f) * radius;
    // lookatPositionHost.z = sin(elapsedTime * speed * 1.27f) * radius;
    return 0;
}


void CudaRunner::teardown() {
    auto result = cuCtxDestroy(_context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }

    delete[] _ptx;

    // untested
    // cuModuleUnload(_module);
    // nvrtcDestroyProgram(&_program);
}

CudaRunner::~CudaRunner() {
    if (_initted) teardown();
}

void CudaRunner::runKernel(void** args, size_t elemCount) {
    int blockSize = 32 * 4;
    int numBlocks = (static_cast<int>(elemCount) + blockSize - 1) / blockSize;

    auto launchResult = cuLaunchKernel(_function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
    if (launchResult) {
        const char *errName = nullptr;
        const char *errString = nullptr;

        cuGetErrorName(launchResult, &errName);
        cuGetErrorString(launchResult, &errString);

        std::cout << "Error launching kernel: " << errName << ": " << errString << std::endl;

        CUcontext currentContext;
        cuCtxGetCurrent(&currentContext);
        if (currentContext != _context) {
            std::cout << "Warning: Context has changed!" << std::endl;
            auto threadId = std::this_thread::get_id();
            std::cout << "Current thread ID: " << threadId << std::endl;

            // throw std::runtime_error("contexts don't match");
            cuCtxSetCurrent(_context);
        }
    }

    //DEBUG: bypass re-launching for new. arg params will need to be re-malloc'ed
    //retry after Context switch
    // if (launchResult) {
    //     launchResult = cuLaunchKernel(_function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
    // }

    // if (launchResult) {
    //     throw std::runtime_error("kernel still failed to launch after switch to original context\n");
    // }
}

void printPositionsWithFabric() {
   //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    auto numBuckets = bucketList.bucketCount();
    for (size_t bucketNum = 0; bucketNum < numBuckets; bucketNum++) {
        auto values = stageReaderWriter.getAttributeArray<pxr::GfVec3f*>(bucketList, bucketNum, FabricTokens::points);
        auto numQuadsSpan = stageReaderWriter.getAttributeArray<int>(bucketList, bucketNum, getNumQuadsAttributeFabricToken());
        //auto size = stageReaderWriter.getArrayAttributeSize()
        int numQuads = numQuadsSpan[0];
        auto elementCount = numQuads * 4;
        //auto data = values.data();
        std::cout << "dataSize is " << elementCount << std::endl;
        // gsl::span<pxr::GfVec3f> span(data, dataSize);
        for (int i = 0; i < elementCount; i++) {
            printf("points of element %d are %f, %f, %f\n", i, values[0][i][0], values[0][i][1], values[0][i][2]);
        }
    }
}

float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

void lookatMultiquad(quad* quads, double3* lookatPosition, int numQuads) {
    // printf("(kernel) numQuads: %d\n", numQuads);

    // printf("(kernel) lookAtPosition is (%lf, %lf, %lf)\n", lookatPosition->x, lookatPosition->y, lookatPosition->z);

    for (int i = 0; i < numQuads; i++) {
        // const float3 up = make_float3(0, 1.0, 0.0);
        const float3 quadCenter = quads[i].getCenter();
        printf("(kernel) quadCenter is (%f, %f, %f)\n", quadCenter.x, quadCenter.y, quadCenter.z);

        double3 lookAtDirectionD = make_double3(lookatPosition->x - quadCenter.x, lookatPosition->y - quadCenter.y, lookatPosition->z - quadCenter.z);
        float3 lookAtDirection = make_float3(static_cast<float>(lookAtDirectionD.x), static_cast<float>(lookAtDirectionD.y), static_cast<float>(lookAtDirectionD.z));
        lookAtDirection = normalize(lookAtDirection);
        printf("(kernel) lookAtDirection is (%f, %f, %f)\n", lookAtDirection.x, lookAtDirection.y, lookAtDirection.z);

        glm::vec3 eye(quadCenter.x, quadCenter.y, quadCenter.z);
        glm::vec3 center{lookatPosition->x, lookatPosition->y, lookatPosition->z};
        auto rotationMatrixGlm = glm::lookAt(eye, center, glm::vec3{0, 1, 0});
        rotationMatrixGlm = glm::affineInverse(rotationMatrixGlm);

        // auto llGlm = toGlm(quads[i].lowerLeft);
        // auto llGlmRotated = rotateVector(rotationMatrixGlm, llGlm);

        //translate to origin
        quads[i].lowerLeft = subtractFloat3(quads[i].lowerLeft, quadCenter);
        quads[i].lowerRight = subtractFloat3(quads[i].lowerRight, quadCenter);
        quads[i].upperLeft = subtractFloat3(quads[i].upperLeft, quadCenter);
        quads[i].upperRight = subtractFloat3(quads[i].upperRight, quadCenter);

        //rotate
        quads[i].lowerLeft = rotateVector(rotationMatrixGlm, quads[i].lowerLeft);
        quads[i].lowerRight = rotateVector(rotationMatrixGlm, quads[i].lowerRight);
        quads[i].upperLeft = rotateVector(rotationMatrixGlm, quads[i].upperLeft);
        quads[i].upperRight = rotateVector(rotationMatrixGlm, quads[i].upperRight);

        //translate back
        quads[i].lowerLeft = addFloat3(quads[i].lowerLeft, quadCenter);
        quads[i].lowerRight = addFloat3(quads[i].lowerRight, quadCenter);
        quads[i].upperLeft = addFloat3(quads[i].upperLeft, quadCenter);
        quads[i].upperRight = addFloat3(quads[i].upperRight, quadCenter);

        // auto lrGlm = toGlm(quads[i].lowerRight);
        // auto lrGlmRotated = rotateVector(rotationMatrixGlm, lrGlm);
        // quads[i].lowerRight = make_float3(lrGlmRotated.x, lrGlmRotated.y, lrGlmRotated.z);

        // glm::fquat newQuat = glm::quatLookAt(center - eye, glm::fvec3{0, 1.f, 0});
            // auto worldPositionGfVec3f = pxr::GfVec3f(worldPositions[i]);
            // auto worldPositionGlm = usdToGlmVector(worldPositionGfVec3f);
            // glm::fvec3 direction = lookatPosition - worldPositionGlm;
            // direction = glm::normalize(direction);
            // glm::fquat newQuat = glm::quatLookAt(direction, glm::fvec3{0, 1.f, 0});
            // auto rotatedQuat = convertToGf(newQuat);
            // orientations[i] = rotatedQuat;

        // mat3 rotationMatrix = matLookAtRH(lookAtDirection, up);
        // quads[i].lowerLeft = rotationMatrix.multiply(quads[i].lowerLeft);
        // quads[i].upperLeft = rotationMatrix.multiply(quads[i].upperLeft);
        // quads[i].upperRight = rotationMatrix.multiply(quads[i].upperRight);
        // quads[i].lowerRight = rotationMatrix.multiply(quads[i].lowerRight);
    }
}

mat3 matLookAtRH(float3 direction, float3 up) {
    mat3 Result;

    Result.col2 = make_float3(-direction.x, -direction.y, -direction.z);
    float3 Right = cross(up, Result.col2);
    Result.col0 = Right * sqrtf(fmax(0.0000f, dot(Right, Right)));

    Result.col1 = cross(Result.col2, Result.col0);

    return Result;
}
float3 operator*(const float3& a, const float& b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float3 normalize(float3 v) {
    float normSquared = v.x * v.x + v.y * v.y + v.z * v.z;
    float inverseSqrtNorm = 1.f / sqrtf(normSquared);
    v.x *= inverseSqrtNorm;
    v.y *= inverseSqrtNorm;
    v.z *= inverseSqrtNorm;
    return v;
}

glm::vec3 rotateVector(const glm::mat4& rotationMatrix, const glm::vec3& vectorToRotate) {
    // Convert the 3D vector to a vec4 with w component as 1.0
    glm::vec4 homogeneousVector(vectorToRotate, 1.0f);

    // Multiply the vector by the rotation matrix to perform the rotation
    glm::vec4 rotatedHomogeneousVector = rotationMatrix * homogeneousVector;

    // Convert the resulting vec4 back to a vec3
    glm::vec3 rotatedVector(rotatedHomogeneousVector.x, rotatedHomogeneousVector.y, rotatedHomogeneousVector.z);

    return rotatedVector;
}

float3 rotateVector(const glm::mat4& rotationMatrix, const float3& vectorToRotate) {
    auto result = rotateVector(rotationMatrix, toGlm(vectorToRotate));
    return make_float3(result.x, result.y, result.z);
}

glm::fvec3 toGlm(float3 input) {
    return glm::fvec3{input.x, input.y, input.z};
}

// Function to subtract a float3 vector from another float3 vector
float3 subtractFloat3(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 addFloat3(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

void createSingleQuad(pxr::GfVec3f center, float size) {
    const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    const auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(cesium::omniverse::Context::instance().getStageId())};
    const auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    const std::string primPathStub{"/quadMesh_"};

    int i = 0;

    const auto fabricPath = omni::fabric::Path((primPathStub + std::to_string(i)).c_str());
    stageReaderWriter.createPrim(fabricPath);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::primvars_displayColor, FabricTokens::primvars_displayColor);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    // attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.createAttributes(fabricPath);

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::points, 4);
    auto pointsFabric = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    auto extentScalar = size;
    pointsFabric[0] = pxr::GfVec3f(-extentScalar, -extentScalar, 0) + center;
    pointsFabric[1] = pxr::GfVec3f(-extentScalar, extentScalar, 0) + center;
    pointsFabric[2] = pxr::GfVec3f(extentScalar, extentScalar, 0) + center;
    pointsFabric[3] = pxr::GfVec3f(extentScalar, -extentScalar, 0) + center;

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexCounts, 2);
    auto faceVertexCountsFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexCounts);
    faceVertexCountsFabric[0] = 3;
    faceVertexCountsFabric[1] = 3;

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexIndices, 6);
    auto faceVertexIndicesFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexIndices);
    faceVertexIndicesFabric[0] = 0;
    faceVertexIndicesFabric[1] = 1;
    faceVertexIndicesFabric[2] = 2;
    faceVertexIndicesFabric[3] = 0;
    faceVertexIndicesFabric[4] = 2;
    faceVertexIndicesFabric[5] = 3;

    auto extent = pxr::GfRange3d(pxr::GfVec3d(-extentScalar * 2.f, -extentScalar * 2.f, 0), pxr::GfVec3d(extentScalar * 2.f, extentScalar * 2.f, 0));
    auto extentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::extent);
    *extentFabric = extent;

    auto worldExtentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::_worldExtent);
    *worldExtentFabric = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));

    auto worldVisibilityFabric = stageReaderWriter.getAttributeWr<bool>(fabricPath, FabricTokens::_worldVisibility);
    *worldVisibilityFabric = true;

    auto worldPositionFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3d>(fabricPath, FabricTokens::_worldPosition);
    *worldPositionFabric = pxr::GfVec3d(0.0, 0.0, 0.0);
    //DEBUG
    // *worldPositionFabric = pxr::GfVec3d(300.0, 300.0, 0.0);

    auto worldOrientationFabric = stageReaderWriter.getAttributeWr<pxr::GfQuatf>(fabricPath, FabricTokens::_worldOrientation);
    //*worldOrientationFabric = pxr::GfQuatf(1.f, 0, 0, 0);
    *worldOrientationFabric = pxr::GfQuatf(0.f, 0, 0, 0);

    // auto worldScaleFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::_worldScale);
    // *worldScaleFabric = pxr::GfVec3f(1.f, 1.f, 1.f);

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::primvars_displayColor, 1);
    auto displayColors = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::primvars_displayColor);
    displayColors[0] = pxr::GfVec3f(0.8f, 0.8f, 0.8f);

    //create a custom attribute for testing
    stageReaderWriter.createAttribute(fabricPath, getCudaTestAttributeFabricToken(), cudaTestAttributeFabricType);

    auto testAttribute = stageReaderWriter.getAttributeWr<double>(fabricPath, getCudaTestAttributeFabricToken());
    *testAttribute = 123.45;
}

void billboardQuad(glm::fvec3 target) {
    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    auto numBuckets = bucketList.bucketCount();

    for (size_t bucketNum = 0; bucketNum < numBuckets; bucketNum++) {
        // auto orientations = stageReaderWriter.getAttributeArray<pxr::GfQuatf>(bucketList, bucketNum, token);
        // auto worldPositions = stageReaderWriter.getAttributeArray<pxr::GfVec3d>(bucketList, bucketNum, worldPositionsTokens);
        auto points = stageReaderWriter.getAttributeArray<pxr::GfVec3f*>(bucketList, bucketNum, FabricTokens::points);
        if (points.data() == nullptr) {
            throw std::runtime_error("Fabric did not retrieve points.\n");
        }
        auto quads = reinterpret_cast<quadGlm*>(points[0]->data());

        auto numElements = 1; //TEST
        for (int i = 0; i < numElements; i++) {
            quadGlm quad = quads[i];
            auto center = quad.getCenter();

            // Compute the direction from the center of the quad to the target
            auto targetDirection = glm::normalize(target - center);
            auto forwardDirection = getForwardDirection(quad);

            // Compute the axis of rotation
            auto axis = glm::cross(forwardDirection, targetDirection);

            // Compute the cosine of the angle of rotation
            auto cosTheta = glm::dot(forwardDirection, targetDirection);

            // Compute the angle of rotation
            auto angle = glm::acos(cosTheta);

            // Create the rotation quaternion
            auto rotationQuat = glm::angleAxis(angle, axis);

            // Convert the quaternion to a rotation matrix
            auto rotationMatrix = glm::mat4_cast(rotationQuat);

            quadGlm rotatedQuad;
            rotatedQuad.lowerLeft = glm::vec3(rotationMatrix * glm::vec4(quad.lowerLeft - center, 1.0f)) + center;
            rotatedQuad.upperLeft = glm::vec3(rotationMatrix * glm::vec4(quad.upperLeft - center, 1.0f)) + center;
            rotatedQuad.upperRight = glm::vec3(rotationMatrix * glm::vec4(quad.upperRight - center, 1.0f)) + center;
            rotatedQuad.lowerRight = glm::vec3(rotationMatrix * glm::vec4(quad.lowerRight - center, 1.0f)) + center;

            quads[i].lowerLeft = rotatedQuad.lowerLeft;
            quads[i].lowerRight = rotatedQuad.lowerRight;
            quads[i].upperLeft = rotatedQuad.upperLeft;
            quads[i].upperRight = rotatedQuad.upperRight;
        }
    }
}

void billboardMultiQuadCpu(glm::fvec3 target, glm::fvec3 targetUp) {
    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    auto numBuckets = bucketList.bucketCount();

    for (size_t bucketNum = 0; bucketNum < numBuckets; bucketNum++) {
        auto numQuadsSpan = stageReaderWriter.getAttributeArray<int>(bucketList, bucketNum, getNumQuadsAttributeFabricToken());
        int numQuads = numQuadsSpan[0];

        auto points = stageReaderWriter.getAttributeArray<pxr::GfVec3f*>(bucketList, bucketNum, FabricTokens::points);
        if (points.data() == nullptr) {
            throw std::runtime_error("Fabric did not retrieve points.\n");
        }
        auto quads = reinterpret_cast<quadGlm*>(points[0]->data());

        auto numElements = numQuads;
        for (int i = 0; i < numElements; i++) {
            rotateQuadToTarget(quads, i, target, targetUp);
        }
    }
}

void billboardMultiQuadCuda(glm::fvec3 lookatPosition, glm::fvec3 lookatUp) {
    lookatPositionHost.x = static_cast<double>(lookatPosition.x);
    lookatPositionHost.y = static_cast<double>(lookatPosition.y);
    lookatPositionHost.z = static_cast<double>(lookatPosition.z);

    lookatUpHost.x = lookatUp.x;
    lookatUpHost.y = lookatUp.y;
    lookatUpHost.z = lookatUp.z;

    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    cudaRunner.init(lookAtMultiquadKernelCode2, "lookAtMultiquadKernel");

    CUresult err;
    CUdeviceptr lookatPositionDevice;

    err = cuMemAlloc(&lookatPositionDevice, sizeof(glm::dvec3));
    if (err != CUDA_SUCCESS) {
        const char *errName;
        const char *errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemAlloc failed: %s: %s\n", errName, errStr);
        return;
    }

    err = cuMemcpyHtoD(lookatPositionDevice, &lookatPositionHost, sizeof(glm::dvec3));
    if (err != CUDA_SUCCESS) {
        const char *errName;
        const char *errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemcpyHtoD failed: %s: %s\n", errName, errStr);
        return;
    }

    CUdeviceptr lookatUpDevice;
    err = cuMemAlloc(&lookatUpDevice, sizeof(glm::fvec3));
    if (err != CUDA_SUCCESS) {
        const char *errName;
        const char *errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemAlloc failed: %s: %s\n", errName, errStr);
        return;
    }

    err = cuMemcpyHtoD(lookatUpDevice, &lookatUpHost, sizeof(glm::fvec3));
    if (err != CUDA_SUCCESS) {
        const char *errName;
        const char *errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemcpyHtoD failed: %s: %s\n", errName, errStr);
        return;
    }


    for (size_t bucketNum = 0; bucketNum != bucketList.bucketCount(); bucketNum++)
    {
        auto positions = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f*>(bucketList, bucketNum, FabricTokens::points);
        auto numQuadsSpan = stageReaderWriter.getAttributeArray<int>(bucketList, bucketNum, getNumQuadsAttributeFabricToken());
        int numQuads = numQuadsSpan[0];
        auto quadsPtr = reinterpret_cast<quad*>(positions.data());
        // std::cout << "(host) numQuads: " << numQuads << std::endl;

        // for (int quadNum = 0; quadNum < numQuads; quadNum++) {
        //     printf("quad %d lowerLeft: %f, %f, %f\n", quadNum,
        //         quadsPtr[quadNum].lowerLeft.x,
        //         quadsPtr[quadNum].lowerLeft.y,
        //         quadsPtr[quadNum].lowerLeft.z);
        // }

        int elemCount = numQuads;
        if (elemCount == 0) {
            throw std::runtime_error("Fabric did not retrieve any elements");
        }
        // std::cout << elemCount << std::endl;
        void *args[] = { &quadsPtr, &lookatPositionDevice, &lookatUpDevice, &elemCount}; //NOLINT

        cudaRunner.runKernel(args, static_cast<size_t>(elemCount));

        // primCount += static_cast<int>(elemCount);
    }

    // std::cout << "modified " << primCount << " quads" << std::endl;

    err = cuMemFree(lookatPositionDevice);
    if (err != CUDA_SUCCESS) {
        const char *errName;
        const char *errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemFree failed: %s: %s\n", errName, errStr);
        return;
    }

    err = cuMemFree(lookatUpDevice);
    if (err != CUDA_SUCCESS) {
        const char *errName;
        const char *errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemFree failed: %s: %s\n", errName, errStr);
        return;
    }

    // lookatPositionHost.x += 10.0;

}


glm::fvec3 toGlm(pxr::GfVec3f input) {
    return glm::fvec3{input[0], input[1], input[2]};
}

glm::fvec3 multiplyHomogenous(const glm::mat4 transformationMatrix, const glm::fvec3 point) {
    glm::vec4 homogeneousPoint{point, 1.0f};

    // Apply the transformation
    glm::vec4 transformedHomogeneousPoint = transformationMatrix * homogeneousPoint;
    return glm::vec3{transformedHomogeneousPoint.x, transformedHomogeneousPoint.y, transformedHomogeneousPoint.z};
}

glm::vec3 getForwardDirection(const quadGlm& quad) {
    // Calculate two edge vectors of the quad
    glm::vec3 up = quad.lowerLeft - quad.upperLeft;
    glm::vec3 right = quad.lowerRight - quad.lowerLeft;

    // Calculate the normal vector (forward direction) of the quad using the cross product of the edge vectors
    glm::vec3 forward = glm::normalize(glm::cross(up, right));

    return forward;
}

void rotateQuadToTarget(quadGlm* quads, int quadIndex, const glm::vec3& target, const glm::vec3& targetUp) {
    auto targetUpN = glm::normalize(targetUp);
    auto quadCenter = quads[quadIndex].getCenter();
    auto newQuadForward = target - quadCenter;
    auto newQuadForwardN = glm::normalize(newQuadForward);
    glm::fvec3 newQuadRight;
    glm::fvec3 newQuadUp;
    if (almostEquals(newQuadForwardN, targetUpN)) {
        //directly beneath the camera, no op
        return;
    } else {
        newQuadRight = glm::normalize(glm::cross(newQuadForward, targetUp));
        newQuadUp = glm::normalize(glm::cross(newQuadRight, newQuadForward));
    }

    // auto translationMatrix = glm::mat3x3(newQuadForward, newQuadRight, newQuadUp);
    auto translationMatrix = glm::mat3x3(newQuadRight, newQuadUp, newQuadForwardN);
    // auto shiftedUL = quad.upperLeft - quadCenter;
    // auto shiftedLL = quad.lowerLeft - quadCenter;
    // auto shiftedUR = quad.upperRight - quadCenter;
    // auto shiftedLR = quad.lowerRight - quadCenter;

    //quad points are assumed to be in XY plane
    auto rotatedLL = translationMatrix * glm::fvec3(-1.0f, -1.0f, 0);
    auto rotatedUL = translationMatrix * glm::fvec3(-1.0f, 1.0f, 0);
    auto rotatedUR = translationMatrix * glm::fvec3(1.0f, 1.0f, 0);
    auto rotatedLR = translationMatrix * glm::fvec3(1.0f, -1.0f, 0);
    auto newQuadUL = rotatedUL + quadCenter;
    auto newQuadUR = rotatedUR + quadCenter;
    auto newQuadLL = rotatedLL + quadCenter;
    auto newQuadLR = rotatedLR + quadCenter;

    quads[quadIndex].upperLeft = newQuadUL;
    quads[quadIndex].upperRight = newQuadUR;
    quads[quadIndex].lowerLeft = newQuadLL;
    quads[quadIndex].lowerRight = newQuadLR;
}

// void rotateQuadToTarget(quadGlm& quad, const glm::vec3& target) {
//     // Compute the direction from the center of the quad to the target
//     glm::vec3 center = quad.getCenter();
//     glm::vec3 targetDirection = glm::normalize(target - center);

//     // Compute the quad's current forward direction
//     glm::vec3 currentForward = getForwardDirection(quad);

//     // Compute the angle and axis for the rotation
//     float angle = glm::acos(glm::dot(currentForward, targetDirection));
//     glm::vec3 axis = glm::normalize(glm::cross(currentForward, targetDirection));

//     // Compute the rotation matrix
//     glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), angle, axis);

//     // Apply the rotation to each vertex of the quad
//     quad.lowerLeft = glm::vec3(rotation * glm::vec4(quad.lowerLeft - center, 1.0)) + center;
//     quad.upperLeft = glm::vec3(rotation * glm::vec4(quad.upperLeft - center, 1.0)) + center;
//     quad.upperRight = glm::vec3(rotation * glm::vec4(quad.upperRight - center, 1.0)) + center;
//     quad.lowerRight = glm::vec3(rotation * glm::vec4(quad.lowerRight - center, 1.0)) + center;
// }

void printQuad(quadGlm q) {
    printf("Quad info:\n");
    printf("  center is is (%f, %f, %f)\n", q.getCenter().x, q.getCenter().y, q.getCenter().z);
    printf("  *upperLeft is (%f, %f, %f)\n", q.upperLeft.x, q.upperLeft.y, q.upperRight.z);
    printf("  *lowerLeft is (%f, %f, %f)\n", q.lowerLeft.x, q.lowerLeft.y, q.lowerLeft.z);
}

bool almostEquals(glm::vec3 a, glm::vec3 b) {
    const float epsilon = 0.0000001f;
    if (abs(a.x - b.x) > epsilon) return false;
    if (abs(a.y - b.y) > epsilon) return false;
    if (abs(a.z - b.z) > epsilon) return false;

    return true;
}

void printMultiquadWithCuda() {
    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    cudaRunner.init(printMultiquadKernel, "printMultiquad");

    for (size_t bucketNum = 0; bucketNum != bucketList.bucketCount(); bucketNum++)
    {
        auto positions = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f*>(bucketList, bucketNum, FabricTokens::points);
        auto numQuadsSpan = stageReaderWriter.getAttributeArray<int>(bucketList, bucketNum, getNumQuadsAttributeFabricToken());
        int numQuads = numQuadsSpan[0];
        auto quadsPtr = reinterpret_cast<quad*>(positions.data());
        std::cout << "(host) numQuads: " << numQuads << std::endl;


        int elemCount = numQuads;
        if (elemCount == 0) {
            throw std::runtime_error("Fabric did not retrieve any elements");
        }
        std::cout << elemCount << std::endl;
        void *args[] = { &quadsPtr, &elemCount}; //NOLINT

        cudaRunner.runKernel(args, static_cast<size_t>(elemCount));
    }
}

void printMultiquadPointsWithCuda() {
    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    auto bucketList = stageReaderWriter.findPrims({primTag});

    cudaRunner.init(printPointsKernel, "printPoints");

    for (size_t bucketNum = 0; bucketNum != bucketList.bucketCount(); bucketNum++)
    {
        auto positions = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f*>(bucketList, bucketNum, FabricTokens::points);
        auto numQuadsSpan = stageReaderWriter.getAttributeArray<int>(bucketList, bucketNum, getNumQuadsAttributeFabricToken());
        int numQuads = numQuadsSpan[0];
        auto quadsPtr = reinterpret_cast<quad*>(positions.data());

        auto elemCount = numQuads * 4;
        if (elemCount == 0) {
            throw std::runtime_error("Fabric did not retrieve any elements");
        }
        std::cout << elemCount << std::endl;
        void *args[] = { &quadsPtr, &elemCount}; //NOLINT

        cudaRunner.runKernel(args, static_cast<size_t>(elemCount));
    }
}

} // namespace cesium::omniverse::FabricProceduralGeometry
