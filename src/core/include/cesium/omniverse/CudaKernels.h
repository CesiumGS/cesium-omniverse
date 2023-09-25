#pragma once

namespace cesium::omniverse::cudaKernels {

inline const char* printPointsKernel = R"(

    extern "C" __global__ void printPoints(float3** points, int numPoints) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numPoints) return;

        int pointIndex = static_cast<int>(i);

        // printf("point %d of %d\n", pointIndex, numPoints);
        printf("point %d: %f, %f, %f\n", pointIndex, points[0][pointIndex].x, points[0][pointIndex].y, points[0][pointIndex].z);
    }
    )";

inline const char* printQuadsKernel = R"(

    struct quad {
        float3 lowerLeft;
        float3 upperLeft;
        float3 upperRight;
        float3 lowerRight;

        __device__ float3 getCenter() {
            return make_float3(
                (lowerLeft.x + upperLeft.x + upperRight.x + lowerRight.x) * .25f,
                (lowerLeft.y + upperLeft.y + upperRight.y + lowerRight.y) * .25f,
                (lowerLeft.z + upperLeft.z + upperRight.z + lowerRight.z) * .25f);
        }
    };

    extern "C" __global__ void run(quad** quads, int numQuads) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numQuads) return;

        int quadIndex = static_cast<int>(i);

        printf("Quad %d upper left: %f, %f, %f\n", quadIndex, quads[0][quadIndex].upperLeft.x, quads[0][quadIndex].upperLeft.y, quads[0][quadIndex].upperLeft.z);
        // quads[0][quadIndex].upperRight = newQuadUR;
        // quads[0][quadIndex].lowerLeft = newQuadLL;
        // quads[0][quadIndex].lowerRight = newQuadLR;
    }
    )";

inline const char* helloWorldKernel = R"(
    extern "C" __global__
    void helloWorld(size_t count)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (count <= i) return;

        printf("Hello world, from index %llu\n", i);
    }
    )";

inline const char* printFloatKernel = R"(
    extern "C" __global__
    void printFloat(float floatVal, size_t count)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (count <= i) return;

        printf("At index %llu float is %f\n", i, floatVal);
    }
    )";

inline const char* createVoxelsKernel = R"(
    extern "C" __global__
    void createVoxels(float3** points, size_t count)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (count <= i) return;

        // printf("Placeholder: create voxel at index %llu\n", i);
    }
    )";

inline const char* lookAtQuadsKernel = R"(

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
            (lowerLeft.x + upperLeft.x + upperRight.x + lowerRight.x) * .25f,
            (lowerLeft.y + upperLeft.y + upperRight.y + lowerRight.y) * .25f,
            (lowerLeft.z + upperLeft.z + upperRight.z + lowerRight.z) * .25f);
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

extern "C" __global__ void lookAtQuads(quad** quads, double3* lookAtPosition, float3* lookAtUp, float *quadSize, int numQuads) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numQuads) return;

    int quadIndex = static_cast<int>(i);

    // printf("Quad %d upper left: %f, %f, %f\n", quadIndex, quads[0][quadIndex].upperLeft.x, quads[0][quadIndex].upperLeft.y, quads[0][quadIndex].upperLeft.z);
    // printf("Quad %d lookAtPosition is %lf, %lf, %lf", lookAtPosition->x, lookAtPosition->y, lookAtPosition->z);
    // printf("Quad %d lookAtUp is %f, %f, %f", lookAtUp->x, lookAtUp->y, lookAtUp->z);

    const float quadHalfSize = *quadSize * 0.5f;

    float3 targetUpN = *lookAtUp;
    float3 quadCenter = quads[0][quadIndex].getCenter();
    double3 quadCenterD = make_double3(static_cast<double>(quadCenter.x), static_cast<double>(quadCenter.y) , static_cast<double>(quadCenter.z));
    double3 newQuadForwardDouble = subtractDouble3(*lookAtPosition, quadCenterD);
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
    float3 rotatedLL = translationMatrix.multiply(make_float3(-quadHalfSize, -quadHalfSize, 0));
    float3 rotatedUL = translationMatrix.multiply(make_float3(-quadHalfSize, quadHalfSize, 0));
    float3 rotatedUR = translationMatrix.multiply(make_float3(quadHalfSize, quadHalfSize, 0));
    float3 rotatedLR = translationMatrix.multiply(make_float3(quadHalfSize, -quadHalfSize, 0));
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
} // namespace cesium::omniverse::cudaKernels
