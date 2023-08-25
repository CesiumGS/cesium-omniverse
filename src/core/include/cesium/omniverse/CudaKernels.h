#pragma once

namespace cesium::omniverse::cudaKernels {

    inline const char* printPointsKernel = R"(

    extern "C" __global__ void printPoints(float3** points, int numPoints) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numPoints) return;

        int pointIndex = static_cast<int>(i);

        printf("point %d: %f, %f, %f\n", pointIndex, points[0][pointIndex].x, points[0][pointIndex].y, points[0][pointIndex].z);
    }
    )";

    inline const char* helloWorldKernel = R"(
    extern "C" __global__
    void helloWorld(double* values, size_t count)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (count <= i) return;

        printf("Hello world, from index %llu\n", i);
    }
    )";

    inline const char* createVoxelsKernel = R"(
    extern "C" __global__
    void createVoxels(float3** points, size_t count)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (count <= i) return;

        printf("Placeholder: create voxel at index %llu\n", i);
    }
    )";
}
