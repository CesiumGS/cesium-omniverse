#pragma once

#include <string>
#include <cuda/include/cuda.h>
#include <cuda/include/nvrtc.h>
#include <cuda/include/cuda_runtime.h>
#include <glm/fwd.hpp>
#include "glm/glm.hpp"
#include "pxr/base/gf/quatd.h"
#include "pxr/base/gf/quatf.h"
#include "glm/gtc/quaternion.hpp"
#include <cmath>

namespace cesium::omniverse::CudaTest {

struct quad {
    float3 lowerLeft;
    float3 upperLeft;
    float3 upperRight;
    float3 lowerRight;

    float3 getCenter() {
        return make_float3(
            (lowerLeft.x + upperRight.x) * .5f,
            (lowerLeft.y + upperRight.y) * .5f,
            0);
    }
};

class CudaRunner {
private:
    const char* _kernelCode;
    const char* _kernelFunctionName;
    CUdevice _device;
    CUcontext _context;
    nvrtcProgram _program;
    char* _ptx;
    CUmodule _module;
    CUfunction _function;
    bool _initted = false;
    int _blockSize, _numBlocks;
    void teardown();

public:
    void runKernel(void** args, size_t elemCount);
    void init(const char* kernelCode, const char* kernelFunctionName);

    ~CudaRunner();
};

void runTestCode();

int createPrims();
void createMultiquadFromPtsFile(const std::string &ptsFile, float quadSize);

int alterPrims(double cameraPositionX, double cameraPositionY, double cameraPositionZ,
    float cameraUpX, float cameraUpY, float cameraUpZ);
int animatePrims(float deltaTime, double cameraPositionX, double cameraPositionY, double cameraPositionZ,
        float cameraUpX, float cameraUpY, float cameraUpZ);
void billboardMultiQuadCuda(glm::fvec3 target, glm::fvec3 targetUp);
}
