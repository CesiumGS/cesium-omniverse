#pragma once

#include "cesium/omniverse/UsdUtil.h"

#include <cuda/include/cuda.h>
#include <cuda/include/cuda_runtime.h>
#include <cuda/include/nvrtc.h>

namespace cesium::omniverse::FabricProceduralGeometry {

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
    bool runKernel(void** args, size_t elemCount);
    void init(const char* kernelCode, const char* kernelFunctionName);

    ~CudaRunner();
};

class CudaError {
  public:
    static void check(CUresult err, const char* operation) {
        if (err != CUDA_SUCCESS) {
            const char* errName;
            const char* errStr;
            cuGetErrorName(err, &errName);
            cuGetErrorString(err, &errStr);
            printf("%s failed: %s: %s\n", operation, errName, errStr);
            throw std::runtime_error(errStr);
        }
    }
};

void createMultiquadFromPtsFile(const std::string& ptsFile, float quadSize, float scale = 1.0f);
void billboardMultiQuadCuda(glm::fvec3 target, glm::fvec3 targetUp);
int createPrims();
int alterPrims(
    double cameraPositionX,
    double cameraPositionY,
    double cameraPositionZ,
    float cameraUpX,
    float cameraUpY,
    float cameraUpZ);
int animatePrims(
    float deltaTime,
    double cameraPositionX,
    double cameraPositionY,
    double cameraPositionZ,
    float cameraUpX,
    float cameraUpY,
    float cameraUpZ);

omni::fabric::Token getBillboardedAttributeFabricToken();
omni::fabric::Token getNumQuadsAttributeFabricToken();
CUdeviceptr allocAndCopyToDevice(void* hostPtr, size_t size);
void freeDeviceMemory(CUdeviceptr devicePtr);
void makeInitialReadCall();

} // namespace cesium::omniverse::FabricProceduralGeometry
