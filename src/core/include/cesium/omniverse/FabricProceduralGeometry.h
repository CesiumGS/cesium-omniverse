#pragma once

#include <cuda/include/cuda.h>
#include <cuda/include/nvrtc.h>
#include <cuda/include/cuda_runtime.h>
#include "glm/glm.hpp"
#include "pxr/base/gf/quatd.h"
#include "pxr/base/gf/quatf.h"
#include "glm/gtc/quaternion.hpp"
#include <cmath>

namespace cesium::omniverse::FabricProceduralGeometry {

int createPrims();
int alterPrims();
int animatePrims(float deltaTime);

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
        void teardown();

    public:
        void runKernel(void** args, size_t elemCount);
        void init(const char* kernelCode, const char* kernelFunctionName);

        ~CudaRunner();
};

/* EXPERIMENTS *//////////////////////////////////////////

//Sanity check. Modify single USD Cube Prim using Fabric.
//Issues:
// 1) changes do not write back to USD stage
// 2) "size" attribute does not update, though "cudaTest" does
void modifyUsdCubePrimWithFabric();

//Sanity check. Modify 1000 USD Cube Prims with Fabric
//Issues:
// 1) changes do not write back to USD stage
// 2) "size" attribute does not update, though "cudaTest" does
void modify1000UsdCubePrimsWithFabric();

//Create 1000 Cube Prims with USD, modify "cudaTest" param via Fabric/CUDA
//Issues: works
void modify1000UsdCubesViaCuda();

//Create 1000 quad Mesh Prims, modify "cudaTest" param via Fabric/CUDA
//Issues:
// Runtime error during prefetching if creating a UsdGeomMesh. UsdGeomGprim is fine. (double check this statement)
void modify1000UsdQuadsViaCuda();

//old test, ignored.
//Create a single quad Mesh in Fabric, edit "cudaTest" attribute with CUDA
//Can test editing without using PrimBucketList
void editSingleFabricAttributeViaCuda();


// Create single quad Mesh in Fabric, shift X position of points via CUDA
// More complex kernel to edit pxr::GfVec3fs with in CUDA
// Issues: not working. TODO: notes
void createQuadViaFabricAndShiftWithCuda();

//Create N quad Mesh Prims in Fabric, edit "cudaTest" attr with CUDA
void createFabricQuadsModifyViaCuda(int numQuads);

void alterFabricPrimTranslationWithFabric();


void setDisplayColor();

//Create USD cubes, translate with USD
//Issues: throws runtime error
void alterUsdPrimTranslationWithUsd();

//create USD cubes, translate with Fabric
//Issues: translates, but not visible
void alterUsdPrimTranslationWithFabric();


/* PRIM CREATION */////////////////////////////////////

void createQuadMeshViaUsd(const char* path, float maxCenterRandomization = 0);
void createQuadMeshViaFabric();
void createQuadsViaFabric(int numQuads, float maxCenterRandomization = 0);
void createQuadMeshWithDisplayColor();

/* PRIM ALTERATIONS */////////////////////////////////////

//Get all prims with "cudaTest" attr and edits the position with Fabric
void repositionAllPrimsWithCustomAttrViaFabric(double spacing = 10.0);
void repositionAllPrimsWithCustomAttrViaCuda(double spacing = 10.0);

void randomizePrimWorldPositionsWithCustomAttrViaCuda();
void randomizeDVec3ViaCuda();

void rotateAllPrimsWithCustomAttrViaFabric();

void billboardAllPrimsWithCustomAttrViaFabric();
void billboardAllPrimsWithCustomAttrViaCuda();

void runSimpleCudaHeaderTest();
void runCurandHeaderTest();
void exportToUsd();


//Get all prims with "cudaTest" attr and edits the attr with CUDA
//Issues: TODO
void modifyAllPrimsWithCustomAttrViaCuda();

// CONVERSION

// glm::dquat convertToGlm(const pxr::GfQuatd& quat);
pxr::GfQuatd convertToGf(const glm::dquat& quat);
glm::fquat convertToGlm(const pxr::GfQuatf& quat);
pxr::GfQuatf convertToGf(const glm::fquat& quat);
glm::fvec3 usdToGlmVector(const pxr::GfVec3f& vector);


/* CUDA SPECIFIC *//////////////////////////////////////////

// Compilation method without nvrtcProgram. Compilation log is less informative.
CUfunction compileKernel(const char *kernelSource, const char *kernelName);
CUfunction compileKernel2(const char *kernelSource, const char *kernelName);
bool checkCudaCompatibility();

//kernel as C function
__global__ void addArrays(int n, float* x, float* y);

//initial CUDA tests
void addOneMillionCPU();
void addOneMillionCuda();
}
