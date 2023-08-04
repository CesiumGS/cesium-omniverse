#pragma once

#include <corecrt_math.h>
#include <cuda/include/cuda.h>
#include <cuda/include/nvrtc.h>
#include <cuda/include/cuda_runtime.h>
#include <glm/fwd.hpp>
#include "glm/glm.hpp"
#include "pxr/base/gf/quatd.h"
#include "pxr/base/gf/quatf.h"
#include "glm/gtc/quaternion.hpp"
#include <cmath>

namespace cesium::omniverse::FabricProceduralGeometry {

int createPrims();
int alterPrims(double cameraPositionX, double cameraPositionY, double cameraPositionZ,
float cameraUpX, float cameraUpY, float cameraUpZ);
int animatePrims(float deltaTime, double cameraPositionX, double cameraPositionY, double cameraPositionZ,
        float cameraUpX, float cameraUpY, float cameraUpZ);

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

struct quad {
    //TODO: vec3 to float3
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

struct quadPxr {
    //TODO: vec3 to float3
    pxr::GfVec3f lowerLeft;
    pxr::GfVec3f upperLeft;
    pxr::GfVec3f upperRight;
    pxr::GfVec3f lowerRight;

    pxr::GfVec3f getCenter() {
        auto center = lowerLeft + upperLeft + upperRight + lowerRight;
        center *= .25f;
        return center;
    }
};

struct quadGlm {
    //TODO: vec3 to float3
    glm::fvec3 lowerLeft;
    glm::fvec3 upperLeft;
    glm::fvec3 upperRight;
    glm::fvec3 lowerRight;

    glm::fvec3 getCenter() {
        auto center = lowerLeft + upperLeft + upperRight + lowerRight;
        center *= .25f;
        return center;
    }
};

float dot(float3 a, float3 b);

struct mat3 {
    float3 col0;
    float3 col1;
    float3 col2;

    [[nodiscard]] float3 multiply(float3 vec) const {
        float3 result;
        result.x = dot(col0, vec);
        result.y = dot(col1, vec);
        result.z = dot(col2, vec);
        return result;
    }
};

float3 normalize(float3 v);

float3 cross(float3 a, float3 b);

float3 operator*(const float3& a, const float& b);

mat3 matLookAtRH(float3 direction, float3 up);

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
void createMultiquadViaFabric();
void createMultiquadMeshViaFabric2(size_t size);
void createQuadsViaFabric(int numQuads, float maxCenterRandomization = 0);
void createQuadMeshWithDisplayColor();
void createSingleQuad(pxr::GfVec3f center, float size);
void createMultiquadFromPtsFile(const std::string &ptsFile);

/* PRIM ALTERATIONS */////////////////////////////////////

//Get all prims with "cudaTest" attr and edits the position with Fabric
void repositionAllPrimsWithCustomAttrViaFabric(double spacing = 10.0);
void repositionAllPrimsWithCustomAttrViaCuda(double spacing = 10.0);

void randomizePrimWorldPositionsWithCustomAttrViaCuda();
void randomizeDVec3ViaCuda();

void rotateAllPrimsWithCustomAttrViaFabric();

void billboardAllPrimsWithCustomAttrViaFabric();
void billboardAllPrimsWithCustomAttrViaCuda();
void billboardMultiquadWithCustomAttrViaFabric();
void billboardMultiquadWithCustomAttrViaCuda();
void billboardQuad(glm::fvec3 target);
void billboardMultiQuadCpu(glm::fvec3 target, glm::fvec3 targetUp);
void billboardMultiQuadCuda(glm::fvec3 target, glm::fvec3 targetUp);
void printPositionsWithFabric();
void printMultiquadPointsWithCuda();
void runSimpleCudaHeaderTest();
void runCurandHeaderTest();
void exportToUsd();
void rotateQuadToTarget(quadGlm* quads, int quadIndex, const glm::vec3& target, const glm::vec3& up);
void printPointsWithCuda();
void printMultiquadWithCuda();

//Get all prims with "cudaTest" attr and edits the attr with CUDA
//Issues: TODO
void modifyAllPrimsWithCustomAttrViaCuda();

// CONVERSION

// glm::dquat convertToGlm(const pxr::GfQuatd& quat);
pxr::GfQuatd convertToGf(const glm::dquat& quat);
glm::fquat convertToGlm(const pxr::GfQuatf& quat);
pxr::GfQuatf convertToGf(const glm::fquat& quat);
glm::fvec3 usdToGlmVector(const pxr::GfVec3f& vector);

/* HELPERS */
void lookatMultiquad(quad* quads, double3* lookatPosition, int numQuads);
glm::vec3 rotateVector(const glm::mat4& rotationMatrix, const glm::vec3& vectorToRotate);
float3 rotateVector(const glm::mat4& rotationMatrix, const float3& vectorToRotate);
glm::fvec3 toGlm(float3 input);
glm::fvec3 toGlm(pxr::GfVec3f input);
// Function to subtract a float3 vector from another float3 vector
float3 subtractFloat3(const float3& a, const float3& b);
float3 addFloat3(const float3& a, const float3& b);
glm::fvec3 multiplyHomogenous(const glm::mat4 transformationMatrix, const glm::fvec3 point);
glm::vec3 getForwardDirection(const quadGlm& quad);
void printQuad(quadGlm q);
bool almostEquals(glm::vec3, glm::vec3);

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
