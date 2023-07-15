#pragma once

#include <cuda/include/cuda.h>
#include <cuda/include/nvrtc.h>
#include <cuda/include/cuda_runtime.h>

namespace cesium::omniverse::FabricProceduralGeometry {

int createPrims();
int alterPrims();



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

//Get all prims with "cudaTest" attr and edits the attr with CUDA
//Issues: TODO
void modifyAllPrimsWithCustomAttrViaCuda();



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
