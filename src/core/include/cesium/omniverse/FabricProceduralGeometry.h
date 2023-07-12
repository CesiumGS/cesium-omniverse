#pragma once

#include <cuda/include/cuda.h>
#include <cuda/include/nvrtc.h>
#include <cuda/include/cuda_runtime.h>

namespace cesium::omniverse::FabricProceduralGeometry {

int runExperiment();

/* EXPERIMENTS */

 //Sanity check. Modify single USD Cube prim using Fabric.
 //Issues:
 // 1) changes do not write back to USD stage
 // 2) "size" attribute does not update, though "cudaTest" does
void modifyUsdCubePrimWithFabric();

//Sanity check. Modify 1000 USD Cube prims with Fabric
//Issues:
 // 1) changes do not write back to USD stage
 // 2) "size" attribute does not update, though "cudaTest" does
void modify1000UsdCubePrimsWithFabric();


void modify1000UsdCubesViaCuda(); //create 1000 cubes with USD, modify "cudaTest" param via CUDA
void modify1000UsdQuadsViaCuda(); //does not work, meshes throw runtime error
void editSingleFabricAttributeViaCuda();
void modifyQuadsViaCuda();

void createFabricQuadsModifyViaCuda(int numQuads);


void createQuadViaFabricAndCuda(); // old experiment



/* GEOMETRY CREATION */

void createQuadMeshViaUsd(const char* path, float maxCenterRandomization = 0);
void createQuadMeshViaFabric();
void createQuadsViaFabric(int numQuads);



/* ANIMATIONS */

void alterScale();



/* CUDA SPECIFIC */

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
