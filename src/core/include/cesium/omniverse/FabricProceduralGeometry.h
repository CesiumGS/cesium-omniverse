#pragma once

#include <cuda/include/cuda.h>
#include <cuda/include/nvrtc.h>
#include <cuda/include/cuda_runtime.h>

namespace cesium::omniverse::FabricProceduralGeometry {

int runExperiment();

//experiments
void modifyUsdPrim();
void modify1000UsdPrimsWithFabric();
void modify1000UsdCubesViaCuda();
void modify1000UsdQuadsViaCuda(); //does not work, meshes throw runtime error
void editSingleFabricAttributeViaCuda();
void modifyQuadsViaCuda();
void createFabricQuadsModifyViaCuda(int numQuads);
void createQuadViaFabricAndCuda();
void createQuadsViaFabric(int numQuads);

//geometry creation
void createQuadMeshViaUsd(const char* path, float maxCenterRandomization = 0);
void createQuadMeshViaFabric();

CUfunction compileKernel(const char *kernelSource, const char *kernelName);
CUfunction compileKernel2(const char *kernelSource, const char *kernelName);
bool checkCudaCompatibility();

__global__ void addArrays(int n, float* x, float* y);
void addOneMillionCPU();
void addOneMillionCuda();

}
