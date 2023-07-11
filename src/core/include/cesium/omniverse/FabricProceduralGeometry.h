#pragma once

#include <cuda/include/cuda.h>
#include <cuda/include/nvrtc.h>
#include <cuda/include/cuda_runtime.h>

namespace cesium::omniverse::FabricProceduralGeometry {

int runExperiment();
void modifyUsdPrim();
void modify1000PrimsWithFabric();
void modify1000UsdPrimsViaCuda();
void createQuadMeshViaFabric();
void editSingleFabricAttributeViaCuda();
void createQuadViaFabricAndCuda();
void createQuadsViaFabric(int numQuads);
void modifyQuadsViaCuda();
void createAndModifyQuadsViaCuda(int numQuads);

CUfunction compileKernel(const char *kernelSource, const char *kernelName);
CUfunction compileKernel2(const char *kernelSource, const char *kernelName);
bool checkCudaCompatibility();

__global__ void addArrays(int n, float* x, float* y);
void addOneMillionCPU();
void addOneMillionCuda();

}
