#pragma once

#include <cuda/include/cuda.h>
#include <cuda/include/nvrtc.h>
#include <cuda/include/cuda_runtime.h>

namespace cesium::omniverse::FabricProceduralGeometry {

int createCube();
void modifyUsdPrim();
void modify1000Prims();
void modify1000PrimsViaCuda();
CUfunction compileKernel(const char *kernelSource, const char *kernelName);
bool checkCudaCompatibility();

}
