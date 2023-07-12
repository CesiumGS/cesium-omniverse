#include "cesium/omniverse/FabricProceduralGeometry.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <glm/gtc/random.hpp>
#include <omni/fabric/FabricUSD.h>
#include <omni/fabric/IFabric.h>
#include <carb/Framework.h>

// #include <omni/usd/omni.h>
// #include <omni/usd/UsdContextIncludes.h>
// #include <omni/usd/UsdContext.h>

#include "pxr/base/tf/token.h"

#include <pxr/usd/usd/prim.h>
#include <iostream>
#include <omni/gpucompute/GpuCompute.h>
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/xform.h"
#include "pxr/usd/usdGeom/xformable.h"

namespace cesium::omniverse::FabricProceduralGeometry {


constexpr int numPrimsForExperiment = 99;

const omni::fabric::Type cudaTestAttributeFabricType(omni::fabric::BaseDataType::eDouble, 1, 0, omni::fabric::AttributeRole::eNone);
omni::fabric::Token getCudaTestAttributeFabricToken() {
    static const auto cudaTestAttributeFabricToken = omni::fabric::Token("cudaTest");
    return cudaTestAttributeFabricToken;
}

//CUDA via CUDA_JIT and string
const char* kernelCode = R"(
extern "C" __global__
void changeValue(double* values, size_t count)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (count <= i) return;

    float oldVal = values[i];
    values[i] = 543.21;
    printf("Changed value of index %llu from %lf to %lf\n", i, oldVal, values[i]);
}
)";

int runExperiment() {
    //modifyUsdPrim(); // does not correctly write back to USD

    //"size" attr does not retain modification, but test attr does
    //No CUDA
    //modify1000PrimsWithFabric();

    //create 1000 cubes with USD, modify params via CUDA
    // modify1000UsdCubesViaCuda();


    //create 1000 quads with USD, modify params via CUDA
    //NOT WORKING: runtime errors if using a Mesh (not with a Cube)
    //modify1000UsdQuadsViaCuda();


    //basic example to add one million values via CPU
    //addOneMillionCPU();
    //addOneMillionCPU function but using CUDA instead of CPU
    //addOneMillionCuda();

    //test to edit a single attribute using CUDA on a quad mesh made in Fabric
    //do not use buckets
    //NOT WORKING
    // editSingleFabricAttributeViaCuda();

    //create Quad in Fabric, edit vert in CUDA
    //NOT WORKING
    // createQuadViaFabricAndCuda();


    // createFabricQuadsModifyViaCuda(numPrimsForExperiment);

    alterScale();

    /* GEOMETRY CREATION */

    //createQuadMeshViaFabric();
    //createQuadMeshViaUsd("/Quad", 200.f);



    return 45;
}

void modifyUsdPrim() {
    //Linker error getting UsdContext using omni::usd
    // auto context = omni::usd::UsdContext::getContext();
    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();
    omni::fabric::StageReaderWriter stageReaderWriter = UsdUtil::getFabricStageReaderWriter();
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    long id = Context::instance().getStageId();
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};

    //create a cube in USD and set its size.
    pxr::UsdPrim prim = usdStagePtr->DefinePrim(pxr::SdfPath("/TestCube"), pxr::TfToken("Cube"));
    auto sizeUsdToken = pxr::TfToken("size");
    prim.CreateAttribute(sizeUsdToken, pxr::SdfValueTypeNames->Double).Set(3.0);
    auto customAttrUsdToken = pxr::TfToken("customTestAttr");
    prim.CreateAttribute(customAttrUsdToken, pxr::SdfValueTypeNames->Double).Set(123.45);

    //prefetch it to Fabric’s cache.
    omni::fabric::Path primPath("/TestCube");
    iStageReaderWriter->prefetchPrim(usdStageId, primPath);

    //use Fabric to modify the cube’s dimensions
    auto sizeFabricToken = omni::fabric::Token("size");
    auto customAttrFabricToken = omni::fabric::Token("customTestAttr");

    double& size = *stageReaderWriter.getAttribute<double>(primPath, sizeFabricToken);
    double sizeTarget = 30;
    size = sizeTarget;

    double& customAttr = *stageReaderWriter.getAttribute<double>(primPath, customAttrFabricToken);
    double customAttrTarget = 987.654;
    customAttr = customAttrTarget;

    //write our changes back to USD.
    //NOTE: does not work
    const auto iFabricUsd = carb::getCachedInterface<omni::fabric::IFabricUsd>();
    auto fabricId = omni::fabric::FabricId();
    iFabricUsd->exportUsdPrimData(fabricId);

    //check that Fabric correctly modified the USD stage.
    pxr::UsdAttribute verifyAttr = prim.GetAttribute(customAttrUsdToken);
    double value;
    verifyAttr.Get(&value);
    if (value == customAttrTarget) {
        std::cout << "modified stage" << std::endl;
    } else {
        std::cout << "did not modify stage" << std::endl;
    }
}

void modify1000UsdPrimsWithFabric() {
    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();

    //use USD to make a thousand cubes
    const size_t cubeCount = 1000;
    auto customAttrUsdToken = pxr::TfToken("cudaTest");
    for (size_t i = 0; i != cubeCount; i++)
    {
        pxr::SdfPath path("/cube_" + std::to_string(i));
        pxr::UsdPrim prim = usdStagePtr->DefinePrim(path, pxr::TfToken("Cube"));
        prim.CreateAttribute(pxr::TfToken("size"), pxr::SdfValueTypeNames->Double).Set(3.3);
        prim.CreateAttribute(customAttrUsdToken, pxr::SdfValueTypeNames->Double).Set(17.3);
    }

    //call prefetchPrim to get the data into Fabric.
    long id = Context::instance().getStageId();
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    for (size_t i = 0; i != cubeCount; i++)
    {
        omni::fabric::Path path(("/cube_" + std::to_string(i)).c_str());
        iStageReaderWriter->prefetchPrim(usdStageId, path);
    }

    //tell Fabric which prims to change. Select all prims of type Cube.
    omni::fabric::AttrNameAndType cubeTag(
        omni::fabric::Type(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName),
        omni::fabric::Token("Cube"));

    const auto stageReaderWriterId = iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(id)});
    auto fabricReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::PrimBucketList cubeBuckets = fabricReaderWriter.findPrims({ cubeTag });

    // Fabric is free to store the 1000 cubes in as many buckets as it likes...iterate over the buckets
    int counter = 0;
    auto fabricSizeToken = omni::fabric::Token("size");
    for (size_t bucket = 0; bucket != cubeBuckets.bucketCount(); bucket++)
    {
        auto sizes = fabricReaderWriter.getAttributeArray<double>(cubeBuckets, bucket, fabricSizeToken);
        for (double& size : sizes)
        {
            size = 77.7;
        }

        auto testValues = fabricReaderWriter.getAttributeArray<double>(cubeBuckets, bucket, getCudaTestAttributeFabricToken());
        for (double& testValue : testValues)
        {
            testValue = 123.45;
        }

        counter++;
    }
    std::cout << "Modified " << counter << " prims" << std::endl;
}

void modify1000UsdCubesViaCuda() {
    const int cubeCount = 1000;

    const auto usdStagePtr = Context::instance().getStage();
    const auto cudaTestAttrUsdToken = pxr::TfToken("cudaTest");
    for (size_t i = 0; i != cubeCount; i++)
    {
        pxr::SdfPath path("/cube_" + std::to_string(i));
        pxr::UsdPrim prim = usdStagePtr->DefinePrim(path, pxr::TfToken("Cube"));
        prim.CreateAttribute(pxr::TfToken("size"), pxr::SdfValueTypeNames->Double).Set(17.3);
        prim.CreateAttribute(cudaTestAttrUsdToken, pxr::SdfValueTypeNames->Double).Set(12.3);
    }

    auto id = Context::instance().getStageId();
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    for (size_t i = 0; i != cubeCount; i++)
    {
        omni::fabric::Path path(("/cube_" + std::to_string(i)).c_str());
        // This guarantees that subsequent gets of the prim from the cache will succeed
        iStageReaderWriter->prefetchPrim(usdStageId, path);
    }

    modifyQuadsViaCuda();
}

//a way to compile without using nvrtc
//note: did not work, but might now work with DLLs correctly handled
CUfunction compileKernel(const char *kernelSource, const char *kernelName) {

    CUfunction kernel_func;
    CUmodule module;

    // Set up JIT compilation options and compile the module
    // CUjit_option jitOptions[] = { //NOLINT
    //     CU_JIT_TARGET_FROM_CUCONTEXT,
    //     CU_JIT_OPTIMIZATION_LEVEL,
    // };

    // void *jitOptVals[] = { //NOLINT
    //     nullptr, // Target is picked from CUcontext
    //     (void *)3 // Optimization level 3
    // };

    const unsigned int jitNumOptions = 2;
    CUjit_option jitOptions[jitNumOptions]; //NOLINT
    void*        jitOptVals[jitNumOptions]; //NOLINT

    // Set up JIT compilation options to print verbose log
    jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    int jitLogBufferSize = 1024;
    jitOptVals[0] = (void *)(size_t)jitLogBufferSize; //NOLINT

    char* jitLogBuffer = new char[jitLogBufferSize];
    jitOptions[1] = CU_JIT_INFO_LOG_BUFFER; //NOLINT
    jitOptVals[1] = jitLogBuffer; //NOLINT

    // Compile the module
    CUresult result = cuModuleLoadDataEx(&module, kernelSource, 2, jitOptions, jitOptVals);
    //attempt to use cuModuleLoadData instead of cuModuleLoadDataEx to see if it solves the CUDA_ERROR_INVALID_IMAGE (200) issue. (It did not.)
    //CUresult result = cuModuleLoadData(&module, kernelSource);

    std::cout << "JIT compilation log:\n" << jitLogBuffer << std::endl;
    delete[] jitLogBuffer;

    if (result != CUDA_SUCCESS) {
        const char *error;
        cuGetErrorString(result, &error);
        std::cerr << "Could not compile module: " << error << std::endl;
        return nullptr;
    }

    // Get kernel function
    result = cuModuleGetFunction(&kernel_func, module, kernelName);
    if (result != CUDA_SUCCESS) {
        const char *error;
        cuGetErrorString(result, &error);
        std::cerr << "Could not get kernel function: " << error << std::endl;
        return nullptr;
    }

    return kernel_func;
}


CUfunction compileKernel2(const char *kernelSource, const char *kernelName) {
    nvrtcProgram prog;
    nvrtcResult result = nvrtcCreateProgram(&prog, kernelSource, kernelName, 0, nullptr, nullptr);
    if (result != NVRTC_SUCCESS) {
        std::cout << "Failed to create the program." << std::endl;
    }

    // Compile the program
    auto compileResult = nvrtcCompileProgram(prog, 0, nullptr);
    if (compileResult != NVRTC_SUCCESS) {
        std::cout << "Failed to compile the program." << std::endl;
    }

    // Get compilation log
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    char* log = new char[logSize];
    nvrtcGetProgramLog(prog, log);
    std::cout << "Compilation log: \n" << log << std::endl;

    // Get the PTX code
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char *ptx = new char[ptxSize];
    result = nvrtcGetPTX(prog, ptx);
    if (result != NVRTC_SUCCESS) {
        std::cout << "Failed to get the PTX code." << std::endl;
    }

    // Load the PTX code into a CUDA module
    CUmodule module;
    CUresult moduleLoadResult = cuModuleLoadData(&module, ptx);
    if (moduleLoadResult != CUDA_SUCCESS) {
        std::cout << "Failed to load the module." << std::endl;
    }

    // Get the kernel function from the module
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, kernelName);

    delete[] log;
    delete[] ptx;

    return kernel;
}

bool checkCudaCompatibility() {
    int runtimeVer, driverVer;

    // Get CUDA Runtime version
    if (cudaRuntimeGetVersion(&runtimeVer) != cudaSuccess) {
        std::cerr << "Failed to get CUDA Runtime version" << std::endl;
        return false;
    }

    // Get CUDA driver version
    if (cudaDriverGetVersion(&driverVer) != cudaSuccess) {
        std::cerr << "Failed to get CUDA driver version" << std::endl;
        return false;
    }

    std::cout << "CUDA Runtime Version: " << runtimeVer / 1000 << "." << (runtimeVer % 100) / 10 << std::endl;
    std::cout << "CUDA Driver Version: " << driverVer / 1000 << "." << (driverVer % 100) / 10 << std::endl;

    // Check compatibility
    if (runtimeVer > driverVer) {
        std::cerr << "CUDA Toolkit version is greater than the driver version. "
                  << "Please update the CUDA driver." << std::endl;
        return false;
    }

    return true;
}

void createQuadMeshViaFabric() {
    const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = Context::instance().getStageId();

    const auto stageReaderWriterId =
        iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(usdStageId)});
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    omni::fabric::Path fabricPath = omni::fabric::Path("/fabricMeshQuad");
    stageReaderWriter.createPrim(fabricPath);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.createAttributes(fabricPath);

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexCounts, 2);
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexIndices, 6);
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::points, 4);

    auto pointsFabric = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    float extentScalar = 50;
    pointsFabric[0] = pxr::GfVec3f(-extentScalar, -extentScalar, 0);
    pointsFabric[1] = pxr::GfVec3f(-extentScalar, extentScalar, 0);
    pointsFabric[2] = pxr::GfVec3f(extentScalar, extentScalar, 0);
    pointsFabric[3] = pxr::GfVec3f(extentScalar, -extentScalar, 0);

    auto faceVertexCountsFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexCounts);
    faceVertexCountsFabric[0] = 3;
    faceVertexCountsFabric[1] = 3;

    auto faceVertexIndicesFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexIndices);
    faceVertexIndicesFabric[0] = 0;
    faceVertexIndicesFabric[1] = 1;
    faceVertexIndicesFabric[2] = 2;
    faceVertexIndicesFabric[3] = 0;
    faceVertexIndicesFabric[4] = 2;
    faceVertexIndicesFabric[5] = 3;

    auto extent = pxr::GfRange3d(pxr::GfVec3d(-extentScalar, -extentScalar, 0), pxr::GfVec3d(extentScalar, extentScalar, 0));
    auto extentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::extent);
    *extentFabric = extent;

    auto worldExtentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::_worldExtent);
    *worldExtentFabric = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));

    auto worldVisibilityFabric = stageReaderWriter.getAttributeWr<bool>(fabricPath, FabricTokens::_worldVisibility);
    *worldVisibilityFabric = true;

    auto worldPositionFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3d>(fabricPath, FabricTokens::_worldPosition);
    *worldPositionFabric = pxr::GfVec3d(0, 0, 0);

    auto worldOrientationFabric = stageReaderWriter.getAttributeWr<pxr::GfQuatf>(fabricPath, FabricTokens::_worldOrientation);
    *worldOrientationFabric = pxr::GfQuatf(1.f, 0, 0, 0);

    auto worldScaleFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::_worldScale);
    *worldScaleFabric = pxr::GfVec3f(1.f, 1.f, 1.f);
}

void createQuadViaFabricAndCuda() {
    const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = Context::instance().getStageId();

    const auto stageReaderWriterId =
        iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(usdStageId)});
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    omni::fabric::Path fabricPath = omni::fabric::Path("/fabricMeshQuad");
    stageReaderWriter.createPrim(fabricPath);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.createAttributes(fabricPath);

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexCounts, 2);
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexIndices, 6);
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::points, 4);

    stageReaderWriter.createAttribute(fabricPath, getCudaTestAttributeFabricToken(), cudaTestAttributeFabricType);
    auto customAttrWriter = stageReaderWriter.getAttribute<float>(fabricPath, getCudaTestAttributeFabricToken());
    *customAttrWriter = 17.3f;

    auto pointsFabric = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    float extentScalar = 50;
    pointsFabric[0] = pxr::GfVec3f(-extentScalar, -extentScalar, 0);
    pointsFabric[1] = pxr::GfVec3f(-extentScalar, extentScalar, 0);
    pointsFabric[2] = pxr::GfVec3f(extentScalar, extentScalar, 0);
    pointsFabric[3] = pxr::GfVec3f(extentScalar, -extentScalar, 0);

    auto faceVertexCountsFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexCounts);
    faceVertexCountsFabric[0] = 3;
    faceVertexCountsFabric[1] = 3;

    auto faceVertexIndicesFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexIndices);
    faceVertexIndicesFabric[0] = 0;
    faceVertexIndicesFabric[1] = 1;
    faceVertexIndicesFabric[2] = 2;
    faceVertexIndicesFabric[3] = 0;
    faceVertexIndicesFabric[4] = 2;
    faceVertexIndicesFabric[5] = 3;

    auto extent = pxr::GfRange3d(pxr::GfVec3d(-extentScalar, -extentScalar, 0), pxr::GfVec3d(extentScalar, extentScalar, 0));
    auto extentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::extent);
    *extentFabric = extent;

    auto worldExtentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::_worldExtent);
    *worldExtentFabric = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));

    auto worldVisibilityFabric = stageReaderWriter.getAttributeWr<bool>(fabricPath, FabricTokens::_worldVisibility);
    *worldVisibilityFabric = true;

    auto worldPositionFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3d>(fabricPath, FabricTokens::_worldPosition);
    *worldPositionFabric = pxr::GfVec3d(0, 0, 0);

    auto worldOrientationFabric = stageReaderWriter.getAttributeWr<pxr::GfQuatf>(fabricPath, FabricTokens::_worldOrientation);
    *worldOrientationFabric = pxr::GfQuatf(1.f, 0, 0, 0);

    auto worldScaleFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::_worldScale);
    *worldScaleFabric = pxr::GfVec3f(1.f, 1.f, 1.f);


    // modify with CUDA
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
    }

    int major, minor;
    result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not get CUDA major version." << std::endl;
    }
    result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not get CUDA minor version." << std::endl;
    }

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
    }

    //scale with CUDA to be oblong
    //CUDA via CUDA_JIT and string
    //gets include error
    // const char* kernelCode = R"(
    // #include <pxr/base/gf/vec3f.h>

    // extern "C" __global__
    // void addTenToXComponentKernel(pxr::GfVec3f* data, size_t size)
    // {
    //     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //     if (idx < size) {
    //         data[idx].Set(data[idx].GetX() + 10.0f, data[idx].GetY(), data[idx].GetZ());
    //     }
    // }
    // )";

    const char* addToXKernelCode = R"(
    struct Triplet
    {
        float x;
        float y;
        float z;
    };

    extern "C" __global__
    void addTenToXComponentKernel(Triplet* data, size_t size)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) {
            data[idx].x += 10.0f;
            printf("Element %zu: x = %f\n", idx, data[idx].x);
        }
    }
    )";

    // CUfunction kernel = compileKernel(scaleCubes, "scaleCubes");
    CUfunction kernel = compileKernel2(addToXKernelCode, "addTenToXComponentKernel");

    struct Triplet
    {
        float x;
        float y;
        float z;
    };



    // // auto pointsFabric = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    // //auto pointsFabricGpu = stageReaderWriter.getAttributeWrGpu<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    // auto pointsFabricGpu = stageReaderWriter.getAttributeGpu<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    // auto* directData = pointsFabricGpu->data();
    // //direct float data access of GfVec3f
    // auto* triplets = reinterpret_cast<Triplet*>(directData);
    // size_t elemCount = pointsFabric.size();
    // void *args[] = { &triplets, &elemCount }; //NOLINT
    // int blockSize, minGridSize;
    // cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, nullptr, 0, 0);
    // auto err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
    // // REQUIRE(!err);
    // if (err) {
    //     std::cout << "error" << std::endl;
    // }

    //omni::fabric::AttrNameAndType quadTag(omni::fabric::BaseDataType::eToken, FabricTokens::points);
    omni::fabric::AttrNameAndType quadMeshTag(omni::fabric::Type(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName), omni::fabric::Token("Mesh"));
    omni::fabric::PrimBucketList quadBuckets = stageReaderWriter.findPrims({ quadMeshTag });
    auto bucketCount = quadBuckets.bucketCount();
    printf("Found %llu buckets\n", bucketCount);

    // omni::fabric::PrimBucketList quadBuckets = stageReaderWriter.findPrims({ quadTag });

    // //iterate over buckets but pass the vector for the whole bucket to the GPU.
    for (size_t bucket = 0; bucket != quadBuckets.bucketCount(); bucket++)
    {

        // Step 2: Get the device pointer to the data
        auto pointsD = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f>(quadBuckets, bucket, FabricTokens::points);
        pxr::GfVec3f* ptr = pointsD.data();
        //direct data access
        auto directData = ptr->data();
        auto* triplets = reinterpret_cast<Triplet*>(directData);
        size_t elemCount = pointsD.size();
        //size_t dataSize = elemCount * 3 * sizeof(float); // Assuming each point has x, y, and z components
        void *args[] = { &triplets, &elemCount }; //NOLINT
        int blockSize, minGridSize;
        cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, nullptr, 0, 0);
        auto err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        // REQUIRE(!err);
        if (err) {
            std::cout << "error" << std::endl;
        }
    }

    result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }
}

void addOneMillionCPU() {
    int N = 1<<20;

    auto* a = new float[N];
    auto* b = new float[N];

    for (int i = 0; i < 1000000; i++) {
        a[i] = 1.f;
        b[i] = 2.f;
    }

    addArrays(N, a, b);

    auto maxError = 0.f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, b[i] - 3.f);
    }
    std::cout << "max error: " << maxError << std::endl;

    cudaFree(a);
    cudaFree(b);
}

void addOneMillionCuda() {

    const char *addKernelCode = R"(
    extern "C" __global__
    void add(int n, float *x, float *y)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = index; i < n; i += stride) {
            y[i] = x[i] + y[i];
            // printf("threadIdx.x: %i, blockIdx.x: %i, blockDim.x: %i, gridDim.x: %i, y[%i]: %f\n", threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, index, y[i]);
        }
    }
    )";

  int N = 1<<20;
  N++;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  nvrtcProgram prog;
  nvrtcCreateProgram(&prog, addKernelCode, "add_program", 0, nullptr, nullptr);

  // Compile the program
  nvrtcResult res = nvrtcCompileProgram(prog, 0, nullptr);
  if (res != NVRTC_SUCCESS) {
    std::cout << "error compiling NVRTC program" << std::endl;
  }

  // Get the PTX (assembly code for the GPU) from the compilation
  size_t ptxSize;
  nvrtcGetPTXSize(prog, &ptxSize);
  char* ptx = new char[ptxSize];
  nvrtcGetPTX(prog, ptx);

  // Load the generated PTX and get a handle to the kernel.
  CUmodule module;
  CUfunction function;
  cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
  cuModuleGetFunction(&function, module, "add");

  // Set kernel parameters and launch the kernel.
  void *args[] = { &N, &x, &y }; //NOLINT
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  cuLaunchKernel(function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

}

__global__ void addArrays(int n, float* x, float* y) {
    for (auto i = 0; i < n; i++) {
        y[i] += x[i];
    }
}

void editSingleFabricAttributeViaCuda() {
   const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = Context::instance().getStageId();

    const auto stageReaderWriterId =
        iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(usdStageId)});
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    omni::fabric::Path fabricPath = omni::fabric::Path("/fabricMeshQuad");
    stageReaderWriter.createPrim(fabricPath);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.createAttributes(fabricPath);

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexCounts, 2);
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexIndices, 6);
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::points, 4);

    stageReaderWriter.createAttribute(fabricPath, getCudaTestAttributeFabricToken() , omni::fabric::BaseDataType::eFloat);
    auto customAttrWriter = stageReaderWriter.getAttribute<float>(fabricPath, getCudaTestAttributeFabricToken());
    *customAttrWriter = 12.3f;

    auto pointsFabric = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    float extentScalar = 50;
    pointsFabric[0] = pxr::GfVec3f(-extentScalar, -extentScalar, 0);
    pointsFabric[1] = pxr::GfVec3f(-extentScalar, extentScalar, 0);
    pointsFabric[2] = pxr::GfVec3f(extentScalar, extentScalar, 0);
    pointsFabric[3] = pxr::GfVec3f(extentScalar, -extentScalar, 0);

    auto faceVertexCountsFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexCounts);
    faceVertexCountsFabric[0] = 3;
    faceVertexCountsFabric[1] = 3;

    auto faceVertexIndicesFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexIndices);
    faceVertexIndicesFabric[0] = 0;
    faceVertexIndicesFabric[1] = 1;
    faceVertexIndicesFabric[2] = 2;
    faceVertexIndicesFabric[3] = 0;
    faceVertexIndicesFabric[4] = 2;
    faceVertexIndicesFabric[5] = 3;

    auto extent = pxr::GfRange3d(pxr::GfVec3d(-extentScalar, -extentScalar, 0), pxr::GfVec3d(extentScalar, extentScalar, 0));
    auto extentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::extent);
    *extentFabric = extent;

    auto worldExtentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::_worldExtent);
    *worldExtentFabric = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));

    auto worldVisibilityFabric = stageReaderWriter.getAttributeWr<bool>(fabricPath, FabricTokens::_worldVisibility);
    *worldVisibilityFabric = true;

    auto worldPositionFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3d>(fabricPath, FabricTokens::_worldPosition);
    *worldPositionFabric = pxr::GfVec3d(0, 0, 0);

    auto worldOrientationFabric = stageReaderWriter.getAttributeWr<pxr::GfQuatf>(fabricPath, FabricTokens::_worldOrientation);
    *worldOrientationFabric = pxr::GfQuatf(1.f, 0, 0, 0);

    auto worldScaleFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::_worldScale);
    *worldScaleFabric = pxr::GfVec3f(1.f, 1.f, 1.f);

    //select and bucket the Cube prims
    // omni::fabric::AttrNameAndType quadTag(
    //     omni::fabric::Type(omni::fabric::BaseDataType::eFloat, 1, 0, omni::fabric::AttributeRole::ePrimTypeName),
    //     omni::fabric::Token("cudaTest"));
    omni::fabric::AttrNameAndType quadMeshTag(omni::fabric::Type(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName), omni::fabric::Token("Mesh"));
    omni::fabric::PrimBucketList cubeBuckets = stageReaderWriter.findPrims({ quadMeshTag });

    auto isCudaCompatible = checkCudaCompatibility();
    if (!isCudaCompatible) {
        std::cout << "error: CUDA drives and toolkit versions are not compatible." << std::endl;
    }

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
    }

    int major, minor;
    result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not get CUDA major version." << std::endl;
    }
    result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not get CUDA minor version." << std::endl;
    }

    std::cout << "Compute capability: " << major << "." << minor << std::endl;

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
    }

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, kernelCode, "changeValue", 0, nullptr, nullptr);

    // Compile the program
    nvrtcResult res = nvrtcCompileProgram(prog, 0, nullptr);
    if (res != NVRTC_SUCCESS) {
        std::cout << "error compiling NVRTC program" << std::endl;
        return;
    }

    // Get the PTX (assembly code for the GPU) from the compilation
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    // Load the generated PTX and get a handle to the kernel.
    CUmodule module;
    CUfunction function;
    cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    cuModuleGetFunction(&function, module, "changeValue");

    auto bucketCount = cubeBuckets.bucketCount();
    printf("Num buckets: %llu", bucketCount);

    //iterate over buckets but pass the vector for the whole bucket to the GPU.
    for (size_t bucket = 0; bucket != cubeBuckets.bucketCount(); bucket++)
    {
        gsl::span<double> sizesD = stageReaderWriter.getAttributeArrayGpu<double>(cubeBuckets, bucket, getCudaTestAttributeFabricToken());

        double* ptr = sizesD.data();
        size_t elemCount = sizesD.size();
        void *args[] = { &ptr, &elemCount }; //NOLINT
        int blockSize, minGridSize;
        cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, nullptr, 0, 0);
        //CUresult err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, 0);
        auto err = cuLaunchKernel(function, minGridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        // REQUIRE(!err);
        if (err) {
            std::cout << "error" << std::endl;
        }
    }

    result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }
}


void createQuadsViaFabric(int numQuads) {
    const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    const auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(cesium::omniverse::Context::instance().getStageId())};
    const auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    const std::string primPathStub{"/quadMesh_"};

    for (int i = 0; i < numQuads; i++) {
        const auto fabricPath = omni::fabric::Path((primPathStub + std::to_string(i)).c_str());
        stageReaderWriter.createPrim(fabricPath);

        FabricAttributesBuilder attributes;
        attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
        attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
        attributes.addAttribute(FabricTypes::points, FabricTokens::points);
        attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
        attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
        attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
        attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
        // attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
        // attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
        // attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
        attributes.createAttributes(fabricPath);

        stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::points, 4);
        auto pointsFabric = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
        auto extentScalar = glm::linearRand(10.f, 200.f);
        float centerBounds = 1000.f;
        auto center = pxr::GfVec3f{
            glm::linearRand(-centerBounds, centerBounds),
            glm::linearRand(-centerBounds, centerBounds),
            glm::linearRand(-centerBounds, centerBounds)
        };
        pointsFabric[0] = pxr::GfVec3f(-extentScalar, -extentScalar, 0) + center;
        pointsFabric[1] = pxr::GfVec3f(-extentScalar, extentScalar, 0) + center;
        pointsFabric[2] = pxr::GfVec3f(extentScalar, extentScalar, 0) + center;
        pointsFabric[3] = pxr::GfVec3f(extentScalar, -extentScalar, 0) + center;

        stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexCounts, 2);
        auto faceVertexCountsFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexCounts);
        faceVertexCountsFabric[0] = 3;
        faceVertexCountsFabric[1] = 3;

        stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexIndices, 6);
        auto faceVertexIndicesFabric = stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexIndices);
        faceVertexIndicesFabric[0] = 0;
        faceVertexIndicesFabric[1] = 1;
        faceVertexIndicesFabric[2] = 2;
        faceVertexIndicesFabric[3] = 0;
        faceVertexIndicesFabric[4] = 2;
        faceVertexIndicesFabric[5] = 3;

        auto extent = pxr::GfRange3d(pxr::GfVec3d(-extentScalar, -extentScalar, 0), pxr::GfVec3d(extentScalar, extentScalar, 0));
        auto extentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::extent);
        *extentFabric = extent;

        auto worldExtentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::_worldExtent);
        *worldExtentFabric = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));

        auto worldVisibilityFabric = stageReaderWriter.getAttributeWr<bool>(fabricPath, FabricTokens::_worldVisibility);
        *worldVisibilityFabric = true;

        // auto worldPositionFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3d>(fabricPath, FabricTokens::_worldPosition);
        // *worldPositionFabric = pxr::GfVec3d(0, 0, 0);

        // auto worldOrientationFabric = stageReaderWriter.getAttributeWr<pxr::GfQuatf>(fabricPath, FabricTokens::_worldOrientation);
        // *worldOrientationFabric = pxr::GfQuatf(1.f, 0, 0, 0);

        // auto worldScaleFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::_worldScale);
        // *worldScaleFabric = pxr::GfVec3f(1.f, 1.f, 1.f);

        //create a custom attribute for testing
        stageReaderWriter.createAttribute(fabricPath, getCudaTestAttributeFabricToken(), cudaTestAttributeFabricType);
        auto testAttribute = stageReaderWriter.getAttributeWr<double>(fabricPath, getCudaTestAttributeFabricToken());
        *testAttribute = 123.45;
    }
}

void modifyQuadsViaCuda() {

    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    omni::fabric::AttrNameAndType quadTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
    omni::fabric::PrimBucketList buckets = stageReaderWriter.findPrims({quadTag});

    if (buckets.bucketCount() == 0 ) {
        std::cout << "No prims found, returning" << std::endl;
    }

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
    }

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
    }

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, kernelCode, "changeValue", 0, nullptr, nullptr);

    // Compile the program
    nvrtcResult res = nvrtcCompileProgram(prog, 0, nullptr);
    if (res != NVRTC_SUCCESS) {
        std::cout << "error compiling NVRTC program" << std::endl;
        return;
    }

    // Get the PTX (assembly code for the GPU) from the compilation
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    // Load the generated PTX and get a handle to the kernel.
    CUmodule module;
    CUfunction function;
    cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    cuModuleGetFunction(&function, module, "changeValue");

    auto bucketCount = buckets.bucketCount();
    printf("Num buckets: %llu\n", bucketCount);

    //iterate over buckets but pass the vector for the whole bucket to the GPU.
    int primCount = 0;
    for (size_t bucket = 0; bucket != buckets.bucketCount(); bucket++)
    {
        gsl::span<double> values = stageReaderWriter.getAttributeArrayGpu<double>(buckets, bucket, getCudaTestAttributeFabricToken());

        double* ptr = values.data();
        size_t elemCount = values.size();
        void *args[] = { &ptr, &elemCount }; //NOLINT
        int blockSize = 32 * 4;
        int numBlocks = (static_cast<int>(elemCount) + blockSize - 1) / blockSize;
        // alternatively, CUDA can calculate these for you
        // int blockSize, minGridSize;
        // cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, nullptr, 0, 0);
        auto err = cuLaunchKernel(function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        if (err) {
            std::cout << "error" << std::endl;
        }
        primCount += static_cast<int>(elemCount);
    }

    std::cout << "modified " << primCount << " quads" << std::endl;

    result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }

    delete[] ptx;
}

void createFabricQuadsModifyViaCuda(int numQuads) {
    createQuadsViaFabric(numQuads);
    modifyQuadsViaCuda();
}

void modify1000UsdQuadsViaCuda() {
    const size_t quadCount = 100;

    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();
    for (size_t i = 0; i != quadCount; i++)
    {
        createQuadMeshViaUsd(("/quad_" + std::to_string(i)).c_str(), 300.f);

        //NOTE: this function has a runtime error. You can define primitives (Cube, Sphere) and an Xform
        //However, a Mesh will lead to runtime errors
        pxr::SdfPath path("/quad_" + std::to_string(i));
        pxr::UsdPrim prim = usdStagePtr->DefinePrim(path, pxr::TfToken("Xform"));

    }

    //Alter cudaTest attribute on all quads with Cuda
    long id = Context::instance().getStageId();
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    for (size_t i = 0; i != quadCount; i++)
    {
        omni::fabric::Path path(("/quad_" + std::to_string(i)).c_str());
        iStageReaderWriter->prefetchPrim(usdStageId, path);
    }

    // //select and bucket the Cube prims
    // omni::fabric::AttrNameAndType meshTag(
    //     omni::fabric::Type(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName),
    //     omni::fabric::Token("meshTag"));
    // const auto stageReaderWriterId = iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(id)});
    // auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    // omni::fabric::PrimBucketList quadBuckets = stageReaderWriter.findPrims({ meshTag });

    // auto bucketCount = quadBuckets.bucketCount();
    // if (bucketCount == 0) {
    //     std::cout << "Found no prims to modify" << std::endl;
    // }
    // printf("Num buckets: %llu", bucketCount);

    // auto isCudaCompatible = checkCudaCompatibility();
    // if (!isCudaCompatible) {
    //     std::cout << "error: CUDA drives and toolkit versions are not compatible." << std::endl;
    // }

    // CUresult result = cuInit(0);
    // if (result != CUDA_SUCCESS) {
    //     std::cout << "error: CUDA did not init." << std::endl;
    // }

    // CUdevice device;
    // result = cuDeviceGet(&device, 0);
    // if (result != CUDA_SUCCESS) {
    //     std::cout << "error: CUDA did not get a device." << std::endl;
    // }

    // int major, minor;
    // result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    // if (result != CUDA_SUCCESS) {
    //     std::cout << "error: could not get CUDA major version." << std::endl;
    // }
    // result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    // if (result != CUDA_SUCCESS) {
    //     std::cout << "error: could not get CUDA minor version." << std::endl;
    // }

    // std::cout << "Compute capability: " << major << "." << minor << std::endl;

    // CUcontext context;
    // result = cuCtxCreate(&context, 0, device);
    // if (result != CUDA_SUCCESS) {
    //     std::cout << "error: could not create CUDA context." << std::endl;
    // }

    // //CUDA via CUDA_JIT and string
    // const char *kernelCode = R"(
    // extern "C" __global__
    // void changeValue(double* values, size_t count)
    // {
    //     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    //     if (count <= i) return;

    //     float oldVal = values[i];
    //     values[i] = 543.21;
    //     printf("Changed value of index %llu from %lf to %lf\n", i, oldVal, values[i]);
    // }
    // )";

    // nvrtcProgram prog;
    // nvrtcCreateProgram(&prog, kernelCode, "changeValue", 0, nullptr, nullptr);

    // // Compile the program
    // nvrtcResult res = nvrtcCompileProgram(prog, 0, nullptr);
    // if (res != NVRTC_SUCCESS) {
    //     std::cout << "error compiling NVRTC program" << std::endl;
    //     return;
    // }

    // // Get the PTX (assembly code for the GPU) from the compilation
    // size_t ptxSize;
    // nvrtcGetPTXSize(prog, &ptxSize);
    // char* ptx = new char[ptxSize];
    // nvrtcGetPTX(prog, ptx);

    // // Load the generated PTX and get a handle to the kernel.
    // CUmodule module;
    // CUfunction function;
    // cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    // cuModuleGetFunction(&function, module, "changeValue");

    // //iterate over buckets but pass the vector for the whole bucket to the GPU.
    // int alterationCount = 0;
    // for (size_t bucket = 0; bucket != quadBuckets.bucketCount(); bucket++)
    // {
    //     gsl::span<double> sizesD = stageReaderWriter.getAttributeArrayGpu<double>(quadBuckets, bucket, omni::fabric::Token("cudaTest"));

    //     double* ptr = sizesD.data();
    //     size_t elemCount = sizesD.size();
    //     void *args[] = { &ptr, &elemCount }; //NOLINT
    //     int blockSize, minGridSize;
    //     cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, nullptr, 0, 0);
    //     //CUresult err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, 0);
    //     auto err = cuLaunchKernel(function, minGridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
    //     // REQUIRE(!err);
    //     if (err) {
    //         std::cout << "error" << std::endl;
    //     }

    //     alterationCount += static_cast<int>(elemCount);
    // }
    // std::cout << "Altered " << alterationCount << " prims" << std::endl;

    // result = cuCtxDestroy(context);
    // if (result != CUDA_SUCCESS) {
    //     std::cout << "error: could not destroy CUDA context." << std::endl;
    // }
}

void createQuadMeshViaUsd(const char* pathString, float maxCenterRandomization) {
    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();
    pxr::SdfPath path(pathString);

    auto mesh = pxr::UsdGeomMesh::Define(usdStagePtr, path);

    //set points
    auto pointsAttr = mesh.GetPointsAttr();

    //if you want to know the type name
    // {
    //     pxr::SdfValueTypeName typeName = pointsAttr.GetTypeName();
    //     auto attrToken = typeName.GetAsToken();
    //     std::cout << "Type name: " << attrToken.GetString() << std::endl;
    // }

    pxr::VtArray<pxr::GfVec3f> points{4};
    float quadScalar = 50.f;
    pxr::GfVec3f center{
        glm::linearRand(-maxCenterRandomization, maxCenterRandomization),
        glm::linearRand(-maxCenterRandomization, maxCenterRandomization),
        0};
    points[0] = pxr::GfVec3f(-quadScalar, -quadScalar, 0) + center;
    points[1] = pxr::GfVec3f(-quadScalar, quadScalar, 0) + center;
    points[2] = pxr::GfVec3f(quadScalar, quadScalar, 0) + center;
    points[3] = pxr::GfVec3f(quadScalar, -quadScalar, 0) + center;
    pointsAttr.Set(points);

    auto faceVertexCountsAttr = mesh.GetFaceVertexCountsAttr();
    pxr::VtArray<int> faceVertexCounts{3, 3};
    faceVertexCountsAttr.Set(faceVertexCounts);

    auto faceVertexIndicesAttr = mesh.GetFaceVertexIndicesAttr();
    pxr::VtArray<int> faceVertexIndices{0, 1, 2, 0, 2, 3};
    faceVertexIndicesAttr.Set(faceVertexIndices);

    auto extentAttr = mesh.GetExtentAttr();
    pxr::VtArray<pxr::GfVec3f> extent{2};
    extent[0] = pxr::GfVec3f{-quadScalar, -quadScalar, 0};
    extent[1] = pxr::GfVec3f{quadScalar, quadScalar, 0};
    extentAttr.Set(extent);

    auto customAttrUsdToken = pxr::TfToken("cudaTest");
    //can only set a custom attr on the prim, not on an object defined by the USD schema
    auto prim = mesh.GetPrim();
    prim.CreateAttribute(customAttrUsdToken, pxr::SdfValueTypeNames->Double).Set(12.3);
}

void alterScale() {
    auto usdStagePtr = Context::instance().getStage();

    const size_t cubeCount = 10;
    auto customAttrUsdToken = pxr::TfToken("cudaTest");
    for (size_t i = 0; i != cubeCount; i++)
    {
        pxr::SdfPath path("/cube_" + std::to_string(i));

        //parenting to an Xform could work
        // pxr::UsdGeomXform xform = pxr::UsdGeomXform::Define(usdStagePtr, path);
        // pxr::UsdPrim prim = usdStagePtr->DefinePrim(xform.GetPath().AppendChild(pxr::TfToken("CubePrim")), pxr::TfToken("Cube"));

        pxr::UsdPrim prim = usdStagePtr->DefinePrim(path, pxr::TfToken("Cube"));
        if (prim.IsA<pxr::UsdGeomXformable>()) {
            pxr::UsdGeomXformable xformable(prim);
            // Add an xformOp to the Xformable prim to define the transform
            xformable.AddTranslateOp().Set(pxr::GfVec3d(3. * static_cast<double>(i), 0, 0));
            prim.CreateAttribute(customAttrUsdToken, pxr::SdfValueTypeNames->Double).Set(123.45);
        }

        //leads to error when moving in the editor
        // prim.CreateAttribute(pxr::TfToken("size"), pxr::SdfValueTypeNames->Double).Set(3.3);
        // // prim.CreateAttribute(pxr::TfToken("xformOp:scale"), pxr::SdfValueTypeNames->Point3f).Set(pxr::GfVec3f(2.f, 2.f, 2.f));
        // prim.CreateAttribute(pxr::TfToken("xformOp:translate"), pxr::SdfValueTypeNames->Point3f).Set(pxr::GfVec3f(static_cast<float>(i * 5), 0.f, 0.f));
    }


    //call prefetchPrim to get the data into Fabric.
    long id = Context::instance().getStageId();
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    for (size_t i = 0; i != cubeCount; i++)
    {
        omni::fabric::Path path(("/cube_" + std::to_string(i)).c_str());
        iStageReaderWriter->prefetchPrim(usdStageId, path);
    }

    //get all USD Cubes
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    auto ptn = omni::fabric::Type(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName);
    auto ct = omni::fabric::Token("Cube");
    omni::fabric::AttrNameAndType ant(ptn, ct);
    auto bucketList = stageReaderWriter.findPrims({ant});

    // edit translations
    auto token = omni::fabric::Token("xformOp:translate");
    auto numBuckets = bucketList.bucketCount();
    const float scaleMin = 0.f;
    const float scaleMax = 3.f;
    for (size_t bucketNum = 0; bucketNum < numBuckets; bucketNum++) {
        gsl::span<pxr::GfVec3d> values = stageReaderWriter.getAttributeArray<pxr::GfVec3d>(bucketList, bucketNum, token);
        auto numElements = values.size();
        for (unsigned long long i = 0; i < numElements; i++) {
            double xVal = values[i].data()[0];
            values[i].Set(xVal, glm::linearRand(scaleMin, scaleMax), static_cast<double>(i));
        }
    }

    for (size_t bucketNum = 0; bucketNum < numBuckets; bucketNum++) {
        gsl::span<double> values = stageReaderWriter.getAttributeArray<double>(bucketList, bucketNum, getCudaTestAttributeFabricToken());
        const auto numElements = values.size();
        for (unsigned long long i = 0; i < numElements; i++) {
            values[i] = 543.21;
        }
    }
}

} // namespace cesium::omniverse::FabricProceduralGeometry


