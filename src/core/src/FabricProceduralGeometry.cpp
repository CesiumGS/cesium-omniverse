#include "cesium/omniverse/FabricProceduralGeometry.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace cesium::omniverse::FabricProceduralGeometry {

glm::dvec3 lookatPositionHost{0.0, 0.0, 0.0};
glm::fvec3 lookatUpHost{0.0, 0.0, 0.0};
CudaRunner cudaRunner;
double elapsedTime = 0;
float quadSizeHost = 0;

const omni::fabric::Type
    billboardedAttributeFabricType(omni::fabric::BaseDataType::eBool, 1, 0, omni::fabric::AttributeRole::eNone);
omni::fabric::Token getBillboardedAttributeFabricToken() {
    static const auto billboardedFabricToken = omni::fabric::Token("billboarded");
    return billboardedFabricToken;
}

const omni::fabric::Type numQuadsFabricType(omni::fabric::BaseDataType::eInt, 1, 0, omni::fabric::AttributeRole::eNone);
omni::fabric::Token getNumQuadsAttributeFabricToken() {
    static const auto quadOrientationAttributeFabricToken = omni::fabric::Token("numQuads");
    return quadOrientationAttributeFabricToken;
}

const char* lookAtMultiquadKernelCode = R"(

__device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct mat3 {
    float3 col0;
    float3 col1;
    float3 col2;

    __device__ float3 multiply(const float3 &vec) const {
        float3 result;
        result.x = dot(make_float3(col0.x, col1.x, col2.x), vec);
        result.y = dot(make_float3(col0.y, col1.y, col2.y), vec);
        result.z = dot(make_float3(col0.z, col1.z, col2.z), vec);
        return result;
    }
};

struct quad {
    float3 lowerLeft;
    float3 upperLeft;
    float3 upperRight;
    float3 lowerRight;

    __device__ float3 getCenter() {
        return make_float3(
            (lowerLeft.x + upperLeft.x + upperRight.x + lowerRight.x) * .25f,
            (lowerLeft.y + upperLeft.y + upperRight.y + lowerRight.y) * .25f,
            (lowerLeft.z + upperLeft.z + upperRight.z + lowerRight.z) * .25f);
    }
};

__device__ float3 normalize(float3 v) {
    float normSquared = v.x * v.x + v.y * v.y + v.z * v.z;
    float inverseSqrtNorm = rsqrtf(normSquared);
    v.x *= inverseSqrtNorm;
    v.y *= inverseSqrtNorm;
    v.z *= inverseSqrtNorm;
    return v;
}

__device__ double3 normalize(double3 v) {
    double norm = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    double inverseNorm = 1.0 / norm;
    v.x *= inverseNorm;
    v.y *= inverseNorm;
    v.z *= inverseNorm;
    return v;
}

__device__ float3 cross(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float3 operator*(const float3 &a, const float &b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ bool almostEquals(float3 a, float3 b) {
    const float epsilon = 0.0000001f;
    if (abs(a.x - b.x) > epsilon) return false;
    if (abs(a.y - b.y) > epsilon) return false;
    if (abs(a.z - b.z) > epsilon) return false;

    return true;
}

__device__ double3 subtractDouble3(double3 a, double3 b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 addFloat3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

extern "C" __global__ void lookAtMultiquadKernel(quad** quads, double3* lookatPosition, float3* lookatUp, float *quadSize, int numQuads) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numQuads) return;

    int quadIndex = static_cast<int>(i);
    const float quadHalfSize = *quadSize * 0.5f;

    float3 targetUpN = *lookatUp;
    float3 quadCenter = quads[0][quadIndex].getCenter();
    double3 quadCenterD = make_double3(static_cast<double>(quadCenter.x), static_cast<double>(quadCenter.y) , static_cast<double>(quadCenter.z));
    double3 newQuadForwardDouble = subtractDouble3(*lookatPosition, quadCenterD);
    float3 newQuadForward = make_float3(
        static_cast<float>(newQuadForwardDouble.x),
        static_cast<float>(newQuadForwardDouble.y),
        static_cast<float>(newQuadForwardDouble.z));
    float3 newQuadForwardN = normalize(newQuadForward);
    float3 newQuadRightN;
    float3 newQuadUpN;

    if (almostEquals(newQuadForwardN, targetUpN)) {
        //directly beneath the camera, no op
        printf("directly beneath camera, no op. returning\n");
        return;
    } else {
        newQuadRightN = normalize(cross(newQuadForwardN, targetUpN));
        newQuadUpN = normalize(cross(newQuadRightN, newQuadForward));
    }

    mat3 translationMatrix = {newQuadRightN, newQuadUpN, newQuadForwardN};

    //untransformed quad points are assumed to be in XY plane
    float3 rotatedLL = translationMatrix.multiply(make_float3(-quadHalfSize, -quadHalfSize, 0));
    float3 rotatedUL = translationMatrix.multiply(make_float3(-quadHalfSize, quadHalfSize, 0));
    float3 rotatedUR = translationMatrix.multiply(make_float3(quadHalfSize, quadHalfSize, 0));
    float3 rotatedLR = translationMatrix.multiply(make_float3(quadHalfSize, -quadHalfSize, 0));
    float3 newQuadUL = addFloat3(rotatedUL, quadCenter);
    float3 newQuadUR = addFloat3(rotatedUR, quadCenter);
    float3 newQuadLL = addFloat3(rotatedLL, quadCenter);
    float3 newQuadLR = addFloat3(rotatedLR, quadCenter);

    quads[0][quadIndex].upperLeft = newQuadUL;
    quads[0][quadIndex].upperRight = newQuadUR;
    quads[0][quadIndex].lowerLeft = newQuadLL;
    quads[0][quadIndex].lowerRight = newQuadLR;

    // printf("Billboarding quad with index %d.\n", quadIndex);
    // printf("  Upper left x coord is %f\n", quads[0][quadIndex].upperLeft.x);
}
)";

int createPrims() {

    createMultiquadFromPtsFile("pointCloudData/simpleTest.pts", 50.0f); // run a simple test of a 3x3 point-cloud grid
    // createMultiquadFromPtsFile("pointCloudData/pump0.pts", 0.125f, 5.0f); // an example with about 175,000  points
    // createMultiquadFromPtsFile("pointCloudData/StSulpice_000000.pts", 0.125f, 5.0f); // an example with about 1,500,000  points

    // makeInitialReadCall();

    return 0;
}

int alterPrims(
    double cameraPositionX,
    double cameraPositionY,
    double cameraPositionZ,
    float cameraUpX,
    float cameraUpY,
    float cameraUpZ) {

    auto cameraPositionf = glm::fvec3(
        static_cast<float>(cameraPositionX), static_cast<float>(cameraPositionY), static_cast<float>(cameraPositionZ));

    billboardMultiQuadCuda(cameraPositionf, glm::fvec3(cameraUpX, cameraUpY, cameraUpZ));
    return 0;
}

int animatePrims(
    float deltaTime,
    double cameraPositionX,
    double cameraPositionY,
    double cameraPositionZ,
    float cameraUpX,
    float cameraUpY,
    float cameraUpZ) {

    elapsedTime += deltaTime;
    lookatPositionHost.x = cameraPositionX;
    lookatPositionHost.y = cameraPositionY;
    lookatPositionHost.z = cameraPositionZ;
    lookatUpHost.x = cameraUpX;
    lookatUpHost.y = cameraUpY;
    lookatUpHost.z = cameraUpZ;

    alterPrims(cameraPositionX, cameraPositionY, cameraPositionZ, cameraUpX, cameraUpY, cameraUpZ);

    return 0;
}

void CudaRunner::init(const char* kernelCodeDEBUG, const char* kernelFunctionName) {
    if (_initted) {
        return;
    }

    _kernelCode = kernelCodeDEBUG;
    _kernelFunctionName = kernelFunctionName;

    CUresult result;

    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
        throw std::runtime_error("ERROR");
    }

    // CUdevice device;
    result = cuDeviceGet(&_device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
        throw std::runtime_error("ERROR");
    }

    result = cuDevicePrimaryCtxRetain(&_context, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
        throw std::runtime_error("ERROR");
    }
    // auto threadId = std::this_thread::get_id();
    // std::cout << "Initial thread ID: " << threadId << std::endl;

    //print the numbers of SMs
    int numDevices;
    cuDeviceGetCount(&numDevices);
    for (int i = 0; i < numDevices; i++) {
        cuDeviceGet(&_device, i);
        int numSMs;
        cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, _device);
        std::cout << "Device " << i << " has " << numSMs << " Streaming Multiprocessors." << std::endl;
    }

    // nvrtcProgram program;
    nvrtcCreateProgram(&_program, _kernelCode, _kernelFunctionName, 0, nullptr, nullptr);

    nvrtcResult res = nvrtcCompileProgram(_program, 0, nullptr);
    if (res != NVRTC_SUCCESS) {
        std::cout << "Error compiling NVRTC program:" << std::endl;

        size_t logSize;
        nvrtcGetProgramLogSize(_program, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(_program, log);
        std::cout << "   Compilation log: \n" << log << std::endl;
        delete[] log;

        throw std::runtime_error("ERROR");
    }

    // Get the PTX (assembly code for the GPU)
    size_t ptxSize;
    nvrtcGetPTXSize(_program, &ptxSize);
    _ptx = new char[ptxSize];
    nvrtcGetPTX(_program, _ptx);

    // CUmodule module;
    // CUfunction function;
    cuModuleLoadDataEx(&_module, _ptx, 0, nullptr, nullptr);
    auto cudaRes = cuModuleGetFunction(&_function, _module, _kernelFunctionName);
    if (cudaRes != CUDA_SUCCESS) {
        const char* errName = nullptr;
        const char* errString = nullptr;
        cuGetErrorName(cudaRes, &errName);
        cuGetErrorString(cudaRes, &errString);
        std::cout << "Error getting function: " << errName << ": " << errString << std::endl;

        std::ostringstream errMsg;
        errMsg << "Error getting function: " << errName << ": " << errString;
        throw std::runtime_error(errMsg.str());
    }

    _initted = true;
}

void CudaRunner::teardown() {
    auto result = cuCtxDestroy(_context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }

    delete[] _ptx;

    // untested
    // cuModuleUnload(_module);
    // nvrtcDestroyProgram(&_program);
}

CudaRunner::~CudaRunner() {
    if (_initted)
        teardown();
}

bool CudaRunner::runKernel(void** args, size_t elemCount) {
    cuCtxSetCurrent(_context);

    int minGridSize;
    cuOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSize, _function, nullptr, 0, 0);
    _numBlocks = (static_cast<int>(elemCount) + _blockSize - 1) / _blockSize;

    auto launchResult = cuLaunchKernel(_function, _numBlocks, 1, 1, _blockSize, 1, 1, 0, nullptr, args, nullptr);
    if (launchResult) {
        const char* errName = nullptr;
        const char* errString = nullptr;

        cuGetErrorName(launchResult, &errName);
        cuGetErrorString(launchResult, &errString);

        std::cout << "Error launching kernel: " << errName << ": " << errString << std::endl;

        CUcontext currentContext;
        cuCtxGetCurrent(&currentContext);
        if (currentContext != _context) {
            std::cout << "Warning: CUDA Context has changed. Resetting." << std::endl;
            // auto threadId = std::this_thread::get_id();
            // std::cout << "Current thread ID: " << threadId << std::endl;
            cuCtxSetCurrent(_context);
        }

        return false;
    }

    //DEBUG: bypass re-launching for now.
    //retry after Context switch
    // if (launchResult) {
    //     launchResult = cuLaunchKernel(_function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
    // }

    // if (launchResult) {
    //     throw std::runtime_error("kernel failed to launch after switch to original context\n");
    // }
    return true;
}

bool firstPass = true;
void billboardMultiQuadCuda(glm::fvec3 lookatPosition, glm::fvec3 lookatUp) {

    lookatPositionHost.x = static_cast<double>(lookatPosition.x);
    lookatPositionHost.y = static_cast<double>(lookatPosition.y);
    lookatPositionHost.z = static_cast<double>(lookatPosition.z);
    lookatUpHost.x = lookatUp.x;
    lookatUpHost.y = lookatUp.y;
    lookatUpHost.z = lookatUp.z;

    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(billboardedAttributeFabricType, getBillboardedAttributeFabricToken());
    omni::fabric::PrimBucketList bucketList = stageReaderWriter.findPrims({primTag});

    cudaRunner.init(lookAtMultiquadKernelCode, "lookAtMultiquadKernel");

    auto lookatPositionDevice = allocAndCopyToDevice(&lookatPositionHost, sizeof(glm::dvec3));
    auto lookatUpDevice = allocAndCopyToDevice(&lookatUpHost, sizeof(glm::fvec3));
    auto quadSizeDevice = allocAndCopyToDevice(&quadSizeHost, sizeof(float));

    for (size_t bucketNum = 0; bucketNum != bucketList.bucketCount(); bucketNum++) {
        auto numQuadsSpan =
            stageReaderWriter.getAttributeArrayRd<int>(bucketList, bucketNum, getNumQuadsAttributeFabricToken());
        int elemCount = numQuadsSpan[0];
        if (elemCount == 0) {
            throw std::runtime_error("Fabric did not retrieve any elements");
        }

        gsl::span<pxr::GfVec3f> positions;
        if (firstPass) {
            positions =
                stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f>(bucketList, bucketNum, FabricTokens::points);
            firstPass = false;
        } else {
            positions =
                stageReaderWriter.getAttributeArrayWrGpu<pxr::GfVec3f>(bucketList, bucketNum, FabricTokens::points);
        }

        auto data = positions.data();
        void* args[] = {&data, &lookatPositionDevice, &lookatUpDevice, &quadSizeDevice, &elemCount}; // NOLINT
        auto success = cudaRunner.runKernel(args, static_cast<size_t>(elemCount));
        if (!success) {
            std::cout << "error running cudaRunner" << std::endl;
        }
    }

    freeDeviceMemory(lookatPositionDevice);
    freeDeviceMemory(lookatUpDevice);
    freeDeviceMemory(quadSizeDevice);
}

void createMultiquadFromPtsFile(const std::string& ptsFile, float quadSize, float scale) {
    quadSizeHost = quadSize;
    std::vector<pxr::GfVec3f> points;
    std::ifstream file(ptsFile);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << ptsFile << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        float x, y, z;
        if (!(ss >> x >> y >> z)) {
            std::cerr << "Error getting points from line: " << line << std::endl;
            continue;
        }
        points.emplace_back(x * scale, y * scale, z * scale);
    }

    file.close();
    std::cout << "read " << points.size() << " points" << std::endl;

    const auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    const auto usdStageId =
        omni::fabric::UsdStageId{static_cast<uint64_t>(cesium::omniverse::Context::instance().getStageId())};
    const auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    const std::string primPathStub{"/multiquadMesh"};
    auto fabricPath = omni::fabric::Path((primPathStub + std::to_string(0)).c_str());
    stageReaderWriter.createPrim(fabricPath);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::primvars_displayColor, FabricTokens::primvars_displayColor);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    // attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.createAttributes(fabricPath);

    auto numQuads = points.size();
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::points, static_cast<size_t>(numQuads * 4));
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexCounts, numQuads * 2);
    auto pointsFabric = stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::points);
    auto faceVertexCountsFabric =
        stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexCounts);
    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::faceVertexIndices, numQuads * 6);
    auto faceVertexIndicesFabric =
        stageReaderWriter.getArrayAttributeWr<int>(fabricPath, FabricTokens::faceVertexIndices);

    glm::dvec3 extentsMin{
        std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    glm::dvec3 extentsMax{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};

    size_t vertIndex = 0;
    size_t vertexCountsIndex = 0;
    size_t faceVertexIndex = 0;
    const float quadHalfSize = quadSizeHost * .5f;
    for (size_t quadNum = 0; quadNum < numQuads; quadNum++) {
        pxr::GfVec3f quadShift = points[quadNum];
        pointsFabric[vertIndex++] = pxr::GfVec3f{-quadHalfSize, -quadHalfSize, 0} + quadShift;
        pointsFabric[vertIndex++] = pxr::GfVec3f{-quadHalfSize, quadHalfSize, 0} + quadShift;
        pointsFabric[vertIndex++] = pxr::GfVec3f{quadHalfSize, quadHalfSize, 0} + quadShift;
        pointsFabric[vertIndex++] = pxr::GfVec3f{quadHalfSize, -quadHalfSize, 0} + quadShift;

        faceVertexCountsFabric[vertexCountsIndex++] = 3;
        faceVertexCountsFabric[vertexCountsIndex++] = 3;

        faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(quadNum * 4);
        faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(quadNum * 4);
        faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(quadNum * 4);
        faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(quadNum * 4);
        faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(quadNum * 4);
        faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(quadNum * 4);

        extentsMin.x = fmin(extentsMin.x, -quadHalfSize + quadShift[0]);
        extentsMin.y = fmin(extentsMin.y, -quadHalfSize + quadShift[1]);
        extentsMin.z = fmin(extentsMin.z, quadShift[2]);
        extentsMax.x = fmax(extentsMax.x, quadHalfSize + quadShift[0]);
        extentsMax.y = fmax(extentsMax.y, quadHalfSize + quadShift[1]);
        extentsMax.z = fmax(extentsMax.z, quadShift[2]);
    }

    auto extent = pxr::GfRange3d(
        pxr::GfVec3d(extentsMin[0], extentsMin[1], extentsMin[2]),
        pxr::GfVec3d(extentsMax[0], extentsMax[1], extentsMax[2]));
    auto extentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::extent);
    *extentFabric = extent;

    auto worldExtentFabric = stageReaderWriter.getAttributeWr<pxr::GfRange3d>(fabricPath, FabricTokens::_worldExtent);
    *worldExtentFabric = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));

    auto worldVisibilityFabric = stageReaderWriter.getAttributeWr<bool>(fabricPath, FabricTokens::_worldVisibility);
    *worldVisibilityFabric = true;

    // TODO: center of mass of point cloud?
    auto worldPositionFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3d>(fabricPath, FabricTokens::_worldPosition);
    *worldPositionFabric = pxr::GfVec3d(0.0, 0.0, 0.0);

    auto worldOrientationFabric =
        stageReaderWriter.getAttributeWr<pxr::GfQuatf>(fabricPath, FabricTokens::_worldOrientation);
    *worldOrientationFabric = pxr::GfQuatf(0.f, 0, 0, 0);

    // NOTE: throws write error
    // auto worldScaleFabric = stageReaderWriter.getAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::_worldScale);
    // *worldScaleFabric = pxr::GfVec3f(1.f, 1.f, 1.f);

    stageReaderWriter.setArrayAttributeSize(fabricPath, FabricTokens::primvars_displayColor, 1);
    auto displayColors =
        stageReaderWriter.getArrayAttributeWr<pxr::GfVec3f>(fabricPath, FabricTokens::primvars_displayColor);
    displayColors[0] = pxr::GfVec3f(0.8f, 0.8f, 0.8f);

    // create a custom attribute for testing
    stageReaderWriter.createAttribute(fabricPath, getBillboardedAttributeFabricToken(), billboardedAttributeFabricType);
    auto testAttribute = stageReaderWriter.getAttributeWr<bool>(fabricPath, getBillboardedAttributeFabricToken());
    *testAttribute = true;

    // record number of quads (for testing purposes)
    stageReaderWriter.createAttribute(fabricPath, getNumQuadsAttributeFabricToken(), numQuadsFabricType);
    auto numQuadsAttribute = stageReaderWriter.getAttributeWr<int>(fabricPath, getNumQuadsAttributeFabricToken());
    *numQuadsAttribute = static_cast<int>(numQuads);

    return;
}

CUdeviceptr allocAndCopyToDevice(void* hostPtr, size_t size) {
    CUdeviceptr devicePtr;
    CudaError::check(cuMemAlloc(&devicePtr, size), "cuMemAlloc");
    CudaError::check(cuMemcpyHtoD(devicePtr, hostPtr, size), "cuMemcpyHtoD");
    return devicePtr;
}

void freeDeviceMemory(CUdeviceptr devicePtr) {
    CUresult err = cuMemFree(devicePtr);
    if (err != CUDA_SUCCESS) {
        const char* errName;
        const char* errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemFree failed for address %p: %s: %s\n", (void*)devicePtr, errName, errStr);
    }
}

void makeInitialReadCall() {
    // make an initial read call to getAttributeArray
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::AttrNameAndType primTag(billboardedAttributeFabricType, getBillboardedAttributeFabricToken());
    omni::fabric::PrimBucketList bucketList = stageReaderWriter.findPrims({primTag});
    for (size_t bucketNum = 0; bucketNum != bucketList.bucketCount(); bucketNum++) {
        // Compile-time error in getAttributeArrayRdGpu:
        // WrapperImpl.h(669): error C2197: 'void (__cdecl *)(omni::fabric::ConstSpanC *,omni::fabric::StageReaderWriterId,omni::fabric::PrimBucketListId,size_t,omni::fabric::TokenC)': too many arguments for call
        // auto positions = stageReaderWriter.getAttributeArrayRdGpu<pxr::GfVec3f*>(bucketList, bucketNum, FabricTokens::points);

        auto positions =
            stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f*>(bucketList, bucketNum, FabricTokens::points);
        std::cout << positions.size() << std::endl; // trick clang
    }
}


} // namespace cesium::omniverse::FabricProceduralGeometry
