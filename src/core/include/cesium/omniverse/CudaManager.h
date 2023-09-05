#pragma once

#include "cesium/omniverse/CudaKernels.h"
#include "cesium/omniverse/CudaManager.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/Tokens.h"

#include <cuda/include/cuda.h>
#include <cuda/include/cuda_runtime.h>
#include <cuda/include/nvrtc.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/fabric/IFabric.h>
#include <omni/fabric/SimStageWithHistory.h>
// #include <omni/fabric/Type.h>
#include <omni/fabric/FabricTypes.h>

#include <any>
#include <gsl/span>
#include <iostream> // TODO: temporary
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>



namespace cesium::omniverse {
class CudaManager;
class CudaRunner;
struct CudaKernel;
struct CudaKernelArgs;

enum CudaKernelType { HELLO_WORLD, CREATE_VOXELS, PRINT_POINTS, LOOKAT_QUADS };
enum CudaUpdateType { ONCE, ON_UPDATE_FRAME };

const omni::fabric::Type tileIdFabricType(omni::fabric::BaseDataType::eDouble, 1, 0, omni::fabric::AttributeRole::eNone);

struct CudaKernelArgs {
    std::unordered_map<std::string, std::any> args;
};

struct CudaKernel {
    nvrtcProgram program;
    char* ptx;
    CUmodule module;
    CUfunction function;
};

struct quad {
    float3 lowerLeft, upperLeft, upperRight, lowerRight;
};

class CudaManager {
  public:

    static CudaManager& getInstance() { // TODO: make like Context::instance().doThing()
        static CudaManager instance;
        return instance;
    }

    void onUpdateFrame();
    // void addRunner(CudaRunner& cudaRunner);
    void removeRunner(int64_t tileId);
    [[nodiscard]] const char* getKernelCode(CudaKernelType kernelType) const;
    [[nodiscard]] const char* getFunctionName(CudaKernelType kernelType) const;
    omni::fabric::Type getTileTokenType() {
      return {omni::fabric::BaseDataType::eInt64, 1, 0, omni::fabric::AttributeRole::eNone};
    }
    void createRunner(CudaKernelType cudaKernelType, CudaUpdateType cudaUpdateType, int64_t tileId, CudaKernelArgs kernelArgs, int numberOfElements);
    omni::fabric::Token getTileToken(int64_t tileId);

  private:
    CUdevice _device;
    CUcontext _context;
    bool _initialized = false;
    std::unordered_map<CudaUpdateType, std::unordered_map<int64_t, std::unique_ptr<CudaRunner>>> _runnersByUpdateType;
    std::unordered_map<CudaKernelType, CudaKernel> _kernels;
    int _blockSize, _numBlocks;

    void compileKernel(CudaKernelType kernelType);
    void runAllRunners();
    void initialize();
    void runRunner(CudaRunner& runner);
    std::unordered_map<int64_t, omni::fabric::Token> _tileTokens;
    // void packKernelArgs(CudaRunner& runner);
};

class CudaRunner {
  public:
    CudaKernelType kernelType;
    int elementCount;
    CudaKernelArgs kernelArgs;
    omni::fabric::Token tileIdToken;
    omni::fabric::PrimBucketList primBucketList;
    std::unordered_map<size_t, quad*>quadBucketMap; // TODO: gsl::span?

    CudaRunner() = delete;
    CudaRunner(
        CudaKernelType cudaKernelType,
        CudaUpdateType cudaUpdateType,
        int64_t tileId,
        CudaKernelArgs args,
        int elementCountArg)
        : kernelType(cudaKernelType)
        , elementCount(elementCountArg)
        , kernelArgs(std::move(args))
        , primBucketList(initializePrimBucketList(tileId))
        , _tileId(tileId)
        , _updateType(cudaUpdateType) {
          // std::cout << "in constructor for tile " << std::to_string(_tileId) << std::endl;
          // auto stageReaderWriter = Context::instance().getFabricStageReaderWriter();

          // for (size_t bucketNum = 0; bucketNum  < primBucketList.size(); bucketNum++) {
          //   std::cout << "  in bucket " << std::to_string(bucketNum) << std::endl;

            // accessing the bucket is corrupting the points buffer
            // (but not if you use stageReaderWriter.getAttributeArray)
            // auto positions = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f*>(primBucketList, bucketNum, FabricTokens::points);
            // bool isNull = positions.data() == nullptr;
            // if (isNull) {
            //   std::cout << " positions.data() was accessed, is null" << std::endl;
            // } else {
            //   std::cout << " positions.data() was accessed, is not null" << std::endl;
            // }

            // auto positions = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f*>(primBucketList, bucketNum, FabricTokens::points);
            // bool isNull = positions.data() == nullptr;
            // if (isNull) {
            //   std::cout << " positions.data() was accessed, is null" << std::endl;
            // } else {
            //   std::cout << " positions.data() was accessed, is not null" << std::endl;
            // }


            // auto quadsPtr = reinterpret_cast<quad*>(positions.data());
            // quadBucketMap[bucketNum] = quadsPtr;
          // }
        };
    // ~CudaRunner() {
    //   std::cout << "TODO: delete tileIdToken " << std::endl;
    //   // delete tileIdToken;
    // }
    [[nodiscard]] int64_t getTileId() const {
        return _tileId;
    }
    [[nodiscard]] const CudaUpdateType& getUpdateType() const {
        return _updateType;
    }

  private:
    int64_t _tileId;
    CudaUpdateType _updateType;
    omni::fabric::PrimBucketList initializePrimBucketList(int64_t tileId) {
        auto stageReaderWriter = Context::instance().getFabricStageReaderWriter();
        omni::fabric::AttrNameAndType tag(
          CudaManager::getInstance().getTileTokenType(),
          CudaManager::getInstance().getTileToken(tileId)
        );

        // // DEBUG
        // auto pbl = stageReaderWriter.findPrims({tag});
        // std::cout << "num buckets: " << pbl.size() << std::endl;
        // if (pbl.size() == 0) {
        //   throw std::runtime_error("no Fabric buckets retrieved for tile " + std::to_string(tileId));
        //   // auto message = "no Fabric buckets retrieved for tile " + std::to_string(tileId);
        //   // std::cout << message << std::endl;
        // }

        return stageReaderWriter.findPrims({tag});
    }
};


} // namespace cesium::omniverse
