#pragma once

#include "cesium/omniverse/CudaManager.h"
#include "cesium/omniverse/CudaKernels.h"

#include <any>
#include <cuda/include/cuda.h>
#include <cuda/include/cuda_runtime.h>
#include <cuda/include/nvrtc.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/fabric/IFabric.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

//TODO: handle DLLs in application directory

namespace cesium::omniverse {
    class CudaManager;
    class CudaRunner;
    class CudaKernel;
    struct CudaKernelArgs;

    enum CudaKernelType {
        HELLO_WORLD,
        CREATE_VOXELS,
        PRINT_POINTS
    };

    enum CudaUpdateType {
        ONCE,
        ON_UPDATE
    };

    struct CudaKernelArgs {
        std::unordered_map<std::string, std::any> args;
    };

    class CudaManager{
        public:
            static CudaManager& getInstance() {
                static CudaManager instance;
                return instance;
            }

            void addRunner(CudaRunner& cudaRunner);
            // void removeRunner(std::string tileId, CudaUpdateType updateType);
            [[nodiscard]] const char* getKernelCode(CudaKernelType kernelType) const;
            [[nodiscard]] const char* getFunctionName(CudaKernelType kernelType) const;

        private:
            CUdevice _device;
            CUcontext _context;
            bool _initialized = false;
            std::unordered_map<CudaUpdateType, std::unordered_map<int64_t, CudaRunner>> _runnersByUpdateType;
            std::unordered_map<CudaKernelType, CudaKernel> _kernels;
            int _blockSize, _numBlocks;

            void onUpdate();
            void compileKernel(CudaKernelType kernelType);
            void runAllRunners();
            void initialize();
            void runRunner(CudaRunner& runner);
            void** packArgs(CudaKernelArgs cudaKernelArgs, CudaKernelType cudaKernelType);
    };

    class CudaRunner {
        //TODO: move semantics
        public:
            CudaKernelType kernelType;

            CudaRunner() {
                throw std::runtime_error("This should never be called\n");
            }
            CudaRunner(CudaKernelType cudaKernelType, CudaUpdateType updateType, int64_t tileId, CudaKernelArgs args, int elementCount) :
                kernelType(cudaKernelType), kernelArgs(std::move(args)), elementCount(elementCount), _tileId(tileId), _updateType(updateType) {};
            [[nodiscard]] int64_t getTileId() const { return _tileId; }
            CudaKernelArgs kernelArgs;
            [[nodiscard]] const CudaUpdateType& getUpdateType() const { return _updateType; }
            int elementCount;
        private:
            // omni::fabric::PrimBucketList _bucketList;
            int64_t _tileId;
            CudaUpdateType _updateType;
    };

    class CudaKernel {
        public:
            nvrtcProgram program;
            char* ptx;
            CUmodule module;
            CUfunction function;
        private:
            // const char* _kernelFunctionName;
    };
}
