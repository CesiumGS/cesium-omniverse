#pragma once

#include <cuda/include/cuda.h>
#include <cuda/include/cuda_runtime.h>
#include <cuda/include/nvrtc.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/fabric/IFabric.h>
#include <unordered_map>
#include <vector>

//TODO: handle DLLs in application directory

namespace cesium::omniverse {
    class CudaRunner;
    class CudaKernel;

    enum CudaKernelType {
        CREATE_VOXELS
    };

    enum CudaUpdateType {
        ONCE,
        ON_UPDATE
    };

    class CudaManager{
        public:
            void addRunner(CudaRunner cudaRunner);
            void removeRunner(std::string tileId);
            static CudaManager* getInstance() {
                if (_instance == nullptr) _instance = nullptr;
                return _instance;
            }
        private:
            static CudaManager* _instance;
            CUdevice _device;
            CUcontext _context;
            void _onUpdate();
            void _compileKernel(CudaKernel cudaKernel);
            void _runAllCudaRunners();
            void _initialize();
            bool _initialized = false;
            std::unordered_map<CudaUpdateType, std::unordered_map<std::string, CudaRunner>> _currentRunners;
    };

    class CudaRunner {
        public:
            CudaKernelType kernelType;
            CudaRunner(CudaKernelType cudaKernelType, std::string& tileId) : kernelType(cudaKernelType), _tileId(tileId) {};
            ~CudaRunner() {
                CudaManager::getInstance()->removeRunner(_tileId);
            };
        private:
            std::string _tileId;
    };

    class CudaKernel {
        public:
            CudaKernel(CudaKernelType cudaKernelType);
        private:
            omni::fabric::PrimBucketList bucketList;
            CudaUpdateType _updateType;
            const char* _kernelFunctionName;
            CUfunction _function;
            nvrtcProgram _program;
            char* _ptx;
            CUmodule _module;
    };
}
