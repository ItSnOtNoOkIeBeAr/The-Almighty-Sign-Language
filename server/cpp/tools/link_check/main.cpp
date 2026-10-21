// Phase 1 gate: proves CUDA runtime, nvJPEG and TensorRT all link and run, and prints versions.
#include <cstdio>
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <NvInfer.h>

#define CK(call)                                                                              \
    do {                                                                                      \
        cudaError_t e_ = (call);                                                              \
        if (e_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
            return 1;                                                                         \
        }                                                                                     \
    } while (0)

int main() {
    int drv = 0, rt = 0;
    CK(cudaDriverGetVersion(&drv));
    CK(cudaRuntimeGetVersion(&rt));
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("GPU: %s (sm_%d%d), %.1f GB\n", p.name, p.major, p.minor, p.totalGlobalMem / 1073741824.0);
    std::printf("CUDA driver %d.%d, runtime %d.%d\n", drv / 1000, (drv % 1000) / 10, rt / 1000, (rt % 1000) / 10);

    int nvMaj = 0, nvMin = 0, nvPatch = 0;
    if (nvjpegGetProperty(MAJOR_VERSION, &nvMaj) || nvjpegGetProperty(MINOR_VERSION, &nvMin) ||
        nvjpegGetProperty(PATCH_LEVEL, &nvPatch)) {
        std::fprintf(stderr, "nvjpegGetProperty failed\n");
        return 1;
    }
    nvjpegHandle_t h{};
    if (nvjpegCreateSimple(&h) != NVJPEG_STATUS_SUCCESS) {
        std::fprintf(stderr, "nvjpegCreateSimple failed\n");
        return 1;
    }
    nvjpegDestroy(h);
    std::printf("nvJPEG %d.%d.%d (handle created)\n", nvMaj, nvMin, nvPatch);

    // getInferLibVersion is a runtime call into nvinfer_10.dll, so this proves the DLL loads.
    std::printf("TensorRT runtime %d (headers %d.%d.%d.%d)\n", getInferLibVersion(), NV_TENSORRT_MAJOR,
                NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD);
    return 0;
}
