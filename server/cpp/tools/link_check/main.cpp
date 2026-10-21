// Phase 1 gate: proves CUDA runtime, nvJPEG and TensorRT all link and run, and prints versions.
#include <cstdio>
#include <windows.h>
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
    // Driver = the newest CUDA the installed driver supports. Runtime = the cudart DLL actually loaded.
    // They are different things and are expected to differ (driver >= runtime).
    std::printf("CUDA driver  version (cudaDriverGetVersion):  %d -> %d.%d\n", drv, drv / 1000, (drv % 1000) / 10);
    std::printf("CUDA runtime version (cudaRuntimeGetVersion): %d -> %d.%d\n", rt, rt / 1000, (rt % 1000) / 10);
    char dll[MAX_PATH] = {};
    if (HMODULE m = GetModuleHandleA("cudart64_13.dll")) GetModuleFileNameA(m, dll, MAX_PATH);
    std::printf("cudart DLL loaded from: %s\n", dll[0] ? dll : "(statically linked)");

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
