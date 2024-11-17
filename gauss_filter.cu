//
// Created by sajith on 17/11/2024.
//

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif


#include <UtilNPP/Exceptions.h>
#include <UtilNPP/ImagesCPU.h>
#include <UtilNPP/ImagesNPP.h>
#include <UtilNPP/ImageIO.h>


#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <nppi.h>

#include <string>

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
           libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
           (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}


void gaussFilter(const std::string &filePath, const std::string &outputFile)
{
    try
    {
        std::cout << "Processing of " << filePath << " started." << std::endl;
        npp::ImageCPU_8u_C1 hostSrc;
        npp::loadImage(filePath, hostSrc);
        npp::ImageNPP_8u_C1 deviceSrc(hostSrc);
        const NppiSize srcSize = {(int) deviceSrc.width(), (int) deviceSrc.height()};
        const NppiPoint srcOffset = {0, 0};

        const NppiSize filterROI = {(int) deviceSrc.width(), (int) deviceSrc.height()};
        npp::ImageNPP_8u_C1 deviceDst(filterROI.width, filterROI.height);

        NPP_CHECK_NPP(nppiFilterGaussBorder_8u_C1R(deviceSrc.data(), deviceSrc.pitch(), srcSize, srcOffset,
                                                   deviceDst.data(), deviceDst.pitch(), filterROI,
                                                   NppiMaskSize::NPP_MASK_SIZE_3_X_3,
                                                   NppiBorderType::NPP_BORDER_REPLICATE));

        npp::ImageCPU_8u_C1 hostDst(deviceDst.size());
        deviceDst.copyTo(hostDst.data(), hostDst.pitch());
        npp::saveImage(outputFile, hostDst);
        std::cout << "Processing of " << filePath << " ended. Result saved to: " << outputFile << std::endl;

//        nppiFree(deviceSrc.data());
//        nppiFree(deviceDst.data());
//        nppiFree(hostSrc.data());
//        nppiFree(hostDst.data());
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
}


int main(int argc, char *argv[])
{

    if (!printfNPPinfo(argc, argv))
    {
        exit(EXIT_SUCCESS);
    }

    findCudaDevice(argc, (const char **) argv);

    return 0;
}