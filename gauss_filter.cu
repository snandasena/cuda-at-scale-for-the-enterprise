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
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>

namespace fs = std::filesystem;  // For filesystem operations like iterating over files

/**
 * Prints NPP library and CUDA driver information.
 * This function retrieves the version of the NPP library, CUDA driver, and runtime,
 * and checks if the system meets the minimum CUDA capability (SM 1.0).
 *
 * @param argc Argument count.
 * @param argv Argument values (command-line arguments).
 * @return True if the system supports the minimum CUDA capability, otherwise false.
 */
bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    // Print NPP library version
    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    // Print CUDA driver and runtime versions
    printf("  CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    // Check CUDA capabilities for SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

/**
 * Applies a Gaussian filter to an image and saves the result to an output file.
 * The function loads the image from the specified file path, applies the Gaussian filter
 * using NPP, and saves the filtered result to the output file.
 *
 * @param filePath Path to the input image file.
 * @param outputFile Path to save the processed image.
 */
void applyGaussFilter(const std::string &filePath, const std::string &outputFile)
{
    try
    {
        std::cout << "Processing of " << filePath << " started." << std::endl;

        // Load image into CPU memory
        npp::ImageCPU_8u_C1 hostSrc;
        npp::loadImage(filePath, hostSrc);

        // Copy image to GPU memory
        npp::ImageNPP_8u_C1 deviceSrc(hostSrc);

        const NppiSize srcSize = {(int) deviceSrc.width(), (int) deviceSrc.height()};
        const NppiPoint srcOffset = {0, 0};

        const NppiSize filterROI = {(int) deviceSrc.width(), (int) deviceSrc.height()};
        npp::ImageNPP_8u_C1 deviceDst(filterROI.width, filterROI.height);

        // Apply Gaussian filter using NPP
        NPP_CHECK_NPP(nppiFilterGaussBorder_8u_C1R(deviceSrc.data(), deviceSrc.pitch(), srcSize, srcOffset,
                                                   deviceDst.data(), deviceDst.pitch(), filterROI,
                                                   NppiMaskSize::NPP_MASK_SIZE_3_X_3,
                                                   NppiBorderType::NPP_BORDER_REPLICATE));

        // Copy filtered image back to CPU memory
        npp::ImageCPU_8u_C1 hostDst(deviceDst.size());
        deviceDst.copyTo(hostDst.data(), hostDst.pitch());

        // Save the processed image to output file
        npp::saveImage(outputFile, hostDst);

        std::cout << "Processing of " << filePath << " ended. Result saved to: " << outputFile << std::endl;

        // Free memory allocated for both input and output images
        nppiFree(deviceSrc.data());
        nppiFree(deviceDst.data());
        nppiFree(hostSrc.data());
        nppiFree(hostDst.data());
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
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * Thread worker that processes images by applying the Gaussian filter.
 *
 * @param taskQueue The queue of tasks (image paths) to process.
 * @param done A flag to signal when processing is complete.
 * @param outputDir The directory where processed images will be saved.
 */
void processImageWorker(std::queue<std::string> &taskQueue, std::mutex &queueMutex, std::condition_variable &cv,
                        std::atomic<bool> &done, const std::string &outputDir)
{
    while (true)
    {
        std::string imagePath;
        {
            std::unique_lock<std::mutex> lock(queueMutex);

            // Wait until there is work to do
            cv.wait(lock, [&] { return !taskQueue.empty() || done; });

            if (taskQueue.empty() && done)
                break;

            // Get the next image path from the queue
            imagePath = taskQueue.front();
            taskQueue.pop();
        }

        // Generate output file path
        std::string outputFile = outputDir + "/" + imagePath.substr(imagePath.find_last_of("/\\") + 1);
        outputFile = outputFile.substr(0, outputFile.find_last_of(".")) + "-filtered.pgm";

        // Process the image
        applyGaussFilter(imagePath, outputFile);
    }
}

/**
 * Cleans up the output directory by deleting all files in it.
 *
 * @param outputDir Path to the output directory.
 */
void cleanupOutputDirectory(const std::string &outputDir)
{
    for (const auto &entry : fs::directory_iterator(outputDir))
    {
        if (entry.is_regular_file())
        {
            std::cout << "Deleting file: " << entry.path() << std::endl;
            fs::remove(entry.path());
        }
    }
}

/**
 * Processes all images in a directory with a maximum number of threads equal to the CPU cores.
 *
 * @param inputDir Path to the input directory containing images.
 * @param outputDir Path to save the processed images.
 */
void processImagesInDirectory(const std::string &inputDir, const std::string &outputDir)
{
    // Clean up the output directory before starting processing
    cleanupOutputDirectory(outputDir);

    std::vector<std::thread> workers;
    std::queue<std::string> taskQueue;
    std::mutex queueMutex;
    std::condition_variable cv;
    std::atomic<bool> done(false);

    // Discover all image files in the input directory
    for (const auto &entry : fs::directory_iterator(inputDir))
    {
        if (entry.is_regular_file() && (entry.path().extension() == ".bmp" || entry.path().extension() == ".jpg"))
        {
            taskQueue.push(entry.path().string());
        }
    }

    // Start worker threads (limit to the number of available CPU cores)
    unsigned int numThreads = std::thread::hardware_concurrency();
    for (unsigned int i = 0; i < numThreads; ++i)
    {
        workers.push_back(std::thread(processImageWorker, std::ref(taskQueue), std::ref(queueMutex), std::ref(cv),
                                      std::ref(done), std::ref(outputDir)));
    }

    // Notify workers to start processing
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        done = false; // Set done to false as we are processing tasks
    }
    cv.notify_all();

    // Wait for all workers to finish processing
    for (auto &worker : workers)
    {
        worker.join();
    }

    // Mark that processing is done and notify all workers to exit
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        done = true;
    }
    cv.notify_all();
}

int main(int argc, char *argv[])
{
    // Print NPP and CUDA version information
    if (!printfNPPinfo(argc, argv))
    {
        exit(EXIT_SUCCESS);
    }

    // Input directory containing images
    std::string inputDir = "/home/sajith/dev/cuda-at-scale-for-the-enterprise/data/";

    // Output directory where processed images will be saved
    std::string outputDir = "/home/sajith/dev/cuda-at-scale-for-the-enterprise/output/";

    // Process images in the input directory
    processImagesInDirectory(inputDir, outputDir);

    return 0;
}
