#include <UtilNPP/Exceptions.h>
#include <UtilNPP/ImagesCPU.h>
#include <UtilNPP/ImagesNPP.h>
#include <UtilNPP/ImageIO.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <nppi.h>

#include <string>
#include <iostream>
#include <vector>
#include <thread>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * Prints detailed information about the NPP library and CUDA driver versions.
 *
 * This function retrieves and displays the version information for the NPP (NVIDIA Performance Primitives) library,
 * the CUDA driver, and the CUDA runtime. It also checks if the system meets the minimum CUDA capabilities.
 *
 * @param argc Number of command-line arguments passed to the program.
 * @param argv Array of command-line arguments.
 * @return True if the system meets the minimum CUDA capability requirements, otherwise false.
 */
bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();
    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    return checkCudaCapabilities(1, 0);  // Checks for a minimum of Compute Capability 1.0
}

/**
 * Applies a Gaussian filter to an input image and saves the processed result to an output file.
 *
 * This function uses the NVIDIA Performance Primitives (NPP) library to apply a Gaussian filter
 * to the input image. The processed image is then saved to the specified output file location.
 *
 * @param filePath Path to the input image file.
 * @param outputFile Path to save the filtered image.
 * @throws npp::Exception if any errors occur during NPP operations.
 * @throws std::exception for any other runtime errors.
 */
void applyGaussFilter(const std::string &filePath, const std::string &outputFile)
{
    try
    {
        std::cout << "Processing: " << filePath << std::endl;

        // Load the input image into CPU memory
        npp::ImageCPU_8u_C1 hostSrc;
        npp::loadImage(filePath, hostSrc);

        // Copy the image to GPU memory
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

        // Copy the processed image back to CPU memory
        npp::ImageCPU_8u_C1 hostDst(deviceDst.size());
        deviceDst.copyTo(hostDst.data(), hostDst.pitch());

        // Save the processed image to the output file
        npp::saveImage(outputFile, hostDst);
        std::cout << "Finished: " << filePath << " -> " << outputFile << std::endl;

        // Free allocated memory
        nppiFree(deviceSrc.data());
        nppiFree(deviceDst.data());
        nppiFree(hostSrc.data());
        nppiFree(hostDst.data());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error processing " << filePath << ": " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown error processing " << filePath << std::endl;
    }
}

/**
 * Cleans up the output directory by deleting all files in it.
 *
 * @param outputDir Path to the directory to clean up.
 * @throws std::filesystem::filesystem_error if any file operations fail.
 */
void cleanupOutputDirectory(const std::string &outputDir)
{
    for (const auto &entry: fs::directory_iterator(outputDir))
    {
        if (entry.is_regular_file())
        {
            std::cout << "Deleting: " << entry.path() << std::endl;
            fs::remove(entry.path());
        }
    }
}

/**
 * Processes a batch of images concurrently using threads.
 *
 * This function takes a batch of image paths and processes them using threads, with each thread
 * applying the Gaussian filter to one image. All threads are joined before the function exits.
 *
 * @param batch A vector of paths to the image files in the batch.
 * @param outputDir Directory to save the processed images.
 */
void processBatch(const std::vector<std::string> &batch, const std::string &outputDir)
{
    std::vector<std::thread> workers;
    for (const auto &imagePath: batch)
    {
        if (fs::exists(imagePath) && fs::is_regular_file(imagePath))
        {
            workers.emplace_back([=]()
                                 {
                                     std::string outputFile =
                                             outputDir + "/" + imagePath.substr(imagePath.find_last_of("/\\") + 1);
                                     outputFile =
                                             outputFile.substr(0, outputFile.find_last_of(".")) + "_gauss_filtered.pgm";
                                     applyGaussFilter(imagePath, outputFile);
                                 });
        }
        else
        {
            std::cerr << "Invalid image path: " << imagePath << std::endl;
        }
    }

    // Wait for all threads in the batch to complete
    for (auto &worker: workers)
    {
        worker.join();
    }
}

/**
 * Processes all images in a directory in batches.
 *
 * This function iterates over all images in the input directory, organizes them into batches,
 * and processes each batch concurrently using `processBatch`. The number of threads used
 * is determined by the number of CPU cores.
 *
 * @param inputDir Directory containing the input images.
 * @param outputDir Directory to save the processed images.
 */
void processImagesInDirectory(const std::string &inputDir, const std::string &outputDir)
{
    if (fs::exists(outputDir))
        cleanupOutputDirectory(outputDir);
    else
        fs::create_directory(outputDir);

    // Collect all valid image paths from the input directory
    std::vector<std::string> imagePaths;
    for (const auto &entry: fs::recursive_directory_iterator(inputDir))
    {
        imagePaths.push_back(entry.path().string());
    }

    // Determine the number of threads to use
    unsigned int numThreads = std::thread::hardware_concurrency();
    std::vector<std::string> batch;

    // Process images in batches
    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        batch.push_back(imagePaths[i]);

        if (batch.size() == numThreads || i == imagePaths.size() - 1)
        {
            processBatch(batch, outputDir);
            batch.clear();
        }
    }
}


// Function to parse input and output directories from command-line arguments
std::tuple<std::string, std::string> parseInputOutputDirs(int argc, char *argv[])
{
    // Default input and output directories
    std::string inputDir = "../data/";
    std::string outputDir = "../output/";

    // Parse command-line arguments for input and output directories
    if (checkCmdLineFlag(argc, (const char **) argv, "input"))
    {
        char *inputPath = nullptr;
        getCmdLineArgumentString(argc, (const char **) argv, "input", &inputPath);
        if (inputPath)
        {
            inputDir = inputPath;
        }
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "output"))
    {
        char *outputPath = nullptr;
        getCmdLineArgumentString(argc, (const char **) argv, "output", &outputPath);
        if (outputPath)
        {
            outputDir = outputPath;
        }
    }

    // Return the directories as a tuple
    return std::make_tuple(inputDir, outputDir);
}


/**
 * Entry point for the program.
 *
 * This function prints CUDA and NPP version information, then processes all images
 * in the specified input directory by applying a Gaussian filter. The processed images
 * are saved to the specified output directory.
 */
int main(int argc, char *argv[])
{
    if (!printfNPPinfo(argc, argv))
    {
        exit(EXIT_SUCCESS);
    }


    // Call the function to parse input and output directories
    std::string inputDir, outputDir;
    std::tie(inputDir, outputDir) = parseInputOutputDirs(argc, argv);

    // Log the directories being used
    std::cout << "Input Directory: " << inputDir << std::endl;
    std::cout << "Output Directory: " << outputDir << std::endl;

    // Process images in the input directory
    processImagesInDirectory(inputDir, outputDir);

    return 0;
}
