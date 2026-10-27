#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace chrono;

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            cerr << "CUDA hatasi: "                           \
                 << cudaGetErrorString(err) << endl;           \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

using Image = vector<unsigned char>;
using Histogram = array<unsigned int, 256>;

const int WIDTH = 1024;
const int HEIGHT = 1024;
const int PIXELS = WIDTH * HEIGHT;
const int BLOCK_SIZE = 256;

// --------------------------------------------------
// 1. TEST GORUNTUSU OLUSTURMA
// --------------------------------------------------

Image createImage(int width, int height) {
    Image img(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img[y * width + x] = 
                (x * 3 + y * 2) % 256;
        }
    }

    return img;
}

// --------------------------------------------------
// 2. PGM FORMATINDA GORUNTU KAYDETME
// --------------------------------------------------

void savePGM(
    const string& filename,
    const Image& img,
    int width,
    int height
) {
    ofstream file(filename, ios::binary);

    file << "P5\n"
         << width << " " << height << "\n"
         << "255\n";

    file.write(
        reinterpret_cast<const char*>(img.data()),
        img.size()
    );
}

// --------------------------------------------------
// 3. CPU ILE GORUNTU YENIDEN BOYUTLANDIRMA
// --------------------------------------------------

Image resizeCPU(
    const Image& input,
    int oldW,
    int oldH,
    int newW,
    int newH
) {
    Image output(newW * newH);

    for (int y = 0; y < newH; y++) {
        for (int x = 0; x < newW; x++) {
            int srcX = x * oldW / newW;
            int srcY = y * oldH / newH;

            output[y * newW + x] = 
                input[srcY * oldW + srcX];
        }
    }

    return output;
}

// --------------------------------------------------
// 4. CPU ILE 8 BITTEN 6 BITE NICEMLEME
// --------------------------------------------------

Image quantizeCPU(const Image& input) {
    Image output(input.size());

    for (size_t i = 0; i < input.size(); i++) {
        int q = input[i] >> 2;

        // 0-63 araligini 0-255 araligina geri tasir.
        output[i] = static_cast<unsigned char>(
            (q * 255 + 31) / 63
        );
    }

    return output;
}

// --------------------------------------------------
// 5. CPU TEK THREAD PARLAKLIK
// --------------------------------------------------

Image brightnessCPU1(
    const Image& input,
    int brightness
) {
    Image output(input.size());

    for (size_t i = 0; i < input.size(); i++) {
        int value = input[i] + brightness;

        output[i] = static_cast<unsigned char>(
            max(0, min(255, value))
        );
    }

    return output;
}

// --------------------------------------------------
// 6. CPU 4 THREAD PARLAKLIK
// --------------------------------------------------

void brightnessPart(
    const Image& input,
    Image& output,
    size_t start,
    size_t end,
    int brightness
) {
    for (size_t i = start; i < end; i++) {
        int value = input[i] + brightness;

        output[i] = static_cast<unsigned char>(
            max(0, min(255, value))
        );
    }
}

Image brightnessCPU4(
    const Image& input,
    int brightness
) {
    Image output(input.size());

    vector<thread> threads;

    size_t n = input.size();

    for (int t = 0; t < 4; t++) {
        size_t start = n * t / 4;
        size_t end = n * (t + 1) / 4;

        threads.emplace_back(
            brightnessPart,
            cref(input),
            ref(output),
            start,
            end,
            brightness
        );
    }

    for (auto& t : threads) {
        t.join();
    }

    return output;
}

// --------------------------------------------------
// 7. CPU TEK THREAD HISTOGRAM
// --------------------------------------------------

Histogram histogramCPU1(const Image& input) {
    Histogram hist{};

    for (unsigned char pixel : input) {
        hist[pixel]++;
    }

    return hist;
}

// --------------------------------------------------
// 8. CPU 4 THREAD HISTOGRAM
// --------------------------------------------------

Histogram histogramCPU4(const Image& input) {
    array<Histogram, 4> localHist{};

    vector<thread> threads;

    size_t n = input.size();

    for (int t = 0; t < 4; t++) {
        size_t start = n * t / 4;
        size_t end = n * (t + 1) / 4;

        threads.emplace_back([&, t, start, end]() {
            for (size_t i = start; i < end; i++) {
                localHist[t][input[i]]++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    Histogram result{};

    for (int t = 0; t < 4; t++) {
        for (int i = 0; i < 256; i++) {
            result[i] += localHist[t][i];
        }
    }

    return result;
}

// --------------------------------------------------
// 9. CUDA PARLAKLIK KERNEL
// --------------------------------------------------

__global__ void brightnessKernel(
    const unsigned char* input,
    unsigned char* output,
    int n,
    int brightness
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int value = input[i] + brightness;

        value = max(0, min(255, value));

        output[i] = static_cast<unsigned char>(value);
    }
}

// --------------------------------------------------
// 10. CUDA NICEMLEME KERNEL
// --------------------------------------------------

__global__ void quantizeKernel(
    const unsigned char* input,
    unsigned char* output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int q = input[i] >> 2;

        output[i] = static_cast<unsigned char>(
            (q * 255 + 31) / 63
        );
    }
}

// --------------------------------------------------
// 11. CUDA YENIDEN BOYUTLANDIRMA KERNEL
// --------------------------------------------------

__global__ void resizeKernel(
    const unsigned char* input,
    unsigned char* output,
    int oldW,
    int oldH,
    int newW,
    int newH
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int total = newW * newH;

    if (i < total) {
        int x = i % newW;
        int y = i / newW;

        int srcX = x * oldW / newW;
        int srcY = y * oldH / newH;

        output[i] = input[srcY * oldW + srcX];
    }
}

// --------------------------------------------------
// 12. CUDA HISTOGRAM KERNEL
// --------------------------------------------------

__global__ void histogramKernel(
    const unsigned char* input,
    unsigned int* histogram,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        atomicAdd(&histogram[input[i]], 1u);
    }
}

// --------------------------------------------------
// 13. CUDA PIXEL DONUSUMU
// --------------------------------------------------

__global__ void transformKernel(
    const unsigned char* input,
    unsigned char* output,
    int n,
    float scale,
    int offset
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int value = static_cast<int>(
            scale * input[i] + offset
        );

        value = max(0, min(255, value));

        output[i] = static_cast<unsigned char>(value);
    }
}

// --------------------------------------------------
// 14. ANA PROGRAM
// --------------------------------------------------

int main() {
    cout << "GORUNTU ISLEME CUDA ODEVI\n";
    cout << "-------------------------\n";

    // Test goruntusu olustur.
    Image input = createImage(WIDTH, HEIGHT);

    savePGM(
        "00_input.pgm",
        input,
        WIDTH,
        HEIGHT
    );

    int n = static_cast<int>(input.size());

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, n));
    CUDA_CHECK(cudaMalloc(&d_output, n));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        input.data(),
        n,
        cudaMemcpyHostToDevice
    ));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // GPU histogram belleği.
    unsigned int* d_histogram = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_histogram,
        256 * sizeof(unsigned int)
    ));

    // ------------------------------
    // A. YENIDEN BOYUTLANDIRMA
    // ------------------------------

    int newW = WIDTH / 2;
    int newH = HEIGHT / 2;

    Image resized(newW * newH);

    unsigned char* d_resized = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_resized,
        resized.size()
    ));

    auto resizeStart = high_resolution_clock::now();

    int resizeBlocks = 
        (newW * newH + BLOCK_SIZE - 1) / BLOCK_SIZE;

    resizeKernel<<<resizeBlocks, BLOCK_SIZE>>>(
        d_input,
        d_resized,
        WIDTH,
        HEIGHT,
        newW,
        newH
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto resizeEnd = high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(
        resized.data(),
        d_resized,
        resized.size(),
        cudaMemcpyDeviceToHost
    ));

    savePGM(
        "01_resized_cuda.pgm",
        resized,
        newW,
        newH
    );

    double resizeTime = 
        duration<double, milli>(
            resizeEnd - resizeStart
        ).count();

    // ------------------------------
    // B. 6 BIT NICEMLEME
    // ------------------------------

    Image quantized(n);

    auto quantStart = high_resolution_clock::now();

    quantizeKernel<<<blocks, BLOCK_SIZE>>>(
        d_input,
        d_output,
        n
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto quantEnd = high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(
        quantized.data(),
        d_output,
        n,
        cudaMemcpyDeviceToHost
    ));

    savePGM(
        "02_quantized_6bit.pgm",
        quantized,
        WIDTH,
        HEIGHT
    );

    double quantTime = 
        duration<double, milli>(
            quantEnd - quantStart
        ).count();

    // ------------------------------
    // C. CPU 1 THREAD PARLAKLIK
    // ------------------------------

    Image bright1;

    auto cpu1Start = high_resolution_clock::now();

    bright1 = brightnessCPU1(input, 40);

    auto cpu1End = high_resolution_clock::now();

    double cpu1Time = 
        duration<double, milli>(
            cpu1End - cpu1Start
        ).count();

    savePGM(
        "03_brightness_cpu1.pgm",
        bright1,
        WIDTH,
        HEIGHT
    );

    // ------------------------------
    // D. CPU 4 THREAD PARLAKLIK
    // ------------------------------

    Image bright4;

    auto cpu4Start = high_resolution_clock::now();

    bright4 = brightnessCPU4(input, 40);

    auto cpu4End = high_resolution_clock::now();

    double cpu4Time = 
        duration<double, milli>(
            cpu4End - cpu4Start
        ).count();

    savePGM(
        "04_brightness_cpu4.pgm",
        bright4,
        WIDTH,
        HEIGHT
    );

    // ------------------------------
    // E. CUDA PARLAKLIK
    // ------------------------------

    Image brightGPU(n);

    auto gpuStart = high_resolution_clock::now();

    brightnessKernel<<<blocks, BLOCK_SIZE>>>(
        d_input,
        d_output,
        n,
        40
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpuEnd = high_resolution_clock::now();

    double gpuTime = 
        duration<double, milli>(
            gpuEnd - gpuStart
        ).count();

    CUDA_CHECK(cudaMemcpy(
        brightGPU.data(),
        d_output,
        n,
        cudaMemcpyDeviceToHost
    ));

    savePGM(
        "05_brightness_cuda.pgm",
        brightGPU,
        WIDTH,
        HEIGHT
    );

    // ------------------------------
    // F. HISTOGRAM CPU 1 THREAD
    // ------------------------------

    Histogram hist1{};

    auto hist1Start = high_resolution_clock::now();

    hist1 = histogramCPU1(input);

    auto hist1End = high_resolution_clock::now();

    double hist1Time = 
        duration<double, milli>(
            hist1End - hist1Start
        ).count();

    // ------------------------------
    // G. HISTOGRAM CPU 4 THREAD
    // ------------------------------

    Histogram hist4{};

    auto hist4Start = high_resolution_clock::now();

    hist4 = histogramCPU4(input);

    auto hist4End = high_resolution_clock::now();

    double hist4Time = 
        duration<double, milli>(
            hist4End - hist4Start
        ).count();

    // ------------------------------
    // H. HISTOGRAM CUDA
    // ------------------------------

    CUDA_CHECK(cudaMemset(
        d_histogram,
        0,
        256 * sizeof(unsigned int)
    ));

    auto histGPUStart = high_resolution_clock::now();

    histogramKernel<<<blocks, BLOCK_SIZE>>>(
        d_input,
        d_histogram,
        n
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto histGPUEnd = high_resolution_clock::now();

    double histGPUTime = 
        duration<double, milli>(
            histGPUEnd - histGPUStart
        ).count();

    Histogram histGPU{};

    CUDA_CHECK(cudaMemcpy(
        histGPU.data(),
        d_histogram,
        256 * sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    ));

    // Histogram dosyasi.
    ofstream histFile("histogram.csv");

    histFile << "Pixel,CPU1,CPU4,CUDA\n";

    for (int i = 0; i < 256; i++) {
        histFile << i << ","
                 << hist1[i] << ","
                 << hist4[i] << ","
                 << histGPU[i] << "\n";
    }

    // ------------------------------
    // I. PIXEL DONUSUMLERI
    // ------------------------------

    Image scaleOnly(n);
    Image scaleOffset(n);

    transformKernel<<<blocks, BLOCK_SIZE>>>(
        d_input,
        d_output,
        n,
        0.75f,
        0
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        scaleOnly.data(),
        d_output,
        n,
        cudaMemcpyDeviceToHost
    ));

    savePGM(
        "06_scale_075.pgm",
        scaleOnly,
        WIDTH,
        HEIGHT
    );

    transformKernel<<<blocks, BLOCK_SIZE>>>(
        d_input,
        d_output,
        n,
        0.75f,
        20
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        scaleOffset.data(),
        d_output,
        n,
        cudaMemcpyDeviceToHost
    ));

    savePGM(
        "07_scale_075_plus20.pgm",
        scaleOffset,
        WIDTH,
        HEIGHT
    );

    // ------------------------------
    // J. SAYISAL KARSILASTIRMA
    // ------------------------------

    long long diffSum = 0;
    int diffPixels = 0;

    for (int i = 0; i < n; i++) {
        int diff = abs(
            static_cast<int>(scaleOffset[i]) -
            static_cast<int>(scaleOnly[i])
        );

        diffSum += diff;

        if (diff != 0) {
            diffPixels++;
        }
    }

    bool brightnessEqual = 
        (bright1 == bright4) && 
        (bright1 == brightGPU);

    bool histogramEqual = 
        (hist1 == hist4) && 
        (hist1 == histGPU);

    // ------------------------------
    // K. SONUCLARI YAZDIR
    // ------------------------------

    cout << "\n--- CALISMA SURELERI ---\n";

    cout << "CUDA GPU modeli: ";

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    cout << prop.name << "\n";

    cout << "Yeniden boyutlandirma CUDA: " 
         << resizeTime << " ms\n";

    cout << "6-bit nicemleme CUDA: " 
         << quantTime << " ms\n";

    cout << "Parlaklik CPU 1 thread: " 
         << cpu1Time << " ms\n";

    cout << "Parlaklik CPU 4 thread: " 
         << cpu4Time << " ms\n";

    cout << "Parlaklik CUDA GPU: " 
         << gpuTime << " ms\n";

    cout << "Histogram CPU 1 thread: " 
         << hist1Time << " ms\n";

    cout << "Histogram CPU 4 thread: " 
         << hist4Time << " ms\n";

    cout << "Histogram CUDA GPU: " 
         << histGPUTime << " ms\n";

    cout << "\n--- DOGRULAMA ---\n";

    cout << "Parlaklik sonuclari ayni mi? " 
         << (brightnessEqual ? "EVET" : "HAYIR") 
         << "\n";

    cout << "Histogram sonuclari ayni mi? " 
         << (histogramEqual ? "EVET" : "HAYIR") 
         << "\n";

    cout << "Donusumde farkli piksel sayisi: " 
         << diffPixels << "\n";

    cout << "Mutlak piksel farklari toplami: " 
         << diffSum << "\n";

    // Sonuclari dosyaya da yaz.
    ofstream report("results.txt");

    report << "GORUNTU ISLEME CUDA ODEVI\n";
    report << "GPU: " << prop.name << "\n";
    report << "Goruntu: " << WIDTH << "x" << HEIGHT << "\n";

    report << "Yeniden boyutlandirma CUDA: " 
           << resizeTime << " ms\n";

    report << "6-bit nicemleme CUDA: " 
           << quantTime << " ms\n";

    report << "Parlaklik CPU 1 thread: " 
           << cpu1Time << " ms\n";

    report << "Parlaklik CPU 4 thread: " 
           << cpu4Time << " ms\n";

    report << "Parlaklik CUDA: " 
           << gpuTime << " ms\n";

    report << "Histogram CPU 1 thread: " 
           << hist1Time << " ms\n";

    report << "Histogram CPU 4 thread: " 
           << hist4Time << " ms\n";

    report << "Histogram CUDA: " 
           << histGPUTime << " ms\n";

    report << "Parlaklik sonuclari ayni mi: " 
           << (brightnessEqual ? "EVET" : "HAYIR") 
           << "\n";

    report << "Histogram sonuclari ayni mi: " 
           << (histogramEqual ? "EVET" : "HAYIR") 
           << "\n";

    report << "Farkli piksel sayisi: " 
           << diffPixels << "\n";

    report << "Mutlak piksel farki toplami: " 
           << diffSum << "\n";

    cout << "\nSonuclar results.txt dosyasina kaydedildi.\n";
    cout << "Histogram histogram.csv dosyasina kaydedildi.\n";

    // CUDA bellegini serbest birak.
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_histogram));
    CUDA_CHECK(cudaFree(d_resized));

    return 0;
}
