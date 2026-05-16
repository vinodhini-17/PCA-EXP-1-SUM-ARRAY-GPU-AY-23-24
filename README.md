# PCA: EXP-1  SUM ARRAY GPU
<h3>ENTER YOUR NAME:VINODHINI K</h3>
<h3>ENTER YOUR REGISTER NO:212223230245</h3>
<h3>EX. NO:01</h3>
<h3>DATE:12/05/2026</h3>
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:
## Threads=512:
```
   %%cuda
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <sys/time.h>
    
    #ifndef _COMMON_H
    #define _COMMON_H
    
    #define CHECK(call)                                                            \
    {                                                                              \
        const cudaError_t error = call;                                            \
        if (error != cudaSuccess)                                                  \
        {                                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
            fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                    cudaGetErrorString(error));                                    \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CUBLAS(call)                                                     \
    {                                                                              \
        cublasStatus_t err;                                                        \
        if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
        {                                                                          \
            fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                    __LINE__);                                                     \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CURAND(call)                                                     \
    {                                                                              \
        curandStatus_t err;                                                        \
        if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
        {                                                                          \
            fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                    __LINE__);                                                     \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CUFFT(call)                                                      \
    {                                                                              \
        cufftResult err;                                                           \
        if ( (err = (call)) != CUFFT_SUCCESS)                                      \
        {                                                                          \
            fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                    __LINE__);                                                     \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CUSPARSE(call)                                                   \
    {                                                                              \
        cusparseStatus_t err;                                                      \
        if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                          \
            fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
            cudaError_t cuda_err = cudaGetLastError();                             \
            if (cuda_err != cudaSuccess)                                           \
            {                                                                      \
                fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                        cudaGetErrorString(cuda_err));                             \
            }                                                                      \
            exit(1);                                                               \
        }                                                                          \
    }
    
    inline double seconds()
    {
        struct timeval tp;
        struct timezone tzp;
        int i = gettimeofday(&tp, &tzp);
        return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
    }
    
    #endif // _COMMON_H
    
    
    
    void checkResult(float *hostRef, float *gpuRef, const int N)
    {
        double epsilon = 1.0E-8;
        bool match = 1;
    
        for (int i = 0; i < N; i++)
        {
            if (abs(hostRef[i] - gpuRef[i]) > epsilon)
            {
                match = 0;
                printf("Arrays do not match!\n");
                printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                       gpuRef[i], i);
                break;
            }
        }
    
        if (match) printf("Arrays match.\n\n");
    
        return;
    }
    
    void initialData(float *ip, int size)
    {
        // generate different seed for random number
        time_t t;
        srand((unsigned) time(&t));
    
        for (int i = 0; i < size; i++)
        {
            ip[i] = (float)( rand() & 0xFF ) / 10.0f;
        }
    
        return;
    }
    
    void sumArraysOnHost(float *A, float *B, float *C, const int N)
    {
        for (int idx = 0; idx < N; idx++)
        {
            C[idx] = A[idx] + B[idx];
        }
    }
    
    
    
    
    
    
    __global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (i < N)
        {
            C[i] = A[i] + B[i];
        }
    }
    
    
    
    int main(int argc, char **argv)
    {
        printf("%s Starting...\n", argv[0]);
    
        // set up device
        int dev = 0;
        cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, dev));
        printf("Using Device %d: %s\n", dev, deviceProp.name);
        CHECK(cudaSetDevice(dev));
    
        // set up data size of vectors
        int nElem =1000;
        printf("Vector size %d\n", nElem);
    
        // malloc host memory
        size_t nBytes = nElem * sizeof(float);
    
        float *h_A, *h_B, *hostRef, *gpuRef;
        h_A     = (float *)malloc(nBytes);
        h_B     = (float *)malloc(nBytes);
        hostRef = (float *)malloc(nBytes);
        gpuRef  = (float *)malloc(nBytes);
    
        double iStart, iElaps;
    
        // initialize data at host side
        iStart = seconds();
        initialData(h_A, nElem);
        initialData(h_B, nElem);
        iElaps = seconds() - iStart;
        printf("initialData Time elapsed %f sec\n", iElaps);
        memset(hostRef, 0, nBytes);
        memset(gpuRef,  0, nBytes);
    
        // add vector at host side for result checks
        iStart = seconds();
        sumArraysOnHost(h_A, h_B, hostRef, nElem);
        iElaps = seconds() - iStart;
        printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);
    
        // malloc device global memory
        float *d_A, *d_B, *d_C;
        CHECK(cudaMalloc((float**)&d_A, nBytes));
        CHECK(cudaMalloc((float**)&d_B, nBytes));
        CHECK(cudaMalloc((float**)&d_C, nBytes));
    
        // transfer data from host to device
        CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
    
        // invoke kernel at host side
        int iLen = 512;
        dim3 block (iLen);
        dim3 grid  ((nElem + block.x - 1) / block.x);
    
        iStart = seconds();
        sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
        CHECK(cudaDeviceSynchronize());
        iElaps = seconds() - iStart;
        printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
               block.x, iElaps);
    
        // check kernel error
        CHECK(cudaGetLastError()) ;
    
        // copy kernel result back to host side
        CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    
        // check device results
        checkResult(hostRef, gpuRef, nElem);
    
        // free device global memory
        CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_B));
        CHECK(cudaFree(d_C));
    
        // free host memory
        free(h_A);
        free(h_B);
        free(hostRef);
        free(gpuRef);
    
        return(0);
    }

```

## Threads=1024:
```
 %%cuda
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <sys/time.h>
    
    #ifndef _COMMON_H
    #define _COMMON_H
    
    #define CHECK(call)                                                            \
    {                                                                              \
        const cudaError_t error = call;                                            \
        if (error != cudaSuccess)                                                  \
        {                                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
            fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                    cudaGetErrorString(error));                                    \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CUBLAS(call)                                                     \
    {                                                                              \
        cublasStatus_t err;                                                        \
        if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
        {                                                                          \
            fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                    __LINE__);                                                     \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CURAND(call)                                                     \
    {                                                                              \
        curandStatus_t err;                                                        \
        if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
        {                                                                          \
            fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                    __LINE__);                                                     \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CUFFT(call)                                                      \
    {                                                                              \
        cufftResult err;                                                           \
        if ( (err = (call)) != CUFFT_SUCCESS)                                      \
        {                                                                          \
            fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                    __LINE__);                                                     \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CUSPARSE(call)                                                   \
    {                                                                              \
        cusparseStatus_t err;                                                      \
        if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                          \
            fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
            cudaError_t cuda_err = cudaGetLastError();                             \
            if (cuda_err != cudaSuccess)                                           \
            {                                                                      \
                fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                        cudaGetErrorString(cuda_err));                             \
            }                                                                      \
            exit(1);                                                               \
        }                                                                          \
    }
    
    inline double seconds()
    {
        struct timeval tp;
        struct timezone tzp;
        int i = gettimeofday(&tp, &tzp);
        return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
    }
    
    #endif // _COMMON_H
    
    
    
    void checkResult(float *hostRef, float *gpuRef, const int N)
    {
        double epsilon = 1.0E-8;
        bool match = 1;
    
        for (int i = 0; i < N; i++)
        {
            if (abs(hostRef[i] - gpuRef[i]) > epsilon)
            {
                match = 0;
                printf("Arrays do not match!\n");
                printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                       gpuRef[i], i);
                break;
            }
        }
    
        if (match) printf("Arrays match.\n\n");
    
        return;
    }
    
    void initialData(float *ip, int size)
    {
        // generate different seed for random number
        time_t t;
        srand((unsigned) time(&t));
    
        for (int i = 0; i < size; i++)
        {
            ip[i] = (float)( rand() & 0xFF ) / 10.0f;
        }
    
        return;
    }
    
    void sumArraysOnHost(float *A, float *B, float *C, const int N)
    {
        for (int idx = 0; idx < N; idx++)
        {
            C[idx] = A[idx] + B[idx];
        }
    }
    
    
    
    
    
    
    __global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (i < N)
        {
            C[i] = A[i] + B[i];
        }
    }
    
    
    
    int main(int argc, char **argv)
    {
        printf("%s Starting...\n", argv[0]);
    
        // set up device
        int dev = 0;
        cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, dev));
        printf("Using Device %d: %s\n", dev, deviceProp.name);
        CHECK(cudaSetDevice(dev));
    
        // set up data size of vectors
        int nElem =1000;
        printf("Vector size %d\n", nElem);
    
        // malloc host memory
        size_t nBytes = nElem * sizeof(float);
    
        float *h_A, *h_B, *hostRef, *gpuRef;
        h_A     = (float *)malloc(nBytes);
        h_B     = (float *)malloc(nBytes);
        hostRef = (float *)malloc(nBytes);
        gpuRef  = (float *)malloc(nBytes);
    
        double iStart, iElaps;
    
        // initialize data at host side
        iStart = seconds();
        initialData(h_A, nElem);
        initialData(h_B, nElem);
        iElaps = seconds() - iStart;
        printf("initialData Time elapsed %f sec\n", iElaps);
        memset(hostRef, 0, nBytes);
        memset(gpuRef,  0, nBytes);
    
        // add vector at host side for result checks
        iStart = seconds();
        sumArraysOnHost(h_A, h_B, hostRef, nElem);
        iElaps = seconds() - iStart;
        printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);
    
        // malloc device global memory
        float *d_A, *d_B, *d_C;
        CHECK(cudaMalloc((float**)&d_A, nBytes));
        CHECK(cudaMalloc((float**)&d_B, nBytes));
        CHECK(cudaMalloc((float**)&d_C, nBytes));
    
        // transfer data from host to device
        CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
    
        // invoke kernel at host side
        int iLen = 1024;
        dim3 block (iLen);
        dim3 grid  ((nElem + block.x - 1) / block.x);
    
        iStart = seconds();
        sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
        CHECK(cudaDeviceSynchronize());
        iElaps = seconds() - iStart;
        printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
               block.x, iElaps);
    
        // check kernel error
        CHECK(cudaGetLastError()) ;
    
        // copy kernel result back to host side
        CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    
        // check device results
        checkResult(hostRef, gpuRef, nElem);
    
        // free device global memory
        CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_B));
        CHECK(cudaFree(d_C));
    
        // free host memory
        free(h_A);
        free(h_B);
        free(hostRef);
        free(gpuRef);
    
        return(0);
    }
```
## Threads=1023:
```
 %%cuda
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <sys/time.h>
    
    #ifndef _COMMON_H
    #define _COMMON_H
    
    #define CHECK(call)                                                            \
    {                                                                              \
        const cudaError_t error = call;                                            \
        if (error != cudaSuccess)                                                  \
        {                                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
            fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                    cudaGetErrorString(error));                                    \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CUBLAS(call)                                                     \
    {                                                                              \
        cublasStatus_t err;                                                        \
        if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
        {                                                                          \
            fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                    __LINE__);                                                     \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CURAND(call)                                                     \
    {                                                                              \
        curandStatus_t err;                                                        \
        if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
        {                                                                          \
            fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                    __LINE__);                                                     \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CUFFT(call)                                                      \
    {                                                                              \
        cufftResult err;                                                           \
        if ( (err = (call)) != CUFFT_SUCCESS)                                      \
        {                                                                          \
            fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                    __LINE__);                                                     \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CUSPARSE(call)                                                   \
    {                                                                              \
        cusparseStatus_t err;                                                      \
        if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                          \
            fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
            cudaError_t cuda_err = cudaGetLastError();                             \
            if (cuda_err != cudaSuccess)                                           \
            {                                                                      \
                fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                        cudaGetErrorString(cuda_err));                             \
            }                                                                      \
            exit(1);                                                               \
        }                                                                          \
    }
    
    inline double seconds()
    {
        struct timeval tp;
        struct timezone tzp;
        int i = gettimeofday(&tp, &tzp);
        return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
    }
    
    #endif // _COMMON_H
    
    
    
    void checkResult(float *hostRef, float *gpuRef, const int N)
    {
        double epsilon = 1.0E-8;
        bool match = 1;
    
        for (int i = 0; i < N; i++)
        {
            if (abs(hostRef[i] - gpuRef[i]) > epsilon)
            {
                match = 0;
                printf("Arrays do not match!\n");
                printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                       gpuRef[i], i);
                break;
            }
        }
    
        if (match) printf("Arrays match.\n\n");
    
        return;
    }
    
    void initialData(float *ip, int size)
    {
        // generate different seed for random number
        time_t t;
        srand((unsigned) time(&t));
    
        for (int i = 0; i < size; i++)
        {
            ip[i] = (float)( rand() & 0xFF ) / 10.0f;
        }
    
        return;
    }
    
    void sumArraysOnHost(float *A, float *B, float *C, const int N)
    {
        for (int idx = 0; idx < N; idx++)
        {
            C[idx] = A[idx] + B[idx];
        }
    }
    
    
    
    
    
    
    __global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (i < N)
        {
            C[i] = A[i] + B[i];
        }
    }
    
    
    
    int main(int argc, char **argv)
    {
        printf("%s Starting...\n", argv[0]);
    
        // set up device
        int dev = 0;
        cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, dev));
        printf("Using Device %d: %s\n", dev, deviceProp.name);
        CHECK(cudaSetDevice(dev));
    
        // set up data size of vectors
        int nElem =1000;
        printf("Vector size %d\n", nElem);
    
        // malloc host memory
        size_t nBytes = nElem * sizeof(float);
    
        float *h_A, *h_B, *hostRef, *gpuRef;
        h_A     = (float *)malloc(nBytes);
        h_B     = (float *)malloc(nBytes);
        hostRef = (float *)malloc(nBytes);
        gpuRef  = (float *)malloc(nBytes);
    
        double iStart, iElaps;
    
        // initialize data at host side
        iStart = seconds();
        initialData(h_A, nElem);
        initialData(h_B, nElem);
        iElaps = seconds() - iStart;
        printf("initialData Time elapsed %f sec\n", iElaps);
        memset(hostRef, 0, nBytes);
        memset(gpuRef,  0, nBytes);
    
        // add vector at host side for result checks
        iStart = seconds();
        sumArraysOnHost(h_A, h_B, hostRef, nElem);
        iElaps = seconds() - iStart;
        printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);
    
        // malloc device global memory
        float *d_A, *d_B, *d_C;
        CHECK(cudaMalloc((float**)&d_A, nBytes));
        CHECK(cudaMalloc((float**)&d_B, nBytes));
        CHECK(cudaMalloc((float**)&d_C, nBytes));
    
        // transfer data from host to device
        CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
    
        // invoke kernel at host side
        int iLen = 1023;
        dim3 block (iLen);
        dim3 grid  ((nElem + block.x - 1) / block.x);
    
        iStart = seconds();
        sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
        CHECK(cudaDeviceSynchronize());
        iElaps = seconds() - iStart;
        printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
               block.x, iElaps);
    
        // check kernel error
        CHECK(cudaGetLastError()) ;
    
        // copy kernel result back to host side
        CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    
        // check device results
        checkResult(hostRef, gpuRef, nElem);
    
        // free device global memory
        CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_B));
        CHECK(cudaFree(d_C));
    
        // free host memory
        free(h_A);
        free(h_B);
        free(hostRef);
        free(gpuRef);
    
        return(0);
    }
```

## Threads=256:
```
 %%cuda
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <sys/time.h>
    
    #ifndef _COMMON_H
    #define _COMMON_H
    
    #define CHECK(call)                                                            \
    {                                                                              \
        const cudaError_t error = call;                                            \
        if (error != cudaSuccess)                                                  \
        {                                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
            fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                    cudaGetErrorString(error));                                    \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CUBLAS(call)                                                     \
    {                                                                              \
        cublasStatus_t err;                                                        \
        if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
        {                                                                          \
            fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                    __LINE__);                                                     \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CURAND(call)                                                     \
    {                                                                              \
        curandStatus_t err;                                                        \
        if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
        {                                                                          \
            fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                    __LINE__);                                                     \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CUFFT(call)                                                      \
    {                                                                              \
        cufftResult err;                                                           \
        if ( (err = (call)) != CUFFT_SUCCESS)                                      \
        {                                                                          \
            fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                    __LINE__);                                                     \
            exit(1);                                                               \
        }                                                                          \
    }
    
    #define CHECK_CUSPARSE(call)                                                   \
    {                                                                              \
        cusparseStatus_t err;                                                      \
        if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                          \
            fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
            cudaError_t cuda_err = cudaGetLastError();                             \
            if (cuda_err != cudaSuccess)                                           \
            {                                                                      \
                fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                        cudaGetErrorString(cuda_err));                             \
            }                                                                      \
            exit(1);                                                               \
        }                                                                          \
    }
    
    inline double seconds()
    {
        struct timeval tp;
        struct timezone tzp;
        int i = gettimeofday(&tp, &tzp);
        return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
    }
    
    #endif // _COMMON_H
    
    
    
    void checkResult(float *hostRef, float *gpuRef, const int N)
    {
        double epsilon = 1.0E-8;
        bool match = 1;
    
        for (int i = 0; i < N; i++)
        {
            if (abs(hostRef[i] - gpuRef[i]) > epsilon)
            {
                match = 0;
                printf("Arrays do not match!\n");
                printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                       gpuRef[i], i);
                break;
            }
        }
    
        if (match) printf("Arrays match.\n\n");
    
        return;
    }
    
    void initialData(float *ip, int size)
    {
        // generate different seed for random number
        time_t t;
        srand((unsigned) time(&t));
    
        for (int i = 0; i < size; i++)
        {
            ip[i] = (float)( rand() & 0xFF ) / 10.0f;
        }
    
        return;
    }
    
    void sumArraysOnHost(float *A, float *B, float *C, const int N)
    {
        for (int idx = 0; idx < N; idx++)
        {
            C[idx] = A[idx] + B[idx];
        }
    }
    
    
    
    
    
    
    __global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (i < N)
        {
            C[i] = A[i] + B[i];
        }
    }
    
    
    
    int main(int argc, char **argv)
    {
        printf("%s Starting...\n", argv[0]);
    
        // set up device
        int dev = 0;
        cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, dev));
        printf("Using Device %d: %s\n", dev, deviceProp.name);
        CHECK(cudaSetDevice(dev));
    
        // set up data size of vectors
        int nElem =1000;
        printf("Vector size %d\n", nElem);
    
        // malloc host memory
        size_t nBytes = nElem * sizeof(float);
    
        float *h_A, *h_B, *hostRef, *gpuRef;
        h_A     = (float *)malloc(nBytes);
        h_B     = (float *)malloc(nBytes);
        hostRef = (float *)malloc(nBytes);
        gpuRef  = (float *)malloc(nBytes);
    
        double iStart, iElaps;
    
        // initialize data at host side
        iStart = seconds();
        initialData(h_A, nElem);
        initialData(h_B, nElem);
        iElaps = seconds() - iStart;
        printf("initialData Time elapsed %f sec\n", iElaps);
        memset(hostRef, 0, nBytes);
        memset(gpuRef,  0, nBytes);
    
        // add vector at host side for result checks
        iStart = seconds();
        sumArraysOnHost(h_A, h_B, hostRef, nElem);
        iElaps = seconds() - iStart;
        printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);
    
        // malloc device global memory
        float *d_A, *d_B, *d_C;
        CHECK(cudaMalloc((float**)&d_A, nBytes));
        CHECK(cudaMalloc((float**)&d_B, nBytes));
        CHECK(cudaMalloc((float**)&d_C, nBytes));
    
        // transfer data from host to device
        CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
    
        // invoke kernel at host side
        int iLen = 256;
        dim3 block (iLen);
        dim3 grid  ((nElem + block.x - 1) / block.x);
    
        iStart = seconds();
        sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
        CHECK(cudaDeviceSynchronize());
        iElaps = seconds() - iStart;
        printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
               block.x, iElaps);
    
        // check kernel error
        CHECK(cudaGetLastError()) ;
    
        // copy kernel result back to host side
        CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    
        // check device results
        checkResult(hostRef, gpuRef, nElem);
    
        // free device global memory
        CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_B));
        CHECK(cudaFree(d_C));
    
        // free host memory
        free(h_A);
        free(h_B);
        free(hostRef);
        free(gpuRef);
    
        return(0);
    }
```
## OUTPUT:

## Threads 512:
<img width="405" height="100" alt="pca1 1" src="https://github.com/user-attachments/assets/b4a9fccd-fd30-4dde-ae8b-e3fd8d4a5083" />

## Threads 1024:
<img width="421" height="94" alt="pca1 2" src="https://github.com/user-attachments/assets/1cda2915-ca7a-4dd4-b586-2bdf6c788e0c" />

## Threads 1023:
<img width="387" height="92" alt="pca1 3" src="https://github.com/user-attachments/assets/30993cf5-de47-46f4-ba49-829fcbd47211" />

## Threads 256:
<img width="442" height="98" alt="pca1 4" src="https://github.com/user-attachments/assets/05de8c4b-f0ae-485f-a1d7-f0a615f7214f" />


## RESULT:
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
