/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



//Based on http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm



#include <assert.h>
#include <helper_cuda.h>
#include "sortingNetworks_common.h"
#include "sortingNetworks_common.cuh"

#define DEVICE_MEM 1048576


////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
__global__ void bitonicSortShared(
        uint *d_DstKey,
        uint *d_SrcKey,
        uint arrayLength,
        uint dir
        )
{
    //Shared memory storage for one or more short vectors
    __shared__ uint s_key[SHARED_SIZE_LIMIT];

    //Offset to the beginning of subbatch and load data
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size < arrayLength; size <<= 1)
    {
        //Bitonic merge
        uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

        for (uint stride = size / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                    s_key[pos +      0], 
                    s_key[pos + stride],
                    ddd
                    );
        }
    }

    //ddd == dir for the last bitonic merge step
    {
        for (uint stride = arrayLength / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                    s_key[pos +      0],
                    s_key[pos + stride],
                    dir
                    );
        }
    }

    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}



////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////
//Bottom-level bitonic sort
//Almost the same as bitonicSortShared with the exception of
//even / odd subarrays being sorted in opposite directions
//Bitonic merge accepts both
//Ascending | descending or descending | ascending sorted pairs
__global__ void bitonicSortShared1(
        uint *d_DstKey,
        uint *d_SrcKey
        )
{
    //Shared memory storage for current subarray
    __shared__ uint s_key[SHARED_SIZE_LIMIT];

    //Offset to the beginning of subarray and load data
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1)
    {
        //Bitonic merge
        uint ddd = (threadIdx.x & (size / 2)) != 0;

        for (uint stride = size / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                    s_key[pos +      0],
                    s_key[pos + stride],
                    ddd
                    );
        }
    }

    //Odd / even arrays of SHARED_SIZE_LIMIT elements
    //sorted in opposite directions
    uint ddd = blockIdx.x & 1;
    {
        for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                    s_key[pos +      0],
                    s_key[pos + stride],
                    ddd
                    );
        }
    }


    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

//Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
__global__ void bitonicMergeGlobal(
        uint *d_DstKey,
        uint *d_SrcKey,
        uint arrayLength,
        uint size,
        uint stride,
        uint dir
        )
{
    uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
    uint        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

    //Bitonic merge
    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

    uint keyA = d_SrcKey[pos +      0];
    uint keyB = d_SrcKey[pos + stride];

    Comparator(
            keyA,
            keyB,
            ddd
            );

    d_DstKey[pos +      0] = keyA;
    d_DstKey[pos + stride] = keyB;
}

//Combined bitonic merge steps for
//size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__global__ void bitonicMergeShared(
        uint *d_DstKey,
        uint *d_SrcKey,
        uint arrayLength,
        uint size,
        uint dir
        )
{
    //Shared memory storage for current subarray
    __shared__ uint s_key[SHARED_SIZE_LIMIT];

    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];

    //Bitonic merge
    uint comparatorI = UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);

    for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(
                s_key[pos +      0],
                s_key[pos + stride],
                ddd
                );
    }

    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}



////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Helper function (also used by odd-even merge sort)
extern "C" uint factorRadix2(uint *log2L, uint L)
{
    if (!L)
    {
        *log2L = 0;
        return 0;
    }
    else
    {
        for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++);

        return L;
    }
}

extern "C" uint bitonicSort(
        uint *d_DstKey,
        uint *d_SrcKey,
        uint *h_SrcKey,
        uint arrayLength,
        uint dir
        )
{
    //Nothing to sort
    if (arrayLength < 2)
        return 0;

    //Only power-of-two array lengths are supported by this implementation
    uint log2L;
    uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
    assert(factorizationRemainder == 1);

    dir = (dir != 0);

    uint blockCount, threadCount;
    uint flag;
    cudaError_t err;
    if(arrayLength > DEVICE_MEM){
        blockCount = DEVICE_MEM / SHARED_SIZE_LIMIT;
        threadCount = SHARED_SIZE_LIMIT / 2;
        flag = 1;
    }
    else if (arrayLength <= DEVICE_MEM && arrayLength > SHARED_SIZE_LIMIT){
        blockCount = arrayLength  / SHARED_SIZE_LIMIT;
        threadCount = SHARED_SIZE_LIMIT / 2;
        flag = 2;
    }

    else{
        blockCount = 1;
        threadCount = SHARED_SIZE_LIMIT / 2;
        flag = 0;
    }

    if (flag == 0){
        err = cudaMemcpy(d_SrcKey, h_SrcKey, arrayLength * sizeof(uint), cudaMemcpyHostToDevice);
        checkCudaErrors(err);
        bitonicSortShared<<<blockCount, threadCount>>>(d_DstKey, d_SrcKey, arrayLength, dir);
        err = cudaMemcpy(h_SrcKey, d_DstKey, arrayLength * sizeof(uint), cudaMemcpyDeviceToHost);
        checkCudaErrors(err);
    }
    else
    {
        for(uint i = 0; i * DEVICE_MEM < arrayLength ; i++){
            uint copy_size = (flag == 1) ? DEVICE_MEM : arrayLength;  
            err = cudaMemcpy(d_SrcKey, h_SrcKey + i * DEVICE_MEM, copy_size * sizeof(uint), cudaMemcpyHostToDevice);
            checkCudaErrors(err);
            bitonicSortShared1<<<blockCount, threadCount>>>(d_DstKey, d_SrcKey);
            err = cudaMemcpy(h_SrcKey + i * DEVICE_MEM, d_DstKey, copy_size * sizeof(uint), cudaMemcpyDeviceToHost);
            checkCudaErrors(err);
        }
        uint start_size = (flag == 2) ? arrayLength : DEVICE_MEM;
        for(uint size = start_size; size <= arrayLength; size <<= 1){
            for (unsigned stride = size / 2; stride > 0; stride >>= 1){
                if (stride >= DEVICE_MEM){ // Comparing/Merging discontinous subarrays seperated by a distance greater than 2^20
                    for (uint i = 0; i < arrayLength / DEVICE_MEM; i++){
                        err = cudaMemcpy(d_SrcKey, h_SrcKey + i*DEVICE_MEM/2 + (i*(DEVICE_MEM/2)/stride)*stride, DEVICE_MEM/2*sizeof(uint), cudaMemcpyHostToDevice);
                        checkCudaErrors(err);
                        err = cudaMemcpy(d_SrcKey + DEVICE_MEM/2, h_SrcKey + i*DEVICE_MEM/2 + (i*(DEVICE_MEM/2)/stride+1)*stride, DEVICE_MEM/2*sizeof(uint), cudaMemcpyHostToDevice);
                        uint ddd = dir ^ ((i*DEVICE_MEM/size) & 1);
                        bitonicMergeGlobal<<<(blockCount * threadCount) /256, 256>>>(d_DstKey, d_SrcKey, arrayLength, size, DEVICE_MEM/2, ddd);
                        err = cudaMemcpy(h_SrcKey + i*DEVICE_MEM/2 + (i*(DEVICE_MEM/2)/stride)*stride, d_DstKey, DEVICE_MEM/2 * sizeof(uint), cudaMemcpyDeviceToHost);
                        checkCudaErrors(err);
                        err = cudaMemcpy(h_SrcKey + i*DEVICE_MEM/2 + (i*(DEVICE_MEM/2)/stride+1)*stride, d_DstKey+DEVICE_MEM/2, DEVICE_MEM/2 * sizeof(uint), cudaMemcpyDeviceToHost);
                        checkCudaErrors(err);

                    }
                }
                else{
                    for(uint i = 0; i * DEVICE_MEM < arrayLength ; i++){ //Moving DEVICE_MEM to the left side of the condition so that the first iteration is gauranteed to be executed
                        err = cudaMemcpy(d_DstKey, h_SrcKey + i*DEVICE_MEM, start_size * sizeof(uint), cudaMemcpyHostToDevice);
                        checkCudaErrors(err);
                        uint ddd = dir ^ ((i*DEVICE_MEM/size) & 1);
                        uint start = 2 * stride;
                        if(size == start_size){
                            start = 2 * SHARED_SIZE_LIMIT; // for the 1st iteration, the size of bitonic sequence is 2048
                        }
                        for(uint sz = start; sz <= start_size; sz <<= 1){ 
                            for(unsigned sd = sz / 2; sd > 0; sd >>=1){
                                if(sd >= SHARED_SIZE_LIMIT){
                                    bitonicMergeGlobal<<<(blockCount*threadCount)/ 256,256>>>(d_DstKey, d_DstKey, arrayLength, sz, sd, ddd);
                                }
                                else{
                                    bitonicMergeShared<<<blockCount, threadCount>>>(d_DstKey, d_DstKey, arrayLength, sz, ddd);
                                    break;
                                }
                            }
                        }
                        err = cudaMemcpy(h_SrcKey + i*DEVICE_MEM, d_DstKey, start_size*sizeof(uint), cudaMemcpyDeviceToHost);
                        checkCudaErrors(err);
                    }
                    break;
                }
            }
        }
    }

    return threadCount; 
}
