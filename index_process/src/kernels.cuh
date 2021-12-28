#include "utils.cuh"
#include <math.h>

template<typename dt, typename dtc>
__global__ void cc2k(
        const dt *x_ori,
        const dt *x_loc,
        const int kH,
        const int kW,
        const int rH,
        const int rW,
        const int patch,
        const int channels,
        const int height,
        const int width,
        const int per_channel,
        dt *y
) {
    // x_ori, x_loc: {c, h, w}
    // y: {h, w, k^2}
    for (int indexO = blockIdx.x; indexO < per_channel; indexO += gridDim.x) {
        const int w_ori = indexO % width - rW;
        const int h_ori = indexO / width - rH;

        KERNEL_LOOP(indexK, patch) {
            const int w = w_ori + indexK % kW;
            const int h = h_ori + indexK / kW;
            dtc val = dtc(0);

            if (h > -1 && h < height && w > -1 && w < width) {
                const dt *p_ori = x_ori + indexO;
                const dt *p_loc = x_loc + h * width + w;
                for (int c = 0; c < channels; ++c) {
                    val += static_cast<dtc> (__ldg(p_ori) * __ldg(p_loc));
                    p_ori += per_channel;
                    p_loc += per_channel;
                }
            }
            y[indexO * patch + indexK] = static_cast<dt> (val);
        }
    }
}



template<typename dt, typename dtc>
__global__ void ck2c_ori(
        const dt *x_loc,
        const dt *x_weight,
        const int kH,
        const int kW,
        const int rH,
        const int rW,
        const int patch,
        const int height,
        const int width,
        const int per_channel,
        const int per_inp,
        dt *y
) {
    // x_loc: {c, h, w}
    // x_weight: {h, w, k^2}
    // y: {c, h, w}
    KERNEL_LOOP1d(index, per_inp) {
        const int index_ = index % per_channel;
        const int w_ori = index_ % width - rW;
        const int h_ori = index_ / width - rH;
        const dt *p_weight = x_weight + index_ * patch;
        const dt *p_loc = x_loc + index - index_;
        dtc val = dtc(0);

        for (int indexK = 0; indexK < patch; ++indexK) {
            const int w = w_ori + indexK % kW;
            const int h = h_ori + indexK / kW;
            if (h > -1 && h < height && w > -1 && w < width) {
                val += static_cast<dtc> (__ldg(p_loc + width * h + w) *
                        __ldg(p_weight + indexK));
            }
        }
        y[index] = static_cast<dt> (val);
    }
}

template<typename dt, typename dtc>
__global__ void ck2c_loc(
        const dt *x_ori,
        const dt *x_weight,
        const int kH,
        const int kW,
        const int rH,
        const int rW,
        const int patch,
        const int height,
        const int width,
        const int per_channel,
        const int per_inp,
        dt *y
) {
    // x_ori: {c, h, w}
    // x_weight: {h, w, k^2}
    // y: {c, h, w}
    KERNEL_LOOP1d(index, per_inp) {
        const int index_ = index % per_channel;
        const int w_ori = index_ % width + rW;
        const int h_ori = index_ / width + rH;
        const dt *p_ori = x_ori + index - index_;
        dtc val = dtc(0);

        for (int indexK = 0; indexK < patch; ++indexK) {
            const int w = w_ori - indexK % kW;
            const int h = h_ori - indexK / kW;
            const int indexW = width * h + w;

            if (h > -1 && h < height && w > -1 && w < width) {
                val += static_cast<dtc> (__ldg(p_ori + indexW) *
                        __ldg(x_weight + indexW * patch + indexK));
            }
        }
        y[index] = static_cast<dt> (val);
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

template<typename dt, typename dtc>
void f_cc2k(
        cudaStream_t stream,
        const dt *x_ori,
        const dt *x_loc,
        const int kH,
        const int kW,
        const int rH,
        const int rW,
        const int patch,
        const int channels,
        const int height,
        const int width,
        const int per_channel,
        dt *y) {
    cc2k<dt, dtc> <<< min(per_channel, MAX_PIXELS_2d), CUDA_NUM_THREADS, 0, stream >>> (
            x_ori, x_loc,
                    kH, kW, rH, rW,
                    patch, channels,
                    height, width, per_channel,
                    y);
}

template<typename dt, typename dtc>
void f_ck2c_ori(
        cudaStream_t stream,
        const dt *x_loc,
        const dt *x_weight,
        const int kH,
        const int kW,
        const int rH,
        const int rW,
        const int patch,
        const int channels,
        const int height,
        const int width,
        const int per_channel,
        const int per_inp,
        dt *y) {
    ck2c_ori<dt, dtc> <<< GET_BLOCKS(min(per_inp, MAX_PIXELS_3d)), CUDA_NUM_THREADS, 0, stream >>> (
            x_loc, x_weight,
                    kH, kW, rH, rW,
                    patch, height, width,
                    per_channel, per_inp,
                    y);

}

template<typename dt, typename dtc>
void f_ck2c_loc(
        cudaStream_t stream,
        const dt *x_ori,
        const dt *x_weight,
        const int kH,
        const int kW,
        const int rH,
        const int rW,
        const int patch,
        const int channels,
        const int height,
        const int width,
        const int per_channel,
        const int per_inp,
        dt *y) {
    ck2c_loc<dt, dtc> <<< GET_BLOCKS(min(per_inp, MAX_PIXELS_3d)), CUDA_NUM_THREADS, 0, stream >>> (
            x_ori, x_weight,
                    kH, kW, rH, rW,
                    patch, height, width,
                    per_channel, per_inp,
                    y);
}




//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


template<typename dt, typename dtc>
__global__ void mydist(
        const dt *query,
        const dt *key,
        const int *idx,
        const int batch,
        const int Nquery,
        const int Nkey,
        const int kernel,
        const int channels,
        dt *y) {
    // query: {batch, Nquery, channels}
    // key: {batch, Nkey, channels}
    // idx: {batch, Nquery, kernel}
    // y: {batch, Nquery, kernel}

    int idx_begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int idx_end = batch * Nquery * kernel;

    for (int idx_all=idx_begin; idx_all < idx_end; idx_all += stride)
    {
        int tmp = idx_all;
        int idx_K = tmp % kernel;
        tmp = tmp / kernel;
        int idx_N = tmp % Nquery;
        tmp = tmp / Nquery;
        int idx_B = tmp;

        const int idx_M = __ldg(&idx[idx_all]);
        dtc val = dtc(0);
        const dt *p_query = query + idx_B * Nquery * channels + idx_N * channels;
        const dt *p_key = key + idx_B * Nkey * channels + idx_M * channels;
        for (int c = 0; c < channels; ++c) {
            dtc dif = static_cast<dtc> (__ldg(p_query) - __ldg(p_key));
            val += dif * dif;
            p_query += 1;
            p_key += 1;
        }
        val = sqrt(val);
        y[idx_all] = static_cast<dt> (val);


    }
}





template<typename dt, typename dtc>
void f_mydist(
        cudaStream_t stream,
        const dt *query,
        const dt *key,
        const int *idx,
        const int batch,
        const int Nquery,
        const int Nkey,
        const int kernel,
        const int channels,
        dt *y) {

        dim3 blockSize(CUDA_NUM_THREADS);
        dim3 gridSize( min( (batch*Nquery*kernel+blockSize.x-1)/blockSize.x, MAX_PIXELS_2d) );

        mydist<dt, dtc> <<< gridSize, blockSize, 0, stream >>> (
            query, key, idx,
            batch, Nquery, Nkey, kernel, channels,
            y);
}




// forward from query, key to attn
template<typename dt, typename dtc>
__global__ void qk2attn(
        const dt *query,
        const dt *key,
        const int *idx,
        const int batch,
        const int Nquery,
        const int Nkey,
        const int kernel,
        const int channels,
        dt *y) {
    // query: {batch, Nquery, channels}
    // key: {batch, Nkey, channels}
    // idx: {batch, Nquery, kernel}
    // y: {batch, Nquery, kernel}

    int idx_begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int idx_end = batch * Nquery * kernel;

    for (int idx_all=idx_begin; idx_all < idx_end; idx_all += stride)
    {
        int tmp = idx_all;
        int idx_K = tmp % kernel;
        tmp = tmp / kernel;
        int idx_N = tmp % Nquery;
        tmp = tmp / Nquery;
        int idx_B = tmp;

        const int idx_M = __ldg(&idx[idx_all]);
        dtc val = dtc(0);
        const dt *p_query = query + idx_B * Nquery * channels + idx_N * channels;
        const dt *p_key = key + idx_B * Nkey * channels + idx_M * channels;
        for (int c = 0; c < channels; ++c) {
            dtc mul = static_cast<dtc> (__ldg(p_query) * __ldg(p_key));
            val += mul;
            p_query += 1;
            p_key += 1;
        }
        y[idx_all] = static_cast<dt> (val);
    }
}



template<typename dt, typename dtc>
void f_qk2attn(
        cudaStream_t stream,
        const dt *query,
        const dt *key,
        const int *idx,
        const int batch,
        const int Nquery,
        const int Nkey,
        const int kernel,
        const int channels,
        dt *y) {

        dim3 blockSize(CUDA_NUM_THREADS);
        dim3 gridSize( min( (batch*Nquery*kernel+blockSize.x-1)/blockSize.x, MAX_PIXELS_2d) );

        qk2attn<dt, dtc> <<< gridSize, blockSize, 0, stream >>> (
            query, key, idx,
            batch, Nquery, Nkey, kernel, channels,
            y);
}



// backward from attn, key to query
// FLOPs: B * N * K * C
// parallel in channels B, N, C, loop in channels K
template<typename dt, typename dtc>
__global__ void attn_key2query(
        const dt *attn,
        const dt *key,
        const int *idx,
        const int batch,
        const int Nquery,
        const int Nkey,
        const int kernel,
        const int channels,
        dt *query) {
    // attn: {batch, Nquery, kernel}
    // key: {batch, Nkey, channels}
    // idx: {batch, Nquery, kernel}
    // query: {batch, Nquery, channels}

    int idx_begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int idx_end = batch * Nquery * channels;

    for (int idx_all=idx_begin; idx_all < idx_end; idx_all += stride)
    {
        int tmp = idx_all;
        int idx_C = tmp % channels;
        tmp = tmp / channels;
        int idx_N = tmp % Nquery;
        tmp = tmp / Nquery;
        int idx_B = tmp;

        const dt *p_attn = attn + idx_B * Nquery * kernel + idx_N * kernel + 0;
        const int *p_idx_M = idx + idx_B * Nquery * kernel + idx_N * kernel + 0;

        dtc val = dtc(0);
        for (int idx_K = 0; idx_K < kernel; ++idx_K) {
            const dt *p_key = key + idx_B * Nkey * channels + __ldg(p_idx_M) * channels + idx_C;
            val += static_cast<dtc> (__ldg(p_attn) * __ldg(p_key));
            p_attn += 1;
            p_idx_M += 1;
        }
        query[idx_B * Nquery * channels + idx_N * channels + idx_C] = static_cast<dt> (val);
    }
}



template<typename dt, typename dtc>
void f_attn_key2query(
        cudaStream_t stream,
        const dt *attn,
        const dt *key,
        const int *idx,
        const int batch,
        const int Nquery,
        const int Nkey,
        const int kernel,
        const int channels,
        dt *query) {

        dim3 blockSize(CUDA_NUM_THREADS);
        dim3 gridSize( min( (batch*Nquery*channels+blockSize.x-1)/blockSize.x, MAX_PIXELS_2d) );

        attn_key2query<dt, dtc> <<< gridSize, blockSize, 0, stream >>> (
            attn, key, idx,
            batch, Nquery, Nkey, kernel, channels,
            query);
}


// backward from attn (query) to key
// FLOPs: B * N * K * C
// in thery, can only be paralleled in B, C dims
// in practice: parallel in dims B, N, C, loop in channels K, use atomicAdd
template<typename dt, typename dtc>
__global__ void attn_query2key(
        const dt *attn,
        const dt *query,
        const int *idx,
        const int batch,
        const int Nquery,
        const int Nkey,
        const int kernel,
        const int channels,
        dt *key) {
    // attn: {batch, Nquery, kernel}
    // query: {batch, Nquery, channels}
    // idx: {batch, Nquery, kernel}
    // key: {batch, Nkey, channels}

    int idx_begin = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int idx_end = batch * Nquery * channels;

    for (int idx_all=idx_begin; idx_all < idx_end; idx_all += stride)
    {
        // idx_all(B, N, C): idx_B * Nquery * channels + idx_N * channels + idx_C
        int tmp = idx_all;
        int idx_C = tmp % channels;
        tmp = tmp / channels;
        int idx_N = tmp % Nquery;
        tmp = tmp / Nquery;
        int idx_B = tmp;

        // const dt *p_query = query + idx_all;
        const dt q = __ldg(query + idx_all);
        const dt *p_attn = attn + idx_B * Nquery * kernel + idx_N * kernel + 0;
        const int *p_idx_M = idx + idx_B * Nquery * kernel + idx_N * kernel + 0;

        for (int idx_K = 0; idx_K < kernel; ++idx_K) {
            // dtc val = static_cast<dtc> (__ldg(p_attn) * __ldg(p_query));
            dtc val = static_cast<dtc> (__ldg(p_attn) * q);

            dt *p_key = key + idx_B * Nkey * channels + __ldg(p_idx_M) * channels + idx_C;

            atomicAdd(p_key, val);

            p_attn += 1;
            p_idx_M += 1;

        }
    }
}



template<typename dt, typename dtc>
void f_attn_query2key(
        cudaStream_t stream,
        const dt *attn,
        const dt *query,
        const int *idx,
        const int batch,
        const int Nquery,
        const int Nkey,
        const int kernel,
        const int channels,
        dt *key) {

        dim3 blockSize(CUDA_NUM_THREADS);
        dim3 gridSize( min( (batch*Nquery*channels+blockSize.x-1)/blockSize.x, MAX_PIXELS_2d) );

        attn_query2key<dt, dtc> <<< gridSize, blockSize, 0, stream >>> (
            attn, query, idx,
            batch, Nquery, Nkey, kernel, channels,
            key);
}


////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

// realization of weighted sum

// Forward: can use f_attn_key2query, the process is totally the same.

// Backward from out to attn_weight: can use f_qk2attn, the process is totally the same.

// Backward from out to value: can use f_attn_query2key

