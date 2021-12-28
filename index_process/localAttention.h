//
// Created by zhang on 20-1-14.
//

#ifndef LOCALATTENTION_LOCALATTENTION_H
#define LOCALATTENTION_LOCALATTENTION_H

#pragma once
#include <torch/extension.h>

torch::Tensor similar_cuda_forward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_loc,
        const int kH,
        const int kW);

torch::Tensor similar_cuda_backward(
        const torch::Tensor &x,
        const torch::Tensor &grad_out,
        const int kH,
        const int kW,
        const bool is_ori);

torch::Tensor weighting_cuda_forward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_weight,
        const int kH,
        const int kW);

torch::Tensor weighting_cuda_backward_ori(
        const torch::Tensor &x_weight,
        const torch::Tensor &grad_out,
        const int kH,
        const int kW);

torch::Tensor weighting_cuda_backward_weight(
        const torch::Tensor &x_ori,
        const torch::Tensor &grad_out,
        const int kH,
        const int kW);

torch::Tensor distance_cuda_forward(
        const torch::Tensor &query,
        const torch::Tensor &key,
        const torch::Tensor &idx);


torch::Tensor query_key2attn_cuda(
        const torch::Tensor &query,
        const torch::Tensor &key,
        const torch::Tensor &idx);

torch::Tensor attn_key2query_cuda(
        const torch::Tensor &attn,
        const torch::Tensor &key,
        const torch::Tensor &idx);

torch::Tensor attn_query2key_cuda(
        const torch::Tensor &attn,
        const torch::Tensor &query,
        const torch::Tensor &idx,
        const int Nkey);

#endif //LOCALATTENTION_LOCALATTENTION_H
