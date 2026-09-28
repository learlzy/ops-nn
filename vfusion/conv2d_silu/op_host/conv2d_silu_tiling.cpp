/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv2d_silu_tiling.cpp
 * \brief Tiling for Conv2dSilu, support FP16 / BF16 / FP32
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "conv2d_silu/op_kernel/conv2d_silu_tiling_data.h"
#include "conv2d_silu/op_kernel/conv2d_silu_tiling_key.h"
#include "conv2d_silu/op_kernel/conv2d_silu_kernel_template.h"

namespace optiling {

struct Conv2dSiluCompileInfo {};

static uint64_t GetTilingKeyByDtype(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT16) {
        return static_cast<uint64_t>(CONV2D_SILU_SCH_FP16);
    }
    return static_cast<uint64_t>(CONV2D_SILU_SCH_FP32);
}

// todo: 从context中获取真实的形状和参数
template <typename T>
static size_t GetWorkspaceSize(gert::TilingContext* context)
{
    uint32_t dataSizes[5] = {2, 33, 43, 112, 80}; // {batch, hi, wi, cin, cout}
    uint8_t filterSizes[2] = {3, 3};              // {kh, kw}
    uint8_t pads[4] = {2, 2, 2, 2};               // {padLeft, padRight, padTop, padBottom}
    uint8_t strides[2] = {1, 1};                  // {strideH, strideW}
    uint8_t dilations[2] = {1, 1};                // {dilationH, dilationW}
    int32_t deviceId{0};
    Catlass::Conv2dParams problemParams = Catlass::Conv2dParams::MakeConv2dParams(
        dataSizes, filterSizes, pads, strides, dilations);
    using Conv2dSiluKernel = NsConv2dSilu::Conv2dSiluKernelTraits<T>::Conv2dKernel;
    Conv2dSiluKernel::Arguments args(problemParams, nullptr, nullptr, nullptr, nullptr);
    return Conv2dSiluKernel::GetWorkspaceSize(args);
}

static ge::graphStatus Conv2dSiluTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin Conv2dSiluTilingFunc");
    auto* xDesc = context->GetInputDesc(0);
    ge::DataType dtype = xDesc->GetDataType();
    context->SetTilingKey(GetTilingKeyByDtype(dtype));

    // 获取平台对象
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // 获取系统workspace大小
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 算子自身业务需要的workspace
    size_t usrWorkspaceSize = 0;
    if (dtype == ge::DT_FLOAT16) {
        usrWorkspaceSize = GetWorkspaceSize<half>(context);
    } else {
        usrWorkspaceSize = GetWorkspaceSize<float>(context);
    }
    // 设置总workspace大小：系统+用户之和
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = sysWorkspaceSize + usrWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForConv2dSilu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Conv2dSilu)
    .Tiling(Conv2dSiluTilingFunc)
    .TilingParse<Conv2dSiluCompileInfo>(TilingParseForConv2dSilu);

} // namespace optiling