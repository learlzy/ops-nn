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
 * \brief Tiling for Conv2dSilu, support FP16 / FP32
 *        直接使用 Conv2dEpilogue::GetWorkspaceSize，不使用 Adapter
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_ascendc.h"
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

template <typename ElementType, typename ElementAccumulator>
static size_t CalcWorkspaceSize(const Catlass::Conv2dParams& problemParams)
{
    using Conv2dKernel = typename NsConv2dSilu::Conv2dSiluKernelTraits<ElementType, ElementAccumulator>::Conv2dKernel;
    typename Conv2dKernel::Arguments args{problemParams, nullptr, nullptr, nullptr, nullptr};
    return Conv2dKernel::GetWorkspaceSize(args);
}

static ge::graphStatus Conv2dSiluTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin Conv2dSiluTilingFunc");

    auto* xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ge::DataType dtype = xDesc->GetDataType();

    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* filterShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterShape);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const gert::ContinuousVector* stridesPtr = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    const gert::ContinuousVector* padsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    const gert::ContinuousVector* dilationsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, dilationsPtr);

    const int64_t* strides = reinterpret_cast<const int64_t*>(stridesPtr->GetData());
    const int64_t* pads = reinterpret_cast<const int64_t*>(padsPtr->GetData());
    const int64_t* dilations = reinterpret_cast<const int64_t*>(dilationsPtr->GetData());

    int64_t strideH = (stridesPtr->GetSize() >= 3) ? strides[2] : strides[0];
    int64_t strideW = (stridesPtr->GetSize() >= 4) ? strides[3] : strides[1];
    int64_t dilH = (dilationsPtr->GetSize() >= 3) ? dilations[2] : dilations[0];
    int64_t dilW = (dilationsPtr->GetSize() >= 4) ? dilations[3] : dilations[1];

    int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
    if (padsPtr->GetSize() >= 4) {
        padTop = pads[0];
        padBottom = pads[1];
        padLeft = pads[2];
        padRight = pads[3];
    }

    const auto& xDims = xShape->GetStorageShape();
    const auto& fDims = filterShape->GetStorageShape();

    uint32_t batch = static_cast<uint32_t>(xDims.GetDim(0));
    uint32_t hi = static_cast<uint32_t>(xDims.GetDim(2));
    uint32_t wi = static_cast<uint32_t>(xDims.GetDim(3));
    uint32_t cin = static_cast<uint32_t>(xDims.GetDim(1));
    uint32_t cout = static_cast<uint32_t>(fDims.GetDim(0));
    uint32_t kh = static_cast<uint32_t>(fDims.GetDim(2));
    uint32_t kw = static_cast<uint32_t>(fDims.GetDim(3));

    uint32_t dataSizes[5] = {batch, hi, wi, cin, cout};
    uint8_t filterSizes[2] = {static_cast<uint8_t>(kh), static_cast<uint8_t>(kw)};
    uint8_t padsArr[4] = {
        static_cast<uint8_t>(padLeft),
        static_cast<uint8_t>(padRight),
        static_cast<uint8_t>(padTop),
        static_cast<uint8_t>(padBottom)
    };
    uint8_t stridesArr[2] = {static_cast<uint8_t>(strideH), static_cast<uint8_t>(strideW)};
    uint8_t dilationsArr[2] = {static_cast<uint8_t>(dilH), static_cast<uint8_t>(dilW)};

    Catlass::Conv2dParams problemParams =
        Catlass::Conv2dParams::MakeConv2dParams(dataSizes, filterSizes, padsArr, stridesArr, dilationsArr);

    // 填充 tiling data
    Conv2dSiluTilingData* tilingData = context->GetTilingData<Conv2dSiluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    tilingData->batch = batch;
    tilingData->hi = hi;
    tilingData->wi = wi;
    tilingData->cin = cin;
    tilingData->cout = cout;
    tilingData->kh = kh;
    tilingData->kw = kw;
    tilingData->padLeft = static_cast<uint32_t>(padLeft);
    tilingData->padRight = static_cast<uint32_t>(padRight);
    tilingData->padTop = static_cast<uint32_t>(padTop);
    tilingData->padBottom = static_cast<uint32_t>(padBottom);
    tilingData->strideH = static_cast<uint32_t>(strideH);
    tilingData->strideW = static_cast<uint32_t>(strideW);
    tilingData->dilationH = static_cast<uint32_t>(dilH);
    tilingData->dilationW = static_cast<uint32_t>(dilW);

    tilingData->ho = problemParams.ho();
    tilingData->wo = problemParams.wo();
    tilingData->cin1 = problemParams.cin1();
    tilingData->cout1 = problemParams.cout1();
    tilingData->coutRound = problemParams.coutRound();
    tilingData->c0 = problemParams.C0;

    // 计算 workspace
    size_t usrWorkspaceSize = 0;
    if (dtype == ge::DT_FLOAT16) {
        usrWorkspaceSize = CalcWorkspaceSize<half, half>(problemParams);
    } else {
        usrWorkspaceSize = CalcWorkspaceSize<float, float>(problemParams);
    }
    tilingData->workspaceSize = static_cast<uint64_t>(usrWorkspaceSize);

    // tiling key
    context->SetTilingKey(GetTilingKeyByDtype(dtype));

    // block dim（简单按核数设置，实际可更精细）
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);

    // workspace：系统 + 用户
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = sysWorkspaceSize + usrWorkspaceSize;

    OP_LOGD(context->GetNodeName(),
            "Tiling done: batch=%u hi=%u wi=%u cin=%u cout=%u ho=%u wo=%u workspace=%lu",
            tilingData->batch, tilingData->hi, tilingData->wi, tilingData->cin, tilingData->cout,
            tilingData->ho, tilingData->wo, tilingData->workspaceSize);

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