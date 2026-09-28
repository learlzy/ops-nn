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
 * \file conv2d_silu_infershape.cpp
 * \brief InferShape for Conv2dSilu (NCHW origin)
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr size_t IDX_X = 0;
static constexpr size_t IDX_FILTER = 1;
static constexpr size_t IDX_Y = 0;
static constexpr size_t NCHW_N = 0;
static constexpr size_t NCHW_C = 1;
static constexpr size_t NCHW_H = 2;
static constexpr size_t NCHW_W = 3;

static ge::graphStatus InferShapeConv2dSilu(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShapeConv2dSilu");

    const gert::Shape* xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* filterShape = context->GetInputShape(IDX_FILTER);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterShape);
    gert::Shape* yShape = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (xShape->GetDimNum() != 4 || filterShape->GetDimNum() != 4) {
        OP_LOGE(context->GetNodeName(), "x and filter must be 4-D NCHW");
        return GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const gert::ContinuousVector* stridesPtr = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    const gert::ContinuousVector* padsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    const gert::ContinuousVector* dilationsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(2);
    const int64_t* groupsPtr = attrs->GetAttrPointer<int64_t>(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, dilationsPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupsPtr);

    const int64_t* strides = reinterpret_cast<const int64_t*>(stridesPtr->GetData());
    const int64_t* pads = reinterpret_cast<const int64_t*>(padsPtr->GetData());
    const int64_t* dilations = reinterpret_cast<const int64_t*>(dilationsPtr->GetData());

    int64_t strideH = (stridesPtr->GetSize() >= 3) ? strides[2] : strides[0];
    int64_t strideW = (stridesPtr->GetSize() >= 4) ? strides[3] : strides[1];
    int64_t dilH = (dilationsPtr->GetSize() >= 3) ? dilations[2] : dilations[0];
    int64_t dilW = (dilationsPtr->GetSize() >= 4) ? dilations[3] : dilations[1];

    int64_t padTop = 0;
    int64_t padBottom = 0;
    int64_t padLeft = 0;
    int64_t padRight = 0;
    if (padsPtr->GetSize() >= 4) {
        padTop = pads[0];
        padBottom = pads[1];
        padLeft = pads[2];
        padRight = pads[3];
    }

    int64_t N = xShape->GetDim(NCHW_N);
    int64_t Hin = xShape->GetDim(NCHW_H);
    int64_t Win = xShape->GetDim(NCHW_W);
    int64_t Cout = filterShape->GetDim(0);
    int64_t Kh = filterShape->GetDim(2);
    int64_t Kw = filterShape->GetDim(3);

    int64_t effectiveKh = dilH * (Kh - 1) + 1;
    int64_t effectiveKw = dilW * (Kw - 1) + 1;
    int64_t Hout = (Hin + padTop + padBottom - effectiveKh) / strideH + 1;
    int64_t Wout = (Win + padLeft + padRight - effectiveKw) / strideW + 1;

    if (Hout <= 0 || Wout <= 0) {
        OP_LOGE(context->GetNodeName(), "invalid Hout/Wout: %ld, %ld", Hout, Wout);
        return GRAPH_FAILED;
    }

    yShape->SetDimNum(4);
    yShape->SetDim(NCHW_N, N);
    yShape->SetDim(NCHW_C, Cout);
    yShape->SetDim(NCHW_H, Hout);
    yShape->SetDim(NCHW_W, Wout);

    OP_LOGD(context->GetNodeName(), "y=[%ld,%ld,%ld,%ld]", N, Cout, Hout, Wout);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Conv2dSilu).InferShape(InferShapeConv2dSilu);

} // namespace ops