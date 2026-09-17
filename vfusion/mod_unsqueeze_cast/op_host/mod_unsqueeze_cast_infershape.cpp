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
 * \file mod_unsqueeze_cast_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeModUnsqueezeCast(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeModUnsqueezeCast");

    const gert::Shape* x1_shape = context->GetInputShape(IDX_0);
    gert::Shape* y_shape = context->GetOutputShape(IDX_0);

    const size_t rank = x1_shape->GetDimNum();
    y_shape->SetDimNum(rank + 1);
    for (size_t i = 0; i < rank; ++i) {
        y_shape->SetDim(i, x1_shape->GetDim(i));
    }
    y_shape->SetDim(rank, 1);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeModUnsqueezeCast");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeModUnsqueezeCast(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeModUnsqueezeCast");

    // 设置输出的数据类型
    // ModUnsqueezeCast算子的输出数据类型固定为float16
    const ge::DataType sizeDtype = ge::DT_FLOAT16;
    context->SetOutputDataType(IDX_0, sizeDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeModUnsqueezeCast");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ModUnsqueezeCast).InferShape(InferShapeModUnsqueezeCast).InferDataType(InferDataTypeModUnsqueezeCast);
} // namespace ops