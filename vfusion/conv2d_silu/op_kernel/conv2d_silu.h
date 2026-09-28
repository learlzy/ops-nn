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
 * \file conv2d_silu.h
 * \brief Catlass-based Conv2d + Bias + SiLU kernel (Atlas A2)
 *        Support Element type: half / bfloat16_t / float
 */

#ifndef __CONV2D_SILU_H__
#define __CONV2D_SILU_H__

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "conv2d_silu_tiling_data.h"
#include "conv2d_silu_tiling_key.h"
#include "conv2d_silu_kernel_template.h"


namespace NsConv2dSilu {

template<typename ElementType>
struct Conv2dSiluKernel : Conv2dSiluKernelTraits<ElementType> {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
    const Conv2dSiluTilingData* tiling) 
    {
        // 告诉框架这是系统 workspace（含硬件同步相关区域）
        SetSysWorkspace(workspace);
        // 用户自己用的中间结果区域
        GM_ADDR userWs = GetUserWorkspace(workspace);

        GET_TILING_DATA(tilingData, tiling);
    }

    __aicore__ inline void Process()
    {}

private:
    __aicore__ inline Catlass::Conv2dParams MakeProblemParams(const Conv2dSiluTilingData* t)
    {
        uint32_t dataSizes[5] = {t->batch, t->hi, t->wi, t->cin, t->cout};
        uint8_t filterSizes[2] = {static_cast<uint8_t>(t->kh), static_cast<uint8_t>(t->kw)};
        uint8_t pads[4] = {
            static_cast<uint8_t>(t->padLeft), static_cast<uint8_t>(t->padRight),
            static_cast<uint8_t>(t->padTop), static_cast<uint8_t>(t->padBottom)};
        uint8_t strides[2] = {static_cast<uint8_t>(t->strideH), static_cast<uint8_t>(t->strideW)};
        uint8_t dilations[2] = {static_cast<uint8_t>(t->dilationH), static_cast<uint8_t>(t->dilationW)};
        return Catlass::Conv2dParams::MakeConv2dParams(dataSizes, filterSizes, pads, strides, dilations);
    }
};

} // namespace NsConv2dSilu

#endif