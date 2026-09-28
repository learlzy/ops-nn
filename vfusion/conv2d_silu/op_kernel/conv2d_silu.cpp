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
 * \file conv2d_silu.cpp
 * \brief Kernel entry: Conv2d + Bias + SiLU (Catlass)
 *        schMode: 0=FP16, 1=BF16, 2=FP32
 */

#include "conv2d_silu.h"

template <uint32_t schMode>
__global__ __aicore__ void conv2d_silu(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Conv2dSiluTilingData);
    GET_TILING_DATA_WITH_STRUCT(Conv2dSiluTilingData, tilingData, tiling);

    if constexpr (schMode == CONV2D_SILU_SCH_FP16) {
        
    } else if constexpr (schMode == CONV2D_SILU_SCH_FP32) {
        
    }
}