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
 * \file conv2d_silu_tiling_data.h
 * \brief Tiling data for Conv2dSilu
 */

#ifndef __CONV2D_SILU_TILING_DATA_H__
#define __CONV2D_SILU_TILING_DATA_H__

#include <cstdint>

// #pragma pack(push, 8)
struct Conv2dSiluTilingData {
    // problem shape
    uint32_t batch;
    uint32_t hi;
    uint32_t wi;
    uint32_t cin;
    uint32_t cout;

    uint32_t kh;
    uint32_t kw;

    uint32_t padLeft;
    uint32_t padRight;
    uint32_t padTop;
    uint32_t padBottom;

    uint32_t strideH;
    uint32_t strideW;

    uint32_t dilationH;
    uint32_t dilationW;

    // derived
    uint32_t ho;
    uint32_t wo;
    uint32_t cin1;
    uint32_t cout1;
    uint32_t coutRound;
    uint32_t c0;

    // workspace size (user workspace, bytes)
    uint64_t workspaceSize;
};
// #pragma pack(pop)

#endif // __CONV2D_SILU_TILING_DATA_H__