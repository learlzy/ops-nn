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


namespace optiling {

struct Conv2dSiluCompileInfo {};

static uint64_t GetTilingKeyByDtype(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT16) {
        return static_cast<uint64_t>(CONV2D_SILU_SCH_FP16);
    }
    return static_cast<uint64_t>(CONV2D_SILU_SCH_FP32);
}

// todo
template<typename ElementType>
static size_t CalcUsrWorkspaceSize(uint32_t batch, uint32_t ho, uint32_t wo,
                                   uint32_t cout)
{
    return 0;
}

static ge::graphStatus Conv2dSiluTilingFunc(gert::TilingContext* context)
{
    
    
    // 计算 workspace
    size_t usrWorkspaceSize = 0;
    if (dtype == ge::DT_FLOAT16) {
        
    } else {
        
    }
    tilingData->workspaceSize = static_cast<uint64_t>(usrWorkspaceSize);

    // tiling key
    context->SetTilingKey(GetTilingKeyByDtype(dtype));

    // block dim
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