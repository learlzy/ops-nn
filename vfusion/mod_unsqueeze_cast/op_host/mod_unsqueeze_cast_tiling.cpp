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
 * \file mod_unsqueeze_cast_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "mod_unsqueeze_cast/op_kernel/mod_unsqueeze_cast_tiling_data.h"
#include "mod_unsqueeze_cast/op_kernel/mod_unsqueeze_cast_tiling_key.h"

namespace optiling {

struct ModUnsqueezeCastCompileInfo {};

// tiling 分发入口
static ge::graphStatus ModUnsqueezeCastTilingFunc(gert::TilingContext* context)
{
    ModUnsqueezeCastTilingData *tiling = context->GetTilingData<ModUnsqueezeCastTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);

    int64_t data_sz = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); ++i) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }
    tiling->size = static_cast<uint32_t>(data_sz);

    constexpr uint32_t BLOCK_DIM = 4;
    context->SetBlockDim(BLOCK_DIM);
    tiling->blockLength = (tiling->size + BLOCK_DIM - 1) / BLOCK_DIM;
    tiling->tileNum = (tiling->blockLength + TILE_SIZE - 1) / TILE_SIZE;   // TILE_SIZE=256

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForModUnsqueezeCast([[maybe_unused]] gert::TilingParseContext* context)
{   
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(ModUnsqueezeCast).Tiling(ModUnsqueezeCastTilingFunc).TilingParse<ModUnsqueezeCastCompileInfo>(TilingParseForModUnsqueezeCast);
} // namespace optiling
