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
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "conv2d_silu/op_kernel/conv2d_silu_tiling_data.h"
#include "conv2d_silu/op_kernel/conv2d_silu_tiling_key.h"

namespace optiling {

struct Conv2dSiluCompileInfo {};

// tiling 分发入口
static ge::graphStatus Conv2dSiluTilingFunc(gert::TilingContext* context)
{
}

static ge::graphStatus TilingParseForConv2dSilu([[maybe_unused]] gert::TilingParseContext* context)
{   
}

// tiling注册入口.
IMPL_OP_OPTILING(Conv2dSilu).Tiling(Conv2dSiluTilingFunc).TilingParse<Conv2dSiluCompileInfo>(TilingParseForConv2dSilu);
} // namespace optiling
