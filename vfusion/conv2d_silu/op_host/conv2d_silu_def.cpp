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
 * \file conv2d_silu_def.cpp
 * \brief Conv2d + Bias + SiLU fused op definition (Ascend910B)
 *        Support dtype: FP16 / BF16 / FP32
 */

#include "register/op_def_registry.h"

namespace ops {

class Conv2dSilu : public OpDef {
public:
    explicit Conv2dSilu(const char* name) : OpDef(name)
    {
        // x: NCHW, FP16 / BF16 / FP32
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW, ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW, ge::FORMAT_NCHW});

        // filter: NCHW (Cout, Cin/groups, Kh, Kw)
        this->Input("filter")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW, ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW, ge::FORMAT_NCHW});

        // bias: optional 1-D
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // y: NCHW
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW, ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW, ge::FORMAT_NCHW});

        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0, 0, 0, 0});
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NCHW");

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "conv2d_silu")
            .ExtendCfgInfo("opInterface.value", "conv2dsilu")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

OP_ADD(Conv2dSilu);

} // namespace ops