/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * ...
 */

/*!
 * \file conv2d_silu_tiling.cpp
 * \brief Tiling for Conv2dSilu, support FP16 / FP32
 *        Host 侧不能编译 Catlass，因此手工复现 Conv2dEpilogue::GetWorkspaceSize 公式
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_ascendc.h"
#include "conv2d_silu/op_kernel/conv2d_silu_tiling_data.h"
#include "conv2d_silu/op_kernel/conv2d_silu_tiling_key.h"
// 注意：不要 include conv2d_silu_kernel_template.h（里面拉了整套 Catlass）

namespace optiling {

struct Conv2dSiluCompileInfo {};

// Ascend C0 字节数（与 Catlass BYTE_PER_C0 一致）
static constexpr uint32_t BYTE_PER_C0 = 32;

static uint64_t GetTilingKeyByDtype(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT16) {
        return static_cast<uint64_t>(CONV2D_SILU_SCH_MODE_FP16);
    }
    return static_cast<uint64_t>(CONV2D_SILU_SCH_MODE_FP32);
}

/**
 * @brief 计算用户 workspace 大小（字节）
 *        复现 Catlass::Conv::Kernel::Conv2dEpilogue::GetWorkspaceSize
 *        workspace 存 AIC 写出的中间结果，布局为 NC1HWC0，元素类型 = 输出类型
 *
 * @param batch, ho, wo, cout1  已由 problem shape 推导
 * @param elementBytes          sizeof(ElementOutput)：FP16=2, FP32=4
 */
static size_t CalcUsrWorkspaceSize(uint32_t batch, uint32_t ho, uint32_t wo,
                                   uint32_t cout1, size_t elementBytes)
{
    // C0_WORKSPACE = BYTE_PER_C0 / sizeof(ElementWorkspace)
    const uint32_t c0Workspace = static_cast<uint32_t>(BYTE_PER_C0 / elementBytes);
    const size_t lenWorkspace =
        static_cast<size_t>(batch) * cout1 * ho * wo * c0Workspace;
    return lenWorkspace * elementBytes;
}

/**
 * @brief 根据 NCHW 输入与卷积参数推导 ho/wo/cin1/cout1/c0 等
 *        逻辑对齐 Catlass::Conv2dParams::MakeConv2dParams
 */
static void DeriveConvShape(uint32_t hi, uint32_t wi, uint32_t cin, uint32_t cout,
                            uint32_t kh, uint32_t kw,
                            uint32_t padTop, uint32_t padBottom,
                            uint32_t padLeft, uint32_t padRight,
                            uint32_t strideH, uint32_t strideW,
                            uint32_t dilH, uint32_t dilW,
                            size_t elementBytes,
                            uint32_t& ho, uint32_t& wo,
                            uint32_t& cin1, uint32_t& cout1,
                            uint32_t& coutRound, uint32_t& c0)
{
    c0 = static_cast<uint32_t>(BYTE_PER_C0 / elementBytes);
    cin1 = (cin + c0 - 1) / c0;
    cout1 = (cout + c0 - 1) / c0;
    coutRound = cout1 * c0;

    // 标准卷积输出尺寸公式
    const int64_t effectiveKh = static_cast<int64_t>(dilH) * (kh - 1) + 1;
    const int64_t effectiveKw = static_cast<int64_t>(dilW) * (kw - 1) + 1;
    ho = static_cast<uint32_t>(
        (static_cast<int64_t>(hi) + padTop + padBottom - effectiveKh) / strideH + 1);
    wo = static_cast<uint32_t>(
        (static_cast<int64_t>(wi) + padLeft + padRight - effectiveKw) / strideW + 1);
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

    // 兼容 [N,C,H,W] 与 [H,W] 两种 attr 长度
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

    // 输入约定 NCHW，filter 约定 NCHW (cout, cin, kh, kw)
    uint32_t batch = static_cast<uint32_t>(xDims.GetDim(0));
    uint32_t cin  = static_cast<uint32_t>(xDims.GetDim(1));
    uint32_t hi   = static_cast<uint32_t>(xDims.GetDim(2));
    uint32_t wi   = static_cast<uint32_t>(xDims.GetDim(3));
    uint32_t cout = static_cast<uint32_t>(fDims.GetDim(0));
    uint32_t kh   = static_cast<uint32_t>(fDims.GetDim(2));
    uint32_t kw   = static_cast<uint32_t>(fDims.GetDim(3));

    // 元素字节数
    size_t elementBytes = (dtype == ge::DT_FLOAT16) ? sizeof(uint16_t) : sizeof(float);

    // 推导输出与 C0 相关维度（不依赖 Catlass）
    uint32_t ho = 0, wo = 0, cin1 = 0, cout1 = 0, coutRound = 0, c0 = 0;
    DeriveConvShape(hi, wi, cin, cout, kh, kw,
                    static_cast<uint32_t>(padTop), static_cast<uint32_t>(padBottom),
                    static_cast<uint32_t>(padLeft), static_cast<uint32_t>(padRight),
                    static_cast<uint32_t>(strideH), static_cast<uint32_t>(strideW),
                    static_cast<uint32_t>(dilH), static_cast<uint32_t>(dilW),
                    elementBytes,
                    ho, wo, cin1, cout1, coutRound, c0);

    // 简单合法性检查（对齐 Catlass CanImplement 的一部分）
    if (strideH == 0 || strideW == 0 || dilH == 0 || dilW == 0 ||
        ho == 0 || wo == 0) {
        OP_LOGE(context->GetNodeName(),
                "Invalid conv params: stride/dilation/output shape zero");
        return ge::GRAPH_FAILED;
    }

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

    tilingData->ho = ho;
    tilingData->wo = wo;
    tilingData->cin1 = cin1;
    tilingData->cout1 = cout1;
    tilingData->coutRound = coutRound;
    tilingData->c0 = c0;

    // 计算用户 workspace（手工复现 GetWorkspaceSize）
    size_t usrWorkspaceSize = CalcUsrWorkspaceSize(batch, ho, wo, cout1, elementBytes);
    tilingData->workspaceSize = static_cast<uint64_t>(usrWorkspaceSize);

    // tiling key
    context->SetTilingKey(GetTilingKeyByDtype(dtype));

    // block dim：按 AIC 核数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);

    // workspace = 系统 workspace + 用户 workspace
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = sysWorkspaceSize + usrWorkspaceSize;

    OP_LOGD(context->GetNodeName(),
            "Tiling done: batch=%u hi=%u wi=%u cin=%u cout=%u ho=%u wo=%u "
            "cin1=%u cout1=%u c0=%u workspace=%lu",
            tilingData->batch, tilingData->hi, tilingData->wi,
            tilingData->cin, tilingData->cout,
            tilingData->ho, tilingData->wo,
            tilingData->cin1, tilingData->cout1, tilingData->c0,
            tilingData->workspaceSize);

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