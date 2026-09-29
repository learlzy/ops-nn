/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * ...
 *
 * 精度验证：自定义 aclnnConv2dSilu  vs  原生 aclnnConvolution + aclnnSilu
 * 公式：y = SiLU(Conv2d(x, filter) + bias)
 * 支持 FP16 / FP32，NCHW
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <acl/acl.h>
#include "aclnnop/aclnn_convolution.h"
#include "aclnnop/aclnn_silu.h"
// 自定义融合算子头文件（由 ops-nn 编译生成，路径按实际工程调整）
#include "aclnn_conv2d_silu.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(fmt, ...)          \
    do {                             \
        printf(fmt, ##__VA_ARGS__);  \
    } while (0)

// ============================================================
// 工具函数
// ============================================================
static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

static std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template <typename T>
static int CreateAclTensor(const std::vector<T>& hostData,
                           const std::vector<int64_t>& shape,
                           void** deviceAddr,
                           aclDataType dataType,
                           aclTensor** tensor,
                           aclFormat format = ACL_FORMAT_NCHW)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);

    auto strides = MakeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, format, shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);
    return 0;
}

// 输出尺寸公式
static void CalcOutputShape(int64_t n, int64_t cin, int64_t hi, int64_t wi,
                            int64_t cout, int64_t kh, int64_t kw,
                            int64_t padT, int64_t padB, int64_t padL, int64_t padR,
                            int64_t strideH, int64_t strideW,
                            int64_t dilH, int64_t dilW,
                            std::vector<int64_t>& outShape)
{
    int64_t ho = (hi + padT + padB - dilH * (kh - 1) - 1) / strideH + 1;
    int64_t wo = (wi + padL + padR - dilW * (kw - 1) - 1) / strideW + 1;
    outShape = {n, cout, ho, wo};
}

// FP16 <-> float 辅助
static inline float Fp16ToFloat(aclFloat16 v) { return aclFloat16ToFloat(v); }
static inline aclFloat16 FloatToFp16(float v) { return aclFloatToFloat16(v); }

// ============================================================
// 数据生成
// ============================================================
template <typename T>
static void FillRandom(std::vector<T>& data, float lo, float hi, int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < data.size(); ++i) {
        float v = dist(gen);
        if constexpr (std::is_same_v<T, aclFloat16>) {
            data[i] = FloatToFp16(v);
        } else {
            data[i] = static_cast<T>(v);
        }
    }
}

// ============================================================
// Golden：aclnnConvolution + aclnnSilu
// ============================================================
template <typename T>
static int ComputeGolden(const std::vector<T>& xHost,
                         const std::vector<T>& filterHost,
                         const std::vector<T>& biasHost,
                         const std::vector<int64_t>& xShape,
                         const std::vector<int64_t>& filterShape,
                         const std::vector<int64_t>& biasShape,
                         const std::vector<int64_t>& outShape,
                         const std::vector<int64_t>& strides,
                         const std::vector<int64_t>& pads,
                         const std::vector<int64_t>& dilations,
                         int64_t groups,
                         std::vector<T>& goldenHost,
                         aclDataType dataType,
                         aclrtStream stream)
{
    void *xDev = nullptr, *wDev = nullptr, *bDev = nullptr;
    void *convOutDev = nullptr, *siluOutDev = nullptr;
    aclTensor *xTensor = nullptr, *wTensor = nullptr, *bTensor = nullptr;
    aclTensor *convOutTensor = nullptr, *siluOutTensor = nullptr;

    int64_t outNum = GetShapeSize(outShape);
    std::vector<T> convOutHost(outNum, T{});

    auto ret = CreateAclTensor(xHost, xShape, &xDev, dataType, &xTensor);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(filterHost, filterShape, &wDev, dataType, &wTensor);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(biasHost, biasShape, &bDev, dataType, &bTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(convOutHost, outShape, &convOutDev, dataType, &convOutTensor);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(goldenHost, outShape, &siluOutDev, dataType, &siluOutTensor);
    CHECK_RET(ret == 0, return ret);

    // ---- aclnnConvolution ----
    aclIntArray* strideArr   = aclCreateIntArray(strides.data(), strides.size());
    aclIntArray* padArr      = aclCreateIntArray(pads.data(), pads.size());
    aclIntArray* dilationArr = aclCreateIntArray(dilations.data(), dilations.size());
    // outputPadding 对非转置卷积传空
    std::vector<int64_t> outPadZero = {0, 0};
    aclIntArray* outPadArr = aclCreateIntArray(outPadZero.data(), outPadZero.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // cubeMathType: 0 = KEEP_DTYPE
    ret = aclnnConvolutionGetWorkspaceSize(
        xTensor, wTensor, bTensor,
        strideArr, padArr, dilationArr,
        false /*transposed*/, outPadArr, groups,
        convOutTensor, 0 /*cubeMathType*/,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ws malloc failed\n"); return ret);
    }
    ret = aclnnConvolution(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnConvolution failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // ---- aclnnSilu ----
    workspaceSize = 0;
    executor = nullptr;
    ret = aclnnSiluGetWorkspaceSize(convOutTensor, siluOutTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSiluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        if (workspaceAddr) { aclrtFree(workspaceAddr); workspaceAddr = nullptr; }
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnSilu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSilu failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 拷回
    ret = aclrtMemcpy(goldenHost.data(), outNum * sizeof(T),
                      siluOutDev, outNum * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 释放
    aclDestroyTensor(xTensor); aclDestroyTensor(wTensor);
    aclDestroyTensor(bTensor); aclDestroyTensor(convOutTensor);
    aclDestroyTensor(siluOutTensor);
    aclDestroyIntArray(strideArr); aclDestroyIntArray(padArr);
    aclDestroyIntArray(dilationArr); aclDestroyIntArray(outPadArr);
    aclrtFree(xDev); aclrtFree(wDev); aclrtFree(bDev);
    aclrtFree(convOutDev); aclrtFree(siluOutDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

// ============================================================
// 自定义算子：aclnnConv2dSilu
// ============================================================
template <typename T>
static int ComputeCustom(const std::vector<T>& xHost,
                         const std::vector<T>& filterHost,
                         const std::vector<T>& biasHost,
                         const std::vector<int64_t>& xShape,
                         const std::vector<int64_t>& filterShape,
                         const std::vector<int64_t>& biasShape,
                         const std::vector<int64_t>& outShape,
                         const std::vector<int64_t>& strides,
                         const std::vector<int64_t>& pads,
                         const std::vector<int64_t>& dilations,
                         int64_t groups,
                         std::vector<T>& outHost,
                         aclDataType dataType,
                         aclrtStream stream)
{
    void *xDev = nullptr, *wDev = nullptr, *bDev = nullptr, *yDev = nullptr;
    aclTensor *xTensor = nullptr, *wTensor = nullptr, *bTensor = nullptr, *yTensor = nullptr;

    auto ret = CreateAclTensor(xHost, xShape, &xDev, dataType, &xTensor);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(filterHost, filterShape, &wDev, dataType, &wTensor);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(biasHost, biasShape, &bDev, dataType, &bTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(outHost, outShape, &yDev, dataType, &yTensor);
    CHECK_RET(ret == 0, return ret);

    aclIntArray* strideArr   = aclCreateIntArray(strides.data(), strides.size());
    aclIntArray* padArr      = aclCreateIntArray(pads.data(), pads.size());
    aclIntArray* dilationArr = aclCreateIntArray(dilations.data(), dilations.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 接口名以实际生成的 aclnn 头文件为准（opInterface.value = "conv2dsilu"）
    // 常见签名：
    // aclnnConv2dSiluGetWorkspaceSize(x, filter, bias, strides, pads, dilations, groups, y, &ws, &exe)
    char format[] = "NCHW";
    ret = aclnnConv2dSiluGetWorkspaceSize(
        xTensor, wTensor, bTensor,
        strideArr, padArr, dilationArr, groups, format,
        yTensor, &workspaceSize, &executor);
    // CHECK_RET(ret == ACL_SUCCESS,
    //           LOG_PRINT("aclnnConv2dSiluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if(ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        printf("aclnn error msg: %s\n", errMsg);
        printf("ret = %d\n", ret);
        return ret;
    }


    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnConv2dSilu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnConv2dSilu failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t outNum = GetShapeSize(outShape);
    ret = aclrtMemcpy(outHost.data(), outNum * sizeof(T),
                      yDev, outNum * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclDestroyTensor(xTensor); aclDestroyTensor(wTensor);
    aclDestroyTensor(bTensor); aclDestroyTensor(yTensor);
    aclDestroyIntArray(strideArr); aclDestroyIntArray(padArr);
    aclDestroyIntArray(dilationArr);
    aclrtFree(xDev); aclrtFree(wDev); aclrtFree(bDev); aclrtFree(yDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

// ============================================================
// 精度比对
// ============================================================
template <typename T>
static bool CompareResult(const std::vector<T>& custom,
                          const std::vector<T>& golden,
                          float atol, float rtol)
{
    size_t total = custom.size();
    size_t failCnt = 0;
    float maxAbs = 0.f, maxRel = 0.f;

    for (size_t i = 0; i < total; ++i) {
        float c, g;
        if constexpr (std::is_same_v<T, aclFloat16>) {
            c = Fp16ToFloat(custom[i]);
            g = Fp16ToFloat(golden[i]);
        } else {
            c = static_cast<float>(custom[i]);
            g = static_cast<float>(golden[i]);
        }
        float absErr = std::fabs(c - g);
        float relErr = absErr / (std::fabs(g) + 1e-8f);
        maxAbs = std::max(maxAbs, absErr);
        maxRel = std::max(maxRel, relErr);
        if (absErr > atol && relErr > rtol) {
            if (failCnt < 16) {
                LOG_PRINT("  mismatch[%zu]: custom=%.6f golden=%.6f abs=%.6e rel=%.6e\n",
                          i, c, g, absErr, relErr);
            }
            ++failCnt;
        }
    }
    LOG_PRINT("Total=%zu  Fail=%zu  MaxAbs=%.6e  MaxRel=%.6e\n",
              total, failCnt, maxAbs, maxRel);
    return failCnt == 0;
}

// ============================================================
// 单 case 入口
// ============================================================
template <typename T>
static int RunOneCase(aclDataType dataType, const char* dtypeName,
                      int64_t n, int64_t cin, int64_t hi, int64_t wi,
                      int64_t cout, int64_t kh, int64_t kw,
                      int64_t padT, int64_t padB, int64_t padL, int64_t padR,
                      int64_t strideH, int64_t strideW,
                      int64_t dilH, int64_t dilW,
                      int64_t groups,
                      float atol, float rtol,
                      aclrtStream stream)
{
    LOG_PRINT("\n========== Case %s  N=%ld C=%ld HxW=%ldx%ld -> Cout=%ld KhxKw=%ldx%ld "
              "pad=[%ld,%ld,%ld,%ld] stride=[%ld,%ld] dil=[%ld,%ld] ==========\n",
              dtypeName, n, cin, hi, wi, cout, kh, kw,
              padT, padB, padL, padR, strideH, strideW, dilH, dilW);

    std::vector<int64_t> xShape      = {n, cin, hi, wi};
    std::vector<int64_t> filterShape = {cout, cin / groups, kh, kw};
    std::vector<int64_t> biasShape   = {cout};
    std::vector<int64_t> outShape;
    CalcOutputShape(n, cin, hi, wi, cout, kh, kw,
                    padT, padB, padL, padR, strideH, strideW, dilH, dilW, outShape);

    // strides/pads/dilations 按 NCHW 约定：长度 2（H,W）或 4（N,C,H,W）均可
    // 这里用长度 2，与多数 aclnnConvolution 示例一致
    std::vector<int64_t> strides   = {strideH, strideW};
    std::vector<int64_t> pads      = {padT, padB, padL, padR};  // 4 值
    std::vector<int64_t> dilations = {dilH, dilW};

    int64_t xNum     = GetShapeSize(xShape);
    int64_t filterNum= GetShapeSize(filterShape);
    int64_t biasNum  = GetShapeSize(biasShape);
    int64_t outNum   = GetShapeSize(outShape);

    std::vector<T> xHost(xNum), filterHost(filterNum), biasHost(biasNum);
    std::vector<T> customOut(outNum), goldenOut(outNum);

    // 小范围随机，避免 FP16 溢出
    FillRandom(xHost,      -1.0f, 1.0f, 11);
    FillRandom(filterHost, -0.5f, 0.5f, 22);
    FillRandom(biasHost,   -0.2f, 0.2f, 33);

    int ret;

    ret = ComputeCustom(xHost, filterHost, biasHost,
                            xShape, filterShape, biasShape, outShape,
                            strides, pads, dilations, groups,
                            customOut, dataType, stream);
    CHECK_RET(ret == 0, LOG_PRINT("ComputeCustom failed\n"); return ret);

    // ret = ComputeGolden(xHost, filterHost, biasHost,
    //                     xShape, filterShape, biasShape, outShape,
    //                     strides, pads, dilations, groups,
    //                     goldenOut, dataType, stream);
    // CHECK_RET(ret == 0, LOG_PRINT("ComputeGolden failed\n"); return ret);

    // bool pass = CompareResult(customOut, goldenOut, atol, rtol);
    // if (pass) {
    //     LOG_PRINT("[PASS] %s\n", dtypeName);
    // } else {
    //     LOG_PRINT("[FAIL] %s\n", dtypeName);
    // }
    // return pass ? 0 : 1;

    return 0;
}

// ============================================================
// main
// ============================================================
int main()
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed %d\n", ret); return ret);
    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int fail = 0;

    // ---------- FP16 典型 case（与 catlass example 接近） ----------
    fail += RunOneCase<aclFloat16>(
        ACL_FLOAT16, "FP16",
        /*N*/2, /*Cin*/112, /*H*/33, /*W*/43,
        /*Cout*/80, /*Kh*/3, /*Kw*/3,
        /*pad*/1,1,1,1, /*stride*/1,1, /*dil*/1,1, /*groups*/1,
        /*atol*/1e-2f, /*rtol*/1e-2f, stream);

    // // ---------- FP16 小 shape ----------
    // fail += RunOneCase<aclFloat16>(
    //     ACL_FLOAT16, "FP16-small",
    //     1, 16, 16, 16, 32, 3, 3,
    //     1,1,1,1, 1,1, 1,1, 1,
    //     1e-2f, 1e-2f, stream);

    // // ---------- FP32 ----------
    // fail += RunOneCase<float>(
    //     ACL_FLOAT, "FP32",
    //     1, 32, 28, 28, 64, 3, 3,
    //     1,1,1,1, 1,1, 1,1, 1,
    //     1e-4f, 1e-4f, stream);

    // // ---------- stride=2（注意 L1 约束，核尽量小） ----------
    // fail += RunOneCase<aclFloat16>(
    //     ACL_FLOAT16, "FP16-s2",
    //     1, 32, 32, 32, 64, 2, 2,
    //     0,0,0,0, 2,2, 1,1, 1,
    //     1e-2f, 1e-2f, stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (fail == 0) {
        LOG_PRINT("\n========== ALL PASSED ==========\n");
        return 0;
    }
    LOG_PRINT("\n========== %d CASE(S) FAILED ==========\n", fail);
    return 1;
}