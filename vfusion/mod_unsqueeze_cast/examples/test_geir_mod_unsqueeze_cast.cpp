

/*!
 * \file test_geir_mod_unsqueeze_cast.cpp
 * \brief Graph 模式性能测试：ModUnsqueezeCast vs 原生 FloorMod + Cast
 *        场景：x1 shape=[8,300] int64，x2 为 int64 标量
 *        输出：自定义算子 y shape=[8,300,1] float16
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <chrono>
#include <cmath>
#include <acl/acl.h>
#include "assert.h"
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"
#include "elewise_calculation_ops.h"


#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT)   return 4;
    if (dt == ge::DT_FLOAT16) return 2;
    if (dt == ge::DT_BF16)    return 2;
    if (dt == ge::DT_INT16)   return 2;
    if (dt == ge::DT_UINT16)  return 2;
    if (dt == ge::DT_INT32)   return 4;
    if (dt == ge::DT_UINT32)  return 4;
    if (dt == ge::DT_INT64)   return 8;
    if (dt == ge::DT_UINT64)  return 8;
    if (dt == ge::DT_INT8)    return 1;
    return 1;
}

static int64_t GetShapeSize(const vector<int64_t>& shape)
{
    int64_t s = 1;
    for (auto d : shape) s *= d;
    return s;
}

static void GenerateRandomData(vector<int64_t>& x1, int64_t& x2, int64_t total, int seed = 0)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int64_t> dist(-10000, 10000);
    for (int64_t i = 0; i < total; ++i) {
        x1[i] = dist(gen);
    }
    do {
        x2 = dist(gen);
    } while (x2 == 0);
}

static Tensor MakeInt64Tensor(const vector<int64_t>& host, const vector<int64_t>& shape)
{
    TensorDesc desc(Shape(shape), FORMAT_ND, DT_INT64);
    desc.SetPlacement(kPlacementHost);
    desc.SetFormat(FORMAT_ND);
    desc.SetRealDimCnt(shape.size());
    size_t bytes = host.size() * sizeof(int64_t);
    return Tensor(desc, reinterpret_cast<const uint8_t*>(host.data()), bytes);
}

/**
 * 构图：自定义算子 ModUnsqueezeCast
 * Data(x1) + Data(x2) -> ModUnsqueezeCast -> y
 */
int CreateCustomGraph(std::vector<ge::Tensor>& input,
                      std::vector<Operator>& inputs,
                      std::vector<Operator>& outputs,
                      Graph& graph,
                      const vector<int64_t>& shapeIn,
                      const vector<int64_t>& shapeOut,
                      const vector<int64_t>& x1Host,
                      int64_t x2Val)
{
    // ---------- Data x1 ----------
    auto data_x1 = op::Data("data_x1").set_attr_index(0);
    TensorDesc desc_x1(Shape(shapeIn), FORMAT_ND, DT_INT64);
    desc_x1.SetPlacement(kPlacementHost);
    desc_x1.SetFormat(FORMAT_ND);
    data_x1.update_input_desc_x(desc_x1);
    data_x1.update_output_desc_y(desc_x1);

    Tensor tin_x1 = MakeInt64Tensor(x1Host, shapeIn);
    input.push_back(tin_x1);
    graph.AddOp(data_x1);
    inputs.push_back(data_x1);

    // ---------- Data x2 (scalar) ----------
    auto data_x2 = op::Data("data_x2").set_attr_index(1);
    TensorDesc desc_x2(Shape({1}), FORMAT_ND, DT_INT64);
    desc_x2.SetPlacement(kPlacementHost);
    desc_x2.SetFormat(FORMAT_ND);
    data_x2.update_input_desc_x(desc_x2);
    data_x2.update_output_desc_y(desc_x2);

    vector<int64_t> x2Host = {x2Val};
    Tensor tin_x2 = MakeInt64Tensor(x2Host, {1});
    input.push_back(tin_x2);
    graph.AddOp(data_x2);
    inputs.push_back(data_x2);

    // ---------- 自定义算子 ModUnsqueezeCast ----------
    // 若已生成 proto，优先使用：
    // auto mod = op::ModUnsqueezeCast("mod_unsqueeze_cast");
    // mod.set_input_x1(data_x1);
    // mod.set_input_x2(data_x2);
    ge::Operator mod("ModUnsqueezeCast");
    mod.SetInput("x1", data_x1);
    mod.SetInput("x2", data_x2);

    TensorDesc desc_y(Shape(shapeOut), FORMAT_ND, DT_FLOAT16);
    mod.UpdateOutputDesc("y", desc_y);

    graph.AddOp(mod);
    outputs.push_back(mod);

    return SUCCESS;
}

/**
 * 构图：原生路径 FloorMod + Cast（对比基线）
 */
int CreateBaselineGraph(std::vector<ge::Tensor>& input,
                        std::vector<Operator>& inputs,
                        std::vector<Operator>& outputs,
                        Graph& graph,
                        const vector<int64_t>& shapeIn,
                        const vector<int64_t>& x1Host,
                        int64_t x2Val)
{
    auto data_x1 = op::Data("data_x1").set_attr_index(0);
    TensorDesc desc_x1(Shape(shapeIn), FORMAT_ND, DT_INT64);
    desc_x1.SetPlacement(kPlacementHost);
    desc_x1.SetFormat(FORMAT_ND);
    data_x1.update_input_desc_x(desc_x1);
    data_x1.update_output_desc_y(desc_x1);

    Tensor tin_x1 = MakeInt64Tensor(x1Host, shapeIn);
    input.push_back(tin_x1);
    graph.AddOp(data_x1);
    inputs.push_back(data_x1);

    auto data_x2 = op::Data("data_x2").set_attr_index(1);
    TensorDesc desc_x2(Shape({1}), FORMAT_ND, DT_INT64);
    desc_x2.SetPlacement(kPlacementHost);
    desc_x2.SetFormat(FORMAT_ND);
    data_x2.update_input_desc_x(desc_x2);
    data_x2.update_output_desc_y(desc_x2);

    vector<int64_t> x2Host = {x2Val};
    Tensor tin_x2 = MakeInt64Tensor(x2Host, {1});
    input.push_back(tin_x2);
    graph.AddOp(data_x2);
    inputs.push_back(data_x2);

    // FloorMod
    auto floor_mod = op::FloorMod("floor_mod");
    floor_mod.set_input_x1(data_x1);
    floor_mod.set_input_x2(data_x2);
    TensorDesc desc_mod(Shape(shapeIn), FORMAT_ND, DT_INT64);
    floor_mod.update_output_desc_y(desc_mod);

    // Cast int64 -> float16
    auto cast_op = op::Cast("cast_fp16");
    cast_op.set_input_x(floor_mod);
    cast_op.set_attr_dst_type(static_cast<int64_t>(DT_FLOAT16));
    TensorDesc desc_cast(Shape(shapeIn), FORMAT_ND, DT_FLOAT16);
    cast_op.update_output_desc_y(desc_cast);

    graph.AddOp(floor_mod);
    graph.AddOp(cast_op);

    outputs.push_back(cast_op);
    return SUCCESS;
}

static double RunGraphPerf(Session* session, uint32_t graphId,
                           const vector<Tensor>& inputs,
                           vector<Tensor>& outputs,
                           int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i) {
        outputs.clear();
        auto ret = session->RunGraph(graphId, inputs, outputs);
        if (ret != SUCCESS) {
            printf("RunGraph warmup failed, ret=%d\n", ret);
            return -1.0;
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        outputs.clear();
        auto ret = session->RunGraph(graphId, inputs, outputs);
        if (ret != SUCCESS) {
            printf("RunGraph failed at iter %d, ret=%d\n", i, ret);
            return -1.0;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return us / iters;
}

int main(int argc, char* argv[])
{
    const int WARMUP = 100;
    const int ITERS  = 10000;
    const vector<int64_t> shapeIn  = {8, 300};

    const int64_t total = GetShapeSize(shapeIn);
    vector<int64_t> x1Host(total);
    int64_t x2Val = 0;
    GenerateRandomData(x1Host, x2Val, total, 0);
    printf("%s - INFO - Test case: shape=[%ld,%ld], x2=%ld, total=%ld\n",
           GetTime().c_str(), shapeIn[0], shapeIn[1], x2Val, total);

    // ========== GE 初始化 ==========
    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {
        {"ge.exec.deviceId", "0"},
        {"ge.graphRunMode", "1"}
    };
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    std::map<AscendString, AscendString> build_options;
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create Session failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    
    std::map<AscendString, AscendString> graph_options;

    // ========== 1. 自定义算子图 ==========
    Graph customGraph("ModUnsqueezeCastGraph");
    std::vector<ge::Tensor> customInput;
    std::vector<Operator> customInOps, customOutOps;

    const std::vector<int64_t> shapeOut = [&shapeIn]() {
        auto tmp = shapeIn;
        tmp.push_back(int64_t{1});
        return tmp;
    }();
    ret = CreateCustomGraph(customInput, customInOps, customOutOps,
                            customGraph, shapeIn, shapeOut, x1Host, x2Val);
    if (ret != SUCCESS) {
        printf("%s - ERROR - CreateCustomGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    customGraph.SetInputs(customInOps).SetOutputs(customOutOps);

    uint32_t customGraphId = 0;
    ret = session->AddGraph(customGraphId, customGraph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - AddGraph custom failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    vector<Tensor> customOutputs;
    double customUs = RunGraphPerf(session, customGraphId, customInput,
                                   customOutputs, WARMUP, ITERS);
    if (customUs < 0) {
        printf("%s - ERROR - Custom graph perf failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("[Custom ModUnsqueezeCast] avg latency = %.3f us  (warmup=%d, iters=%d)\n",
           customUs, WARMUP, ITERS);

    // ========== 2. 原生基线图 FloorMod + Cast ==========
    Graph baselineGraph("BaselineFloorModCastGraph");
    std::vector<ge::Tensor> baselineInput;
    std::vector<Operator> baseInOps, baseOutOps;

    ret = CreateBaselineGraph(baselineInput, baseInOps, baseOutOps,
                              baselineGraph, shapeIn, x1Host, x2Val);
    if (ret != SUCCESS) {
        printf("%s - ERROR - CreateBaselineGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    baselineGraph.SetInputs(baseInOps).SetOutputs(baseOutOps);

    uint32_t baselineGraphId = 1;
    ret = session->AddGraph(baselineGraphId, baselineGraph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - AddGraph baseline failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    vector<Tensor> baselineOutputs;
    double baselineUs = RunGraphPerf(session, baselineGraphId, baselineInput,
                                     baselineOutputs, WARMUP, ITERS);
    if (baselineUs < 0) {
        printf("%s - ERROR - Baseline graph perf failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("[Baseline FloorMod+Cast] avg latency = %.3f us  (warmup=%d, iters=%d)\n",
           baselineUs, WARMUP, ITERS);

    if (baselineUs > 1e-6) {
        printf("Speedup (baseline / custom) = %.3fx\n", baselineUs / customUs);
    }

    // // ========== 精度抽查（直接用上面两次跑的输出） ==========
    // if (!customOutputs.empty() && !baselineOutputs.empty()) {
    //     float atol = 1e-3f;
    //     float rtol = 1e-3f;
    //     const aclFloat16* cPtr = reinterpret_cast<const aclFloat16*>(customOutputs[0].GetData());
    //     const aclFloat16* bPtr = reinterpret_cast<const aclFloat16*>(baselineOutputs[0].GetData());
    //     size_t cSize = customOutputs[0].GetSize();
    //     size_t bSize = baselineOutputs[0].GetSize();
    //     size_t cNum = cSize / 2;   // float16
    //     size_t bNum = bSize / 2;
    //     if (cNum != bNum) {
    //         printf("Output size mismatch: custom=%zu, baseline=%zu\n", cNum, bNum);
    //         delete session;
    //         GEFinalize();
    //         return FAILED;
    //     }
    //     size_t n = cNum;
    //     size_t failCnt = 0;
    //     float maxAbsErr = 0.0f;
    //     for (size_t i = 0; i < n; ++i) {
    //         float cv = aclFloat16ToFloat((cPtr)[i]);
    //         float bv = aclFloat16ToFloat((bPtr)[i]);
    //         float absErr = std::fabs(cv - bv);
    //         float relErr = absErr / (std::fabs(bv) + 1e-8f);
    //         if (absErr > maxAbsErr) maxAbsErr = absErr;
    //         if (absErr > atol && relErr > rtol) {
    //         if (failCnt < 10) {
    //             LOG_PRINT("Mismatch at %zu: host=%ld, divisor=%ld, custom=%.6f, golden=%.6f, abs=%.6f, rel=%.6f\n",
    //                     i, x1Host[i], x2Val, cv, bv, absErr, relErr);
    //         }
    //             failCnt++;
    //         }
    //     }
    //     printf("Precision check: compared %zu elements, fail=%zu\n", n, failCnt);
    //     if (failCnt == 0) {
    //         printf("[SUCCESS] Precision verification passed!\n");
    //     } else {
    //         printf("[FAILED] Precision verification failed!\n");
    //     }
    // }

    // ========== 清理 ==========
    delete session;
    GEFinalize();
    printf("%s - INFO - Done.\n", GetTime().c_str());
    return SUCCESS;
}