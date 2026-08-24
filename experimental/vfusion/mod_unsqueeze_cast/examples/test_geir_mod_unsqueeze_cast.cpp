/**
 * Graph 模式性能测试：ModUnsqueezeCast vs 原生 FmodScalar + Cast
 * 场景：x1 shape=[8,300] int64，x2 为 int64 标量
 * 输出：自定义算子 y shape=[8,300,1] float16（数据连续，可按元素对比）
 */
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <cmath>
#include <string>
#include <map>

#include "acl/acl.h"
#include "ge/ge_api.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "graph/operator_reg.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "ge/ge_api_types.h"
#include "ge/ge_error_codes.h"

// 若工程生成了 proto，可改为：
// #include "mod_unsqueeze_cast_proto.h"  // 然后用 op::ModUnsqueezeCast
// 原生算子原型（CANN 自带）
#include "all_ops.h"   // 含 FloorMod / Cast 等；若没有可改用 ge::Operator("FloorMod") 等

using namespace ge;
using namespace std;

#define CHECK_RET(cond, msg) \
  do { \
    if (!(cond)) { \
      printf("[ERROR] %s\n", msg); \
      return -1; \
    } \
  } while (0)

static int64_t GetShapeSize(const vector<int64_t>& shape) {
  int64_t s = 1;
  for (auto d : shape) s *= d;
  return s;
}

static void GenerateRandomData(vector<int64_t>& x1, int64_t& x2, int64_t total, int seed = 42) {
  mt19937 gen(seed);
  uniform_int_distribution<int64_t> dist(-10000, 10000);
  for (int64_t i = 0; i < total; ++i) x1[i] = dist(gen);
  do { x2 = dist(gen); } while (x2 == 0);
}

// 构造 Host Tensor（int64）
static Tensor MakeInt64Tensor(const vector<int64_t>& host, const vector<int64_t>& shape) {
  TensorDesc desc(Shape(shape), FORMAT_ND, DT_INT64);
  desc.SetPlacement(kPlacementHost);
  size_t bytes = host.size() * sizeof(int64_t);
  return Tensor(desc, reinterpret_cast<const uint8_t*>(host.data()), bytes);
}

// 构造 Host Tensor（float16 占位，仅用于描述；实际输出由 GE 分配）
static TensorDesc MakeFp16Desc(const vector<int64_t>& shape) {
  TensorDesc desc(Shape(shape), FORMAT_ND, DT_FLOAT16);
  desc.SetPlacement(kPlacementHost);
  return desc;
}

/**
 * 构图：自定义算子 ModUnsqueezeCast
 *   Data(x1) + Data(x2) -> ModUnsqueezeCast -> y
 */
static int BuildCustomGraph(Graph& graph,
                            vector<Operator>& inputs,
                            vector<Operator>& outputs,
                            const vector<int64_t>& shapeIn,
                            const vector<int64_t>& shapeOut) {
  // 输入 Data
  auto data_x1 = op::Data("data_x1").set_attr_index(0);
  TensorDesc desc_x1(Shape(shapeIn), FORMAT_ND, DT_INT64);
  data_x1.update_input_desc_x(desc_x1);
  data_x1.update_output_desc_y(desc_x1);

  auto data_x2 = op::Data("data_x2").set_attr_index(1);
  TensorDesc desc_x2(Shape({1}), FORMAT_ND, DT_INT64);
  data_x2.update_input_desc_x(desc_x2);
  data_x2.update_output_desc_y(desc_x2);

  // 自定义算子（OpDef 注册名必须与这里一致：ModUnsqueezeCast）
  // 若有生成的 op::ModUnsqueezeCast，优先用类接口
  ge::Operator mod("ModUnsqueezeCast");
  mod.SetInput("x1", data_x1);
  mod.SetInput("x2", data_x2);

  TensorDesc desc_y(Shape(shapeOut), FORMAT_ND, DT_FLOAT16);
  mod.UpdateOutputDesc("y", desc_y);

  graph.AddOp(data_x1);
  graph.AddOp(data_x2);
  graph.AddOp(mod);

  inputs = {data_x1, data_x2};
  outputs = {mod};
  graph.SetInputs(inputs).SetOutputs(outputs);
  return 0;
}

/**
 * 构图：原生路径 Fmod(FloorMod) + Cast
 * 注意：原生路径不做 Unsqueeze，输出 shape 仍为 [8,300]，但数据连续，可按元素对比/计时
 */
static int BuildBaselineGraph(Graph& graph,
                              vector<Operator>& inputs,
                              vector<Operator>& outputs,
                              const vector<int64_t>& shapeIn) {
  auto data_x1 = op::Data("data_x1").set_attr_index(0);
  TensorDesc desc_x1(Shape(shapeIn), FORMAT_ND, DT_INT64);
  data_x1.update_input_desc_x(desc_x1);
  data_x1.update_output_desc_y(desc_x1);

  auto data_x2 = op::Data("data_x2").set_attr_index(1);
  TensorDesc desc_x2(Shape({1}), FORMAT_ND, DT_INT64);
  data_x2.update_input_desc_x(desc_x2);
  data_x2.update_output_desc_y(desc_x2);

  // FloorMod（或 Fmod，按你环境支持的原生算子名调整）
  // 很多版本里标量取模可用 FloorMod / Mod
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

  graph.AddOp(data_x1);
  graph.AddOp(data_x2);
  graph.AddOp(floor_mod);
  graph.AddOp(cast_op);

  inputs = {data_x1, data_x2};
  outputs = {cast_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  return 0;
}

static double RunGraphPerf(Session* session, uint32_t graphId,
                           const vector<Tensor>& inputs,
                           vector<Tensor>& outputs,
                           int warmup, int iters) {
  // warmup
  for (int i = 0; i < warmup; ++i) {
    outputs.clear();
    auto ret = session->RunGraph(graphId, inputs, outputs);
    if (ret != SUCCESS) {
      printf("RunGraph warmup failed, ret=%d\n", ret);
      return -1.0;
    }
  }

  auto t0 = chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    outputs.clear();
    auto ret = session->RunGraph(graphId, inputs, outputs);
    if (ret != SUCCESS) {
      printf("RunGraph failed at iter %d, ret=%d\n", i, ret);
      return -1.0;
    }
  }
  auto t1 = chrono::steady_clock::now();
  double us = chrono::duration<double, micro>(t1 - t0).count();
  return us / iters;  // 平均微秒
}

int main() {
  const vector<int64_t> shapeIn  = {8, 300};
  const vector<int64_t> shapeOut = {8, 300, 1};
  const int64_t total = GetShapeSize(shapeIn);

  // 1. 准备数据（与 aclnn 测试一致）
  vector<int64_t> x1Host(total);
  int64_t x2Val = 0;
  GenerateRandomData(x1Host, x2Val, total, 0);
  printf("Test case: shape=[%ld,%ld], x2=%ld, total=%ld\n",
         shapeIn[0], shapeIn[1], x2Val, total);

  vector<int64_t> x2Host = {x2Val};
  Tensor tin_x1 = MakeInt64Tensor(x1Host, shapeIn);
  Tensor tin_x2 = MakeInt64Tensor(x2Host, {1});
  vector<Tensor> feedInputs = {tin_x1, tin_x2};

  // 2. GE 初始化
  map<AscendString, AscendString> globalOpts = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"}  // 0: 推理
  };
  Status ret = GEInitialize(globalOpts);
  CHECK_RET(ret == SUCCESS, "GEInitialize failed");

  map<AscendString, AscendString> sessOpts;
  Session* session = new Session(sessOpts);
  CHECK_RET(session != nullptr, "Create Session failed");

  const int WARMUP = 10;
  const int ITERS  = 100;

  // ========== 自定义算子图 ==========
  Graph customGraph("ModUnsqueezeCastGraph");
  vector<Operator> customInOps, customOutOps;
  CHECK_RET(BuildCustomGraph(customGraph, customInOps, customOutOps, shapeIn, shapeOut) == 0,
            "BuildCustomGraph failed");

  uint32_t customGraphId = 0;
  ret = session->AddGraph(customGraphId, customGraph);
  CHECK_RET(ret == SUCCESS, "AddGraph custom failed");

  vector<Tensor> customOutputs;
  double customUs = RunGraphPerf(session, customGraphId, feedInputs, customOutputs, WARMUP, ITERS);
  CHECK_RET(customUs > 0, "Custom graph perf failed");
  printf("[Custom ModUnsqueezeCast] avg latency = %.3f us  (warmup=%d, iters=%d)\n",
         customUs, WARMUP, ITERS);

  // ========== 原生基线：FloorMod + Cast ==========
  Graph baselineGraph("BaselineFmodCastGraph");
  vector<Operator> baseInOps, baseOutOps;
  CHECK_RET(BuildBaselineGraph(baselineGraph, baseInOps, baseOutOps, shapeIn) == 0,
            "BuildBaselineGraph failed");

  uint32_t baselineGraphId = 1;
  ret = session->AddGraph(baselineGraphId, baselineGraph);
  CHECK_RET(ret == SUCCESS, "AddGraph baseline failed");

  vector<Tensor> baselineOutputs;
  double baselineUs = RunGraphPerf(session, baselineGraphId, feedInputs, baselineOutputs, WARMUP, ITERS);
  CHECK_RET(baselineUs > 0, "Baseline graph perf failed");
  printf("[Baseline FloorMod+Cast] avg latency = %.3f us  (warmup=%d, iters=%d)\n",
         baselineUs, WARMUP, ITERS);

  if (baselineUs > 1e-6) {
    printf("Speedup (baseline / custom) = %.3fx\n", baselineUs / customUs);
  }

  // ========== 简单精度抽查（按元素，忽略 shape 多出来的 1） ==========
  if (!customOutputs.empty() && !baselineOutputs.empty()) {
    const uint8_t* cPtr = customOutputs[0].GetData();
    const uint8_t* bPtr = baselineOutputs[0].GetData();
    size_t cSize = customOutputs[0].GetSize();
    size_t bSize = baselineOutputs[0].GetSize();
    // float16 = 2 bytes
    size_t cNum = cSize / 2;
    size_t bNum = bSize / 2;
    size_t n = min(cNum, bNum);
    size_t fail = 0;
    for (size_t i = 0; i < n; ++i) {
      // 简单按 bit 比较 float16（或转 float 比较）
      uint16_t cv = reinterpret_cast<const uint16_t*>(cPtr)[i];
      uint16_t bv = reinterpret_cast<const uint16_t*>(bPtr)[i];
      if (cv != bv) {
        if (fail < 5) {
          printf("Mismatch at %zu: custom=0x%04x baseline=0x%04x\n", i, cv, bv);
        }
        fail++;
      }
    }
    printf("Precision check: compared %zu elements, fail=%zu\n", n, fail);
  }

  // 清理
  delete session;
  GEFinalize();
  printf("Done.\n");
  return 0;
}