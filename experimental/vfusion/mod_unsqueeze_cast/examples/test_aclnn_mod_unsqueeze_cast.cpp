/**
 * 精度测试：ModUnsqueezeCast vs CANN 原生 FmodScalar + Cast
 * 测试场景：x1 shape=[8,300] int64，x2 为 int64 标量
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <acl/acl.h>
#include "aclnnop/aclnn_fmod_scalar.h"   // 标量取模
#include "aclnnop/aclnn_cast.h"           // 类型转换
// 自定义算子头文件（根据实际生成路径调整）
#include "aclnn_mod_unsqueeze_cast.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  return size;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                            0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void GenerateRandomData(std::vector<int64_t>& x1, int64_t& x2, int64_t total) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<int64_t> dist(-10000, 10000);
  for (int64_t i = 0; i < total; ++i) {
    x1[i] = dist(gen);
  }
  do {
    x2 = dist(gen);
  } while (x2 == 0);  // 避免除零
}

void GenerateFixData(std::vector<int64_t>& x1, int64_t& x2, int64_t total) {
  for (int64_t i = 0; i < total; ++i) {
    x1[i] = 5;
  }
  x2 = 2;
}

void GenerateData(std::vector<int64_t>& x1, int64_t& x2, int64_t total, bool fix=true) {
  if (fix) {
    GenerateFixData(x1, x2, total);
  } else {
    GenerateRandomData(x1, x2, total);
  }
}

// 使用原生 aclnnFmodScalar + aclnnCast 计算 golden
int ComputeGolden(const std::vector<int64_t>& x1Host,
                  int64_t x2Val,
                  std::vector<aclFloat16>& goldenHost,
                  aclrtStream stream) {
  const std::vector<int64_t> shape = {8, 300};
  const int64_t total = GetShapeSize(shape);

  void *x1Device = nullptr, *fmodOutDevice = nullptr, *castOutDevice = nullptr;
  aclTensor *x1Tensor = nullptr, *fmodOutTensor = nullptr, *castOutTensor = nullptr;
  aclScalar *otherScalar = nullptr;

  std::vector<int64_t> x1Data = x1Host;
  std::vector<int64_t> fmodOutHost(total, 0);
  std::vector<aclFloat16> castOutHost(total);

  // 1. 创建输入 Tensor 和 Scalar
  auto ret = CreateAclTensor(x1Data, shape, &x1Device, ACL_INT64, &x1Tensor);
  CHECK_RET(ret == 0, return ret);

  otherScalar = aclCreateScalar(&x2Val, ACL_INT64);
  CHECK_RET(otherScalar != nullptr, LOG_PRINT("aclCreateScalar failed\n"); return -1);

  // 2. Fmod 输出（保持 int64）
  ret = CreateAclTensor(fmodOutHost, shape, &fmodOutDevice, ACL_INT64, &fmodOutTensor);
  CHECK_RET(ret == 0, return ret);

  // 3. Cast 输出（float16）
  ret = CreateAclTensor(castOutHost, shape, &castOutDevice, ACL_FLOAT16, &castOutTensor);
  CHECK_RET(ret == 0, return ret);

  // ---------- 调用 aclnnFmodScalar ----------
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnFmodScalarGetWorkspaceSize(x1Tensor, otherScalar, fmodOutTensor,
                                        &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnFmodScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnFmodScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFmodScalar failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // ---------- 调用 aclnnCast (int64 -> float16) ----------
  workspaceSize = 0;
  executor = nullptr;
  ret = aclnnCastGetWorkspaceSize(fmodOutTensor, ACL_FLOAT16, castOutTensor,
                                  &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  if (workspaceSize > 0) {
    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
      workspaceAddr = nullptr;
    }
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 拷回 host
  ret = aclrtMemcpy(goldenHost.data(), total * sizeof(aclFloat16),
                    castOutDevice, total * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy golden failed. ERROR: %d\n", ret); return ret);

  // 释放资源
  aclDestroyTensor(x1Tensor);
  aclDestroyTensor(fmodOutTensor);
  aclDestroyTensor(castOutTensor);
  aclDestroyScalar(otherScalar);
  aclrtFree(x1Device);
  aclrtFree(fmodOutDevice);
  aclrtFree(castOutDevice);
  if (workspaceAddr) aclrtFree(workspaceAddr);

  return 0;
}

// 调用自定义算子 ModUnsqueezeCast
int ComputeCustom(const std::vector<int64_t>& x1Host,
                  int64_t x2Val,
                  std::vector<aclFloat16>& outHost,
                  aclrtStream stream) {
  const std::vector<int64_t> shapeIn  = {8, 300};
  const std::vector<int64_t> shapeOut = {8, 300, 1};  // unsqueeze 后的 shape
  const int64_t total = GetShapeSize(shapeIn);

  void *x1Device = nullptr, *x2Device = nullptr, *yDevice = nullptr;
  aclTensor *x1Tensor = nullptr, *x2Tensor = nullptr, *yTensor = nullptr;

  std::vector<int64_t> x1Data = x1Host;
  std::vector<int64_t> x2Data = {x2Val};
  std::vector<aclFloat16> yData(total);

  auto ret = CreateAclTensor(x1Data, shapeIn, &x1Device, ACL_INT64, &x1Tensor);
  CHECK_RET(ret == 0, return ret);
  ret = CreateAclTensor(x2Data, {1}, &x2Device, ACL_INT64, &x2Tensor);
  CHECK_RET(ret == 0, return ret);
  ret = CreateAclTensor(yData, shapeOut, &yDevice, ACL_FLOAT16, &yTensor);
  CHECK_RET(ret == 0, return ret);

  // 调用自定义算子两段式接口
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnModUnsqueezeCastGetWorkspaceSize(x1Tensor, x2Tensor, yTensor,
                                              &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnModUnsqueezeCastGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnModUnsqueezeCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnModUnsqueezeCast failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 拷回（数据连续，按元素比较即可）
  ret = aclrtMemcpy(outHost.data(), total * sizeof(aclFloat16),
                    yDevice, total * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy custom result failed. ERROR: %d\n", ret); return ret);

  // 释放
  aclDestroyTensor(x1Tensor);
  aclDestroyTensor(x2Tensor);
  aclDestroyTensor(yTensor);
  aclrtFree(x1Device);
  aclrtFree(x2Device);
  aclrtFree(yDevice);
  if (workspaceAddr) aclrtFree(workspaceAddr);

  return 0;
}

// 精度比对
bool CompareResult(const std::vector<aclFloat16>& custom,
                   const std::vector<aclFloat16>& golden,
                   float atol = 1e-3f, float rtol = 1e-3f) {
  size_t total = custom.size();
  size_t failCnt = 0;
  float maxAbsErr = 0.0f;

  for (size_t i = 0; i < total; ++i) {
    float c = aclFloat16ToFloat(custom[i]);
    float g = aclFloat16ToFloat(golden[i]);
    float absErr = std::fabs(c - g);
    float relErr = absErr / (std::fabs(g) + 1e-8f);
    if (absErr > maxAbsErr) maxAbsErr = absErr;
    if (absErr > atol && relErr > rtol) {
      if (failCnt < 10) {
        LOG_PRINT("Mismatch at %zu: custom=%.6f, golden=%.6f, abs=%.6f, rel=%.6f\n",
                  i, c, g, absErr, relErr);
      }
      failCnt++;
    }
  }

  LOG_PRINT("Total elements: %zu, Fail count: %zu, Max abs error: %.6f\n",
            total, failCnt, maxAbsErr);
  return failCnt == 0;
}

int main() {
  // 1. 初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  int32_t deviceId = 0;
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  aclrtStream stream = nullptr;
  ret = aclrtCreateStream(&stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);

  // 2. 准备数据
  const std::vector<int64_t> shape = {8, 300};
  const int64_t total = GetShapeSize(shape);
  std::vector<int64_t> x1Host(total);
  int64_t x2Val = 0;
  GenerateData(x1Host, x2Val, total);
  LOG_PRINT("Test case: shape=[%ld,%ld], x2 scalar = %ld\n", shape[0], shape[1], x2Val);

  std::vector<aclFloat16> customOut(total);
  std::vector<aclFloat16> goldenOut(total);

  // 3. 自定义算子
  ret = ComputeCustom(x1Host, x2Val, customOut, stream);
  CHECK_RET(ret == 0, LOG_PRINT("ComputeCustom failed\n"); return ret);

  // std::cout << "Custom output (first 10 elements): ";
  // for (int i = 0; i < 10; ++i) {
  //   std::cout << aclFloat16ToFloat(customOut[i]) << " ";
  // }
  // std::cout << std::endl;

  // 4. Golden（FmodScalar + Cast）
  ret = ComputeGolden(x1Host, x2Val, goldenOut, stream);
  CHECK_RET(ret == 0, LOG_PRINT("ComputeGolden failed\n"); return ret);

  // 5. 精度比对
  bool pass = CompareResult(customOut, goldenOut);
  if (pass) {
    LOG_PRINT("\n[SUCCESS] Precision verification passed!\n");
  } else {
    LOG_PRINT("\n[FAILED] Precision verification failed!\n");
  }

  // 6. 清理
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return pass ? 0 : 1;
}