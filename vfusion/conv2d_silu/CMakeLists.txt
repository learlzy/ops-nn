# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE conv2d_silu ACLNNTYPE aclnn)

# ============================================================
# 添加 catlass 头文件路径（关键修改点）
# ============================================================
# catlass 放在 ops-nn 根目录下的 third_party/catlass，也可以写成相对路径：
set(CATLASS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/catlass/include" CACHE PATH "Catlass include directory")

# ============================================================
# 把 catlass 路径注入到本算子的 Kernel 编译选项中
# ============================================================
# 这行会让编译 kernel 时自动带上 -I${CATLASS_INCLUDE_DIR}
# 之后你在 op_kernel/*.cpp 或 *.h 里就可以直接 #include "catlass/xxx.hpp"
add_ops_compile_options(
    OP_NAME conv2d_silu
    OPTIONS -I${CATLASS_INCLUDE_DIR}
)