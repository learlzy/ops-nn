#ifndef OP_API_ACLNN_MOD_UNSQUEEZE_CAST_H_
#define OP_API_ACLNN_MOD_UNSQUEEZE_CAST_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnModUnsqueezeCastGetWorkspaceSize
 * @param [in] x1: 输入tensor，int64，shape任意
 * @param [in] x2: 输入tensor，int64，标量或shape=[1]
 * @param [in] y:  输出tensor，float16，shape = x1.shape + [1]
 * @param [out] workspaceSize: 返回需要的workspace大小
 * @param [out] executor: 返回op执行器
 */
aclnnStatus aclnnModUnsqueezeCastGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, aclTensor *y,
                                                  uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnModUnsqueezeCast
 */
aclnnStatus aclnnModUnsqueezeCast(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_MOD_UNSQUEEZE_CAST_H_