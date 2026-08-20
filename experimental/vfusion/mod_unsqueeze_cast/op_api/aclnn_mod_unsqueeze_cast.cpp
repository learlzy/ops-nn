#include "aclnn_mod_unsqueeze_cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> DTYPE_SUPPORT_X = {DataType::DT_INT64};
static const std::initializer_list<DataType> DTYPE_SUPPORT_Y = {DataType::DT_FLOAT16};

static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *y)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *y)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_X, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_X, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, DTYPE_SUPPORT_Y, return false);
    return true;
}

static bool CheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *y)
{
    // x2 必须是标量或长度为1的1维tensor
    auto x2Shape = x2->GetViewShape();
    if (x2Shape.GetDimNum() > 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 dim num must be 0 or 1, but got %zu", x2Shape.GetDimNum());
        return false;
    }
    if (x2Shape.GetDimNum() == 1 && x2Shape.GetDim(0) != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 shape[0] must be 1 when dimNum==1");
        return false;
    }

    // y 的 shape 应该是 x1 的 shape 后面多一个 1
    auto x1Shape = x1->GetViewShape();
    auto yShape = y->GetViewShape();
    if (yShape.GetDimNum() != x1Shape.GetDimNum() + 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y dimNum should be x1.dimNum+1");
        return false;
    }
    for (size_t i = 0; i < x1Shape.GetDimNum(); ++i) {
        if (yShape.GetDim(i) != x1Shape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y shape mismatch with x1 at dim %zu", i);
            return false;
        }
    }
    if (yShape.GetDim(x1Shape.GetDimNum()) != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y last dim must be 1");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *y)
{
    CHECK_RET(CheckNotNull(x1, x2, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x1, x2, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x1, x2, y), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnModUnsqueezeCastGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, aclTensor *y,
                                                  uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnModUnsqueezeCast, DFX_IN(x1, x2), DFX_OUT(y));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x1, x2, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x1->IsEmpty() || y->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 转成连续tensor
    auto x1Contiguous = l0op::Contiguous(x1, uniqueExecutor.get());
    CHECK_RET(x1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto x2Contiguous = l0op::Contiguous(x2, uniqueExecutor.get());
    CHECK_RET(x2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用底层算子（框架会根据OpDef自动找到对应的AICore实现）
    auto modUnsqueezeCastOut = l0op::ModUnsqueezeCast(x1Contiguous, x2Contiguous, uniqueExecutor.get());
    CHECK_RET(modUnsqueezeCastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(modUnsqueezeCastOut, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnModUnsqueezeCast(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnModUnsqueezeCast);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif