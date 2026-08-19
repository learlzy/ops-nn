#include "mod_unsqueeze_cast.h"
#include "mod_unsqueeze_cast_tiling_data.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void mod_unsqueeze_cast(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ModUnsqueezeCastTilingData);
    GET_TILING_DATA_WITH_STRUCT(ModUnsqueezeCastTilingData, tilingData, tiling);

    KernelModUnsqueezeCast op;
    op.Init(x1, x2, y, &tilingData);
    op.Process();
}