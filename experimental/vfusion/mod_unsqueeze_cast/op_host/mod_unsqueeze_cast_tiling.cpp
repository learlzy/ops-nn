#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../op_kernel/mod_unsqueeze_cast_tiling_data.h"

using namespace ge;

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ModUnsqueezeCastTilingData *tiling = context->GetTilingData<ModUnsqueezeCastTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);

    int64_t data_sz = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); ++i) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }
    tiling->size = static_cast<uint32_t>(data_sz);

    constexpr uint32_t BLOCK_DIM = 4;
    context->SetBlockDim(BLOCK_DIM);
    tiling->blockLength = (tiling->size + BLOCK_DIM - 1) / BLOCK_DIM;
    tiling->tileNum = (tiling->blockLength + TILE_SIZE - 1) / TILE_SIZE;   // TILE_SIZE=256

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

IMPL_OP_OPTILING(ModUnsqueezeCast)
    .Tiling(optiling::TilingFunc);
    