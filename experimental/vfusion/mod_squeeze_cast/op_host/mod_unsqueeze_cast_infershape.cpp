#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace {
ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);

    const size_t rank = x1_shape->GetDimNum();
    y_shape->SetDimNum(rank + 1);
    for (size_t i = 0; i < rank; ++i) {
        y_shape->SetDim(i, x1_shape->GetDim(i));
    }
    y_shape->SetDim(rank, 1);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(ModUnsqueezeCast)
    .InferShape(InferShape)
    .InferDataType(InferDataType);