
/*!
 * \file mod_unsqueeze_cast.h
 * \brief Kernel implementation of ModUnsqueezeCast
 */
#ifndef MOD_UNSQUEEZE_CAST_H
#define MOD_UNSQUEEZE_CAST_H

#include "kernel_operator.h"
#include "mod_unsqueeze_cast_tiling_data.h"

namespace NsModUnsqueezeCast {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelModUnsqueezeCast {
public:
    __aicore__ inline KernelModUnsqueezeCast() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const ModUnsqueezeCastTilingData* tilingData)
    {
        this->blockLength = tilingData->blockLength;
        this->tileNum     = tilingData->tileNum;
        this->tileLength  = TILE_SIZE;
        uint32_t size     = tilingData->size;

        // 当前核处理的数据范围
        uint32_t blockIdx = GetBlockIdx();
        uint32_t globalOffset = blockIdx * blockLength;
        uint32_t remain = size - globalOffset;
        this->curBlockLength = (remain < blockLength) ? remain : blockLength;

        x1Gm.SetGlobalBuffer((__gm__ int64_t*)x1 + globalOffset, curBlockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + globalOffset, curBlockLength);  // 输出连续，unsqueeze 只改 shape

        // x2 是 scalar，只读一次
        x2Gm.SetGlobalBuffer((__gm__ int64_t*)x2, 1);
        scalarVal = x2Gm.GetValue(0);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(int64_t));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf, tileLength * sizeof(float));   // 中间 float 缓冲（cast 用）
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount = (curBlockLength + tileLength - 1) / tileLength;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t curTileLen = (i == loopCount - 1) ?
                (curBlockLength - i * tileLength) : tileLength;
            CopyIn(i, curTileLen);
            Compute(curTileLen);
            CopyOut(i, curTileLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t len)
    {
        LocalTensor<int64_t> xLocal = inQueueX.AllocTensor<int64_t>();
        DataCopy(xLocal, x1Gm[progress * tileLength], len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        LocalTensor<int64_t> xLocal = inQueueX.DeQue<int64_t>();
        LocalTensor<half>    yLocal = outQueueY.AllocTensor<half>();
        LocalTensor<float>   tmpLocal = tmpBuf.Get<float>();

        // 1. int64 mod（标量）
        // Ascend C 对 int64 的向量 mod 支持有限，这里用标量循环保证正确性
        // （数据量小 [8,300]=2400，性能可接受；若后续要极致性能可换 int32 路径）
        for (uint32_t i = 0; i < len; ++i) {
            int64_t v = xLocal.GetValue(i);
            int64_t m = v % scalarVal;
            // 向正数方向调整（与常见框架行为对齐，可按需求修改）
            if (m < 0) {
                m += (scalarVal > 0 ? scalarVal : -scalarVal);
            }
            tmpLocal.SetValue(i, static_cast<float>(m));
        }

        // 2. float -> half
        Cast(yLocal, tmpLocal, RoundMode::CAST_NONE, len);

        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t len)
    {
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        DataCopy(yGm[progress * tileLength], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuf;

    GlobalTensor<int64_t> x1Gm;
    GlobalTensor<int64_t> x2Gm;
    GlobalTensor<half>    yGm;

    int64_t  scalarVal;
    uint32_t blockLength;
    uint32_t curBlockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

} // namespace NsModUnsqueezeCast

#endif // MOD_UNSQUEEZE_CAST_H