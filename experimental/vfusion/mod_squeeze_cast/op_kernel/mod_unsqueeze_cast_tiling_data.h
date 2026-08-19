#ifndef MOD_UNSQUEEZE_CAST_TILING_DATA_H
#define MOD_UNSQUEEZE_CAST_TILING_DATA_H
#include <cstdint>

constexpr int32_t TILE_SIZE = 256;

struct ModUnsqueezeCastTilingData {
    uint32_t size;
    uint32_t blockLength;
    uint32_t tileNum;
};
#endif