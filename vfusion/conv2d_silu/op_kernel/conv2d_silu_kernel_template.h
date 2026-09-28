#ifndef __CONV2D_SILU_KERNEL_TEMPLATE_H__
#define __CONV2D_SILU_KERNEL_TEMPLATE_H__


namespace NsConv2dSilu {
    using namespace Catlass;
    using namespace Catlass::Conv::Kernel;
    using namespace Catlass::Epilogue;

    template<typename ElementType>
    struct Conv2dSiluKernelTraits {
        using ElementAccumulator = half;
        using ElementFmap = ElementType;
        using ElementFilter = ElementType;
        using ElementBias = ElementType;
        using ElementOutput = ElementType;

        using ArchTag = Arch::AtlasA2;
        constexpr uint32_t L1A_STAGES = 2;
        constexpr uint32_t L1B_STAGES = 2;
        constexpr uint32_t L0A_STAGES = 2;
        constexpr uint32_t L0B_STAGES = 2;
        constexpr uint32_t L0C_STAGES = 1;
        constexpr bool ENABLE_UNIT_FLAG = false;
        
        using DispatchPolicy =
            Conv::ConvPingpong<ArchTag, L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES, L0C_STAGES, ENABLE_UNIT_FLAG>;
        using FmapL1TileShape = Catlass::Conv2dFmapL1Shape<8, 12, 8>;
        using FilterL1TileShape = Catlass::Conv2dFilterL1Shape<96, 8>;
        using L0TileShape = Catlass::Conv2dL0Shape<16, 96, 16>;
        using BlockConv2d = Conv::Block::BlockConv2dTla<DispatchPolicy, FmapL1TileShape, FilterL1TileShape,
            L0TileShape, ElementFmap, ElementFilter, ElementOutput, ElementAccumulator>;
        using BlockScheduler = typename Conv::Block::Conv2dIdentityBlockSwizzle<3, 0>;
        using StorageType = Gemm::GemmType<ElementOutput, layout::NC1HWC0>; // 存储类型始终为half（匹配AIC写入workspace的精度）
        using ComputeType = Gemm::GemmType<ElementAccumulator, layout::NC1HWC0>; // SiLU计算精度
        using FinalOutputType = Gemm::GemmType<ElementOutput, layout::NC1HWC0>;
        using TileEpilogue = Tile::TileElemWiseSilu<ArchTag, ComputeType, 256>;
        using TileCopy = Tile::TileCopy<ArchTag, StorageType, FinalOutputType>;
        using BlockEpilogue = Block::BlockEpilogue<
            EpilogueAtlasA2Conv2dElemWise, StorageType, FinalOutputType, TileEpilogue, TileCopy,
            std::integral_constant<uint32_t, FmapL1TileShape::Ho>, std::integral_constant<uint32_t, FmapL1TileShape::Wo>,
            std::integral_constant<uint32_t, FilterL1TileShape::Cout>>;
        using Conv2dKernel = Conv2dEpilogue<BlockConv2d, BlockEpilogue, BlockScheduler>;
    };
}

#endif