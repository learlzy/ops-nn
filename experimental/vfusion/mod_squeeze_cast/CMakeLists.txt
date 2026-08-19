
# 算子根目录 CMakeLists.txt
# 统一使用 ops-nn 标准构建方式

add_modules_sources(
    HOSTNAME ${OPHOST_NAME}
    MODE PRIVATE
    DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE mod_unsqueeze_cast
    ACLNNTYPE aclnn_exclude
)