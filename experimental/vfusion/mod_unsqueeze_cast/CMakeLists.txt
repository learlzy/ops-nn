
add_modules_sources(
    HOSTNAME ${OPHOST_NAME}
    MODE PRIVATE
    DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE mod_unsqueeze_cast
    ACLNNTYPE aclnn
)