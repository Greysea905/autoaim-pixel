# FindOpenVINO.cmake
# OpenVINO 自带 OpenVINOConfig.cmake,但 conda 把它装在 lib/cmake/openvino<版本>/ 这种
# 带版本号的目录,find_package 默认搜不到。这里自动 glob 出目录,再调 find_package。

if(NOT OpenVINO_DIR)
    # 候选前缀:CONDA_PREFIX(激活的 conda 环境)、~/miniconda3/envs/*(所有 conda 环境)
    set(_ov_prefixes "")
    if(DEFINED ENV{CONDA_PREFIX})
        list(APPEND _ov_prefixes "$ENV{CONDA_PREFIX}")
    endif()
    if(DEFINED ENV{HOME})
        file(GLOB _ov_envs "$ENV{HOME}/miniconda3/envs/*")
        list(APPEND _ov_prefixes ${_ov_envs})
        list(APPEND _ov_prefixes "$ENV{HOME}/miniconda3")
    endif()

    foreach(_p IN LISTS _ov_prefixes)
        file(GLOB _ov_dirs "${_p}/lib/cmake/openvino*/")
        if(_ov_dirs)
            list(GET _ov_dirs 0 OpenVINO_DIR)
            set(OpenVINO_DIR "${OpenVINO_DIR}" CACHE PATH "OpenVINO cmake 目录" FORCE)
            break()
        endif()
    endforeach()
endif()

find_package(OpenVINO CONFIG REQUIRED)
