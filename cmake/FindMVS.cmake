# FindMVS.cmake
#
# 查找海康机器人 MVS SDK 的头文件和库，并导出导入目标 MVS::MVS。
#
# 用法（在主 CMakeLists.txt 中）：
#   list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
#   find_package(MVS REQUIRED)
#   target_link_libraries(<target> PRIVATE MVS::MVS)
#
# SDK 根目录查找顺序：
#   1. 命令行缓存变量：  -DMVS_ROOT=/path/to/MVS
#   2. 环境变量：        export MVS_ROOT=/path/to/MVS
#   3. 默认安装位置：    /opt/MVS
#
# 输出变量：
#   MVS_FOUND        是否找到（true/false）
#   MVS_INCLUDE_DIRS 头文件目录
#   MVS_LIBRARIES    需要链接的库
#   MVS_LIBRARY_DIR  库目录（可用于设置运行时 rpath / LD_LIBRARY_PATH）
#   MVS::MVS         导入目标（推荐直接链接它）

# 允许通过缓存变量指定 SDK 根目录；若已通过 -DMVS_ROOT 传入则不会覆盖
set(MVS_ROOT "" CACHE PATH "MVS SDK 安装根目录（例如 /opt/MVS）")

# 命令行未指定时，尝试环境变量
if(NOT MVS_ROOT)
  set(MVS_ROOT "$ENV{MVS_ROOT}")
endif()

# 构造候选根目录列表
set(_MVS_SEARCH_ROOTS)
if(MVS_ROOT)
  list(APPEND _MVS_SEARCH_ROOTS "${MVS_ROOT}")
endif()
list(APPEND _MVS_SEARCH_ROOTS "/opt/MVS")

# 按指针宽度选择 64 位或 32 位库目录
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(_MVS_LIB_SUBDIR "lib/64")
else()
  set(_MVS_LIB_SUBDIR "lib/32")
endif()

find_path(MVS_INCLUDE_DIR
  NAMES MvCameraControl.h
  PATHS ${_MVS_SEARCH_ROOTS}
  PATH_SUFFIXES include
)

find_library(MVS_LIBRARY
  NAMES MvCameraControl
  PATHS ${_MVS_SEARCH_ROOTS}
  PATH_SUFFIXES ${_MVS_LIB_SUBDIR}
)

# 提取库所在目录，供设置运行时搜索路径使用
if(MVS_LIBRARY)
  get_filename_component(MVS_LIBRARY_DIR "${MVS_LIBRARY}" DIRECTORY)
endif()

# 统一处理 REQUIRED / QUIET / VERSION 语义，并给出未找到时的报错信息
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MVS
  REQUIRED_VARS MVS_LIBRARY MVS_INCLUDE_DIR
)

# 找到后创建导入目标，供 target_link_libraries 使用
if(MVS_FOUND AND NOT TARGET MVS::MVS)
  add_library(MVS::MVS UNKNOWN IMPORTED)
  set_target_properties(MVS::MVS PROPERTIES
    IMPORTED_LOCATION "${MVS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MVS_INCLUDE_DIR}"
  )
endif()

# 供需要手动使用变量（而非导入目标）的旧式写法使用
set(MVS_INCLUDE_DIRS "${MVS_INCLUDE_DIR}")
set(MVS_LIBRARIES "${MVS_LIBRARY}")

mark_as_advanced(MVS_ROOT MVS_INCLUDE_DIR MVS_LIBRARY)
