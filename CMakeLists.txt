cmake_minimum_required(VERSION 3.16)
project(RMVTraining LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_WEB_DEBUG "Build the scalar Web Debug dashboard" ON)
option(BUILD_TESTING "Build the starter-project tests" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc calib3d dnn imgcodecs highgui)
find_package(OpenVINO REQUIRED)

# conda 的 OpenVINO 用了较新的 libstdc++,链接/运行时需用 conda 的 libstdc++
if(OpenVINO_DIR)
    get_filename_component(_ov_conda_lib "${OpenVINO_DIR}/../.." ABSOLUTE)
    link_directories("${_ov_conda_lib}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${_ov_conda_lib}")
endif()
find_package(Threads REQUIRED)
find_package(MVS REQUIRED)

add_subdirectory(serialPort)
add_subdirectory(camera)
add_subdirectory(detect)
add_subdirectory(tracker)
add_subdirectory(PixelEKF)
if(ENABLE_WEB_DEBUG)
    add_subdirectory(web_debug)
endif()

add_library(training_core STATIC
    src/AppSupport.cpp
    src/AutoAimTypes.cpp
    student/src/StudentTasks.cpp
)
target_include_directories(training_core
    PUBLIC
        include
        student/include
        serialPort/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(training_core PUBLIC serialPort camera detect tracker ${OpenCV_LIBS})

add_executable(rmv_training main.cpp)
target_include_directories(rmv_training PRIVATE web_debug/include)
target_link_libraries(rmv_training
    PRIVATE
        training_core
        Threads::Threads
)
if(ENABLE_WEB_DEBUG)
    target_link_libraries(rmv_training PRIVATE web_debug)
endif()

# 灯条提取调试:实时相机 + 检测,可视化二值化/fitLine/端点/匹配框,trackbar 手动调参。
add_executable(lightbar_debug lightbar_debug.cpp)
target_link_libraries(lightbar_debug PRIVATE training_core)

# 视频离线调参:读录像替代相机,带可视化窗口 + Web UI(渲染独立线程)。
add_executable(ekf_video_debug ekf_video_debug.cpp)
target_link_libraries(ekf_video_debug PRIVATE training_core)
if(ENABLE_WEB_DEBUG)
    target_link_libraries(ekf_video_debug PRIVATE web_debug)
endif()

# 我们自己的 EKF 跑视频保存可视化
add_executable(ekf_video_save ekf_video_save.cpp)
target_link_libraries(ekf_video_save PRIVATE training_core)

# 实车调试：接相机 + 串口(只读)，弹窗口(左相机+右俯视图)+Web UI，不写回串口。
add_executable(ekf_live_debug ekf_live_debug.cpp)
target_link_libraries(ekf_live_debug PRIVATE training_core)
if(ENABLE_WEB_DEBUG)
    target_link_libraries(ekf_live_debug PRIVATE web_debug)
endif()

# 像素直滤 EKF 视频离线调参：跳过 PnP，角点直接喂 PixelEKF。
add_executable(pixelekf_video_debug pixelekf_video_debug.cpp)
target_link_libraries(pixelekf_video_debug PRIVATE training_core PixelEKF)
if(ENABLE_WEB_DEBUG)
    target_link_libraries(pixelekf_video_debug PRIVATE web_debug)
endif()

if(BUILD_TESTING)
    enable_testing()
    add_executable(test_scaffold tests/test_scaffold.cpp)
    target_link_libraries(test_scaffold PRIVATE training_core)
    add_test(NAME scaffold_safety COMMAND test_scaffold)
    add_test(NAME cli_help COMMAND rmv_training --help)

    add_executable(test_ekf tests/test_ekf.cpp)
    target_link_libraries(test_ekf PRIVATE training_core)
    add_test(NAME ekf_tracking COMMAND test_ekf)
endif()
