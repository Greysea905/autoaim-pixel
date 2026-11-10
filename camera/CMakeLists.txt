# 相机模块：封装海康 MVS SDK 的取流与参数读写。
# 编译成静态库，供 training_core（StudentTasks.cpp 的 get_pic）使用。
add_library(camera STATIC
    src/MvsCamera.cpp
)

target_include_directories(camera
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_compile_features(camera PUBLIC cxx_std_17)

# MVS::MVS 由 cmake/FindMVS.cmake 提供，默认在 /opt/MVS 查找。
# OpenCV：Frame 结构体用 cv::Mat（零拷贝），需要 opencv 头文件 + core 库。
target_link_libraries(camera PUBLIC MVS::MVS ${OpenCV_LIBS})
