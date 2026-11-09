# 检测模块:YOLO 推理 + 灯条角点提取
add_library(detect STATIC
    src/Detector.cpp
    src/LightBarExtractor.cpp
    src/ArmorClassifier.cpp
)

target_include_directories(detect
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(detect PUBLIC cxx_std_17)
target_link_libraries(detect PUBLIC ${OpenCV_LIBS} openvino::runtime openvino::frontend::onnx)
