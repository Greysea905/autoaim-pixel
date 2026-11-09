add_library(PixelEKF STATIC
    src/PixelEKF.cpp
)
target_include_directories(PixelEKF PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tracker/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(PixelEKF PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
