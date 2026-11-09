add_library(tracker STATIC
    src/BulletModel.cpp
    src/EKFEstimator.cpp
    src/Planner.cpp
)
target_include_directories(tracker PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(tracker PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
