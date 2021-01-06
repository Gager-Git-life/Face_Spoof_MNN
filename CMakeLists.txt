cmake_minimum_required(VERSION 3.10)
project(face_spoof)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(
        mnn/include
        src
)

link_directories(mnn/lib)

add_executable(face_spoof src/video.cpp src/FaceSpoof.cpp src/FaceAligner.cpp src/FaceDetect.cpp src/UltraFace.cpp)
target_link_libraries(face_spoof MNN ${OpenCV_LIBS})
