cmake_minimum_required(VERSION 3.22.1)
project(wespeaker_jni)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# runtime/core — cpp→main→src→app→android = runtime，再进入 core（共 5 段 ..）
set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core")

include(deps.cmake)

# 与 wekws/runtime/android 一致：从 Gradle 解压的 AAR 目录取 headers 与 jni（不用 Prefab find_package）
# 解压目录形如：app/build/onnxruntime-android-x.y.z.aar/
set(build_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../build")
file(GLOB ORT_ROOT_LIST "${build_DIR}/onnxruntime*.aar")
list(LENGTH ORT_ROOT_LIST _ort_len)
if(_ort_len EQUAL 0)
  message(FATAL_ERROR
    "未找到 ${build_DIR}/onnxruntime*.aar。请先执行 Gradle 任务 extractAARForNativeBuild（由 configureCMake 依赖触发）。")
endif()
list(GET ORT_ROOT_LIST 0 ORT_ROOT)
# onnxruntime-android AAR 内头文件在 headers/ 根目录（无 headers/include/ 子目录）
include_directories("${ORT_ROOT}/headers")
link_directories("${ORT_ROOT}/jni/${ANDROID_ABI}")

add_definitions(-DUSE_ONNX)

add_library(utils STATIC "${CORE_DIR}/utils/utils.cc")
target_include_directories(utils PUBLIC "${CORE_DIR}")
target_link_libraries(utils PUBLIC glog gflags)

add_library(frontend STATIC
    "${CORE_DIR}/frontend/feature_pipeline.cc"
    "${CORE_DIR}/frontend/fft.cc"
)
target_include_directories(frontend PUBLIC "${CORE_DIR}")
target_link_libraries(frontend PUBLIC utils)

add_library(speaker STATIC
    "${CORE_DIR}/speaker/speaker_engine.cc"
    "${CORE_DIR}/speaker/onnx_speaker_model.cc"
)
target_include_directories(speaker PUBLIC "${CORE_DIR}")
target_link_libraries(speaker PUBLIC frontend onnxruntime)

add_library(wespeaker_jni SHARED wespeaker_jni.cpp)
target_include_directories(wespeaker_jni PRIVATE "${CORE_DIR}")
target_link_libraries(wespeaker_jni
    PRIVATE
    speaker
    onnxruntime
    glog
    gflags
    log
)
