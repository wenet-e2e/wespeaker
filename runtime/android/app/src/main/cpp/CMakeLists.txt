cmake_minimum_required(VERSION 3.22.1)
project(wespeaker_jni)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# runtime/core: from this dir, five levels up to runtime/, then into core/
set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core")

include(deps.cmake)

# Same as wekws/runtime/android: headers and jni from Gradle-extracted AAR (no Prefab find_package).
# Typical path: app/build/onnxruntime-android-x.y.z.aar/
set(build_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../build")
file(GLOB ORT_ROOT_LIST "${build_DIR}/onnxruntime*.aar")
list(LENGTH ORT_ROOT_LIST _ort_len)
if(_ort_len EQUAL 0)
  message(FATAL_ERROR
    "No ${build_DIR}/onnxruntime*.aar found. Run Gradle task extractAARForNativeBuild first (pulled in before configureCMake).")
endif()
list(GET ORT_ROOT_LIST 0 ORT_ROOT)
# onnxruntime-android AAR: headers live under headers/ (not headers/include/).
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
