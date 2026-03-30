# Third-party deps for Android NDK (same sources as runtime/onnxruntime/cmake)
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

FetchContent_Declare(gflags
    URL https://github.com/gflags/gflags/archive/v2.2.2.zip
    URL_HASH SHA256=19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5
)
FetchContent_MakeAvailable(gflags)

# glog 的 symbolize_unittest 等在 Android/NDK 上常无法通过编译；应用仅需 libglog，无需第三方单测。
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_Declare(glog
    URL https://github.com/google/glog/archive/v0.4.0.zip
    URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
)
set(WITH_GFLAGS ON CACHE BOOL "" FORCE)

# Android NDK：execinfo.h 可能存在但 backtrace 不可用，glog 仍会走 stacktrace_generic 并编译失败。
# 在 Android 上跳过 execinfo 检测，强制 HAVE_EXECINFO_H=0（与常见 Android+glog 做法一致）。
if(ANDROID)
    FetchContent_GetProperties(glog)
    if(NOT glog_POPULATED)
        FetchContent_Populate(glog)
        file(READ ${glog_SOURCE_DIR}/CMakeLists.txt _glog_cmake)
        string(REPLACE
            "check_include_file (execinfo.h HAVE_EXECINFO_H)"
            "if(ANDROID)\n  set(HAVE_EXECINFO_H 0)\nelse()\n  check_include_file (execinfo.h HAVE_EXECINFO_H)\nendif()"
            _glog_cmake "${_glog_cmake}"
        )
        file(WRITE ${glog_SOURCE_DIR}/CMakeLists.txt "${_glog_cmake}")
        add_subdirectory(${glog_SOURCE_DIR} ${glog_BINARY_DIR})
    endif()
else()
    FetchContent_MakeAvailable(glog)
endif()

include_directories(${gflags_BINARY_DIR}/include ${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})
