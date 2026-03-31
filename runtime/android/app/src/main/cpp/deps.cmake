# Android NDK: FetchContent for gflags / glog.
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

FetchContent_Declare(gflags
  URL https://github.com/gflags/gflags/archive/refs/tags/v2.3.0.zip
  URL_HASH SHA256=ca732b5fd17bf3a27a01a6784b947cbe6323644ecc9e26bbe2117ec43bf7e13b)
FetchContent_MakeAvailable(gflags)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(WITH_GFLAGS ON CACHE BOOL "" FORCE)

FetchContent_Declare(glog
  URL https://github.com/google/glog/archive/v0.4.0.zip
  URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc)
FetchContent_GetProperties(glog)
if(NOT glog_POPULATED)
  FetchContent_Populate(glog)
  file(READ ${glog_SOURCE_DIR}/CMakeLists.txt _glog_cm)
  # glog 0.4.0: bump cmake_minimum for CMake 4+; on Android, execinfo probe can pass but link fails.
  string(REGEX REPLACE
    "cmake_minimum_required[ ]*\\([ ]*VERSION[ ]+[^)]+\\)"
    "cmake_minimum_required(VERSION 3.10)" _glog_cm "${_glog_cm}")
  if(ANDROID)
    string(REPLACE
      "check_include_file (execinfo.h HAVE_EXECINFO_H)"
      "if(ANDROID)\n  set(HAVE_EXECINFO_H 0)\nelse()\n  check_include_file (execinfo.h HAVE_EXECINFO_H)\nendif()"
      _glog_cm "${_glog_cm}")
  endif()
  file(WRITE ${glog_SOURCE_DIR}/CMakeLists.txt "${_glog_cm}")
  add_subdirectory(${glog_SOURCE_DIR} ${glog_BINARY_DIR})
endif()

include_directories(${gflags_BINARY_DIR}/include ${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})
