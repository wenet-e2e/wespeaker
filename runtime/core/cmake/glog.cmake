FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/v0.4.0.zip
  URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
)
FetchContent_GetProperties(glog)
if(NOT glog_POPULATED)
  FetchContent_Populate(glog)
  file(READ ${glog_SOURCE_DIR}/CMakeLists.txt _glog_cmake)
  # glog 0.4.0 uses cmake_minimum_required(VERSION 3.0); CMake 4+ rejects <3.5.
  string(REGEX REPLACE
    "cmake_minimum_required[ ]*\\([ ]*VERSION[ ]+[^)]+\\)"
    "cmake_minimum_required(VERSION 3.10)"
    _glog_cmake "${_glog_cmake}")
  file(WRITE ${glog_SOURCE_DIR}/CMakeLists.txt "${_glog_cmake}")
  add_subdirectory(${glog_SOURCE_DIR} ${glog_BINARY_DIR})
endif()
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})
