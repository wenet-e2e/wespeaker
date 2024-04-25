if(MNN)
  set(MNN_URL "https://github.com/alibaba/MNN/archive/976d1d7c0f916ea8a7acc3d31352789590f00b18.zip")
  set(URL_HASH "SHA256=7fcef0933992658e8725bdc1df2daff1410c8577c9c1ce838fd5d6c8c01d1ec1")

  FetchContent_Declare(mnn
    URL ${MNN_URL}
    URL_HASH ${URL_HASH}
  )

  set(MNN_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(MNN_SUPPORT_DEPRECATED_OP OFF CACHE BOOL "" FORCE)
  set(MNN_SEP_BUILD ON CACHE BOOL "" FORCE)
  set(MNN_BUILD_MINI ${MINI_LIBS} CACHE BOOL "" FORCE)  # mini version
  set(MNN_JNI OFF CACHE BOOL "" FORCE)
  set(MNN_USE_CPP11 ON CACHE BOOL "" FORCE)
  set(MNN_SUPPORT_BF16 OFF CACHE BOOL "" FORCE)
  set(MNN_BUILD_OPENCV OFF CACHE BOOL "" FORCE)
  set(MNN_LOW_MEMORY OFF CACHE BOOL "" FORCE)

  FetchContent_GetProperties(mnn)
  if(NOT mnn_POPULATED)
    message(STATUS "Downloading mnn from ${MNN_URL}")
    FetchContent_Populate(mnn)
  endif()

  message(STATUS "mnn is downloaded to ${mnn_SOURCE_DIR}")
  message(STATUS "mnn's binary dir is ${mnn_BINARY_DIR}")
  add_subdirectory(${mnn_SOURCE_DIR} ${mnn_BINARY_DIR})
  include_directories(${mnn_SOURCE_DIR}/include)
  link_directories(${mnn_BINARY_DIR})
  add_definitions(-DUSE_MNN)
endif()
