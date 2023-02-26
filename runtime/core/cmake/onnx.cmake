if(ONNX)
  set(ONNX_VERSION "1.12.0")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz")
    set(URL_HASH "SHA256=5820d9f343df73c63b6b2b174a1ff62575032e171c9564bcf92060f46827d0ac")
  else()
    set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz")
    set(URL_HASH "SHA256=5d503ce8540358b59be26c675e42081be14a3e833a5301926f555451046929c5")
  endif()

  FetchContent_Declare(onnxruntime
    URL ${ONNX_URL}
    URL_HASH ${URL_HASH}
  )
  FetchContent_MakeAvailable(onnxruntime)
  include_directories(${onnxruntime_SOURCE_DIR}/include)
  link_directories(${onnxruntime_SOURCE_DIR}/lib)

  add_definitions(-DUSE_ONNX)
endif()