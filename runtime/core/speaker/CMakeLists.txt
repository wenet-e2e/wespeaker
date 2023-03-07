set(speaker_srcs
  speaker_engine.cc)

if(NOT ONNX)
  message(FATAL_ERROR "Please build with ONNX!")
endif()
if(ONNX)
  list(APPEND speaker_srcs onnx_speaker_model.cc)
endif()

add_library(speaker STATIC ${speaker_srcs})
target_link_libraries(speaker PUBLIC frontend)

if(ONNX)
  target_link_libraries(speaker PUBLIC onnxruntime)
endif()

