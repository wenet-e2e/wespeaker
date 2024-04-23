set(speaker_srcs
  speaker_engine.cc)

if(NOT ONNX AND NOT MNN)
  message(FATAL_ERROR "Please build with ONNX or MNN!")
endif()
if(ONNX)
  list(APPEND speaker_srcs onnx_speaker_model.cc)
endif()
if(MNN)
  list(APPEND speaker_srcs mnn_speaker_model.cc)
endif()

add_library(speaker STATIC ${speaker_srcs})
target_link_libraries(speaker PUBLIC frontend)

if(ONNX)
  target_link_libraries(speaker PUBLIC onnxruntime)
endif()
if(MNN)
  target_link_libraries(speaker PUBLIC MNN)
endif()

