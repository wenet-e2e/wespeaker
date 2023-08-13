set(speaker_srcs
  speaker_engine.cc)

if(NOT ONNX AND NOT BPU)
  message(FATAL_ERROR "Please build with ONNX or BPU!")
endif()
if(ONNX)
  list(APPEND speaker_srcs onnx_speaker_model.cc)
else()
  list(APPEND speaker_srcs bpu_speaker_model.cc)
endif()

add_library(speaker STATIC ${speaker_srcs})
target_link_libraries(speaker PUBLIC frontend)

if(ONNX)
  target_link_libraries(speaker PUBLIC onnxruntime)
else()
  target_link_libraries(speaker PUBLIC easy_dnn dnn)
endif()

