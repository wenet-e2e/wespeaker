if(ONNX)
 add_library(speaker_api SHARED speaker_api.cc)
 target_link_libraries(speaker_api PUBLIC speaker)
endif()