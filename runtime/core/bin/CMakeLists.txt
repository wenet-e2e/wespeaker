add_executable(asv_main asv_main.cc)
target_link_libraries(asv_main PUBLIC speaker)

add_executable(extract_emb_main extract_emb_main.cc)
target_link_libraries(extract_emb_main PUBLIC speaker)
