add_library(utils STATIC
  utils.cc
)
target_link_libraries(utils PUBLIC glog gflags)
