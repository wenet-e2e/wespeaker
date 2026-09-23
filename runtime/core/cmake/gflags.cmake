FetchContent_Declare(gflags
  URL      https://github.com/gflags/gflags/archive/refs/tags/v2.3.0.zip
  URL_HASH SHA256=ca732b5fd17bf3a27a01a6784b947cbe6323644ecc9e26bbe2117ec43bf7e13b
)
FetchContent_MakeAvailable(gflags)
include_directories(${gflags_BINARY_DIR}/include)