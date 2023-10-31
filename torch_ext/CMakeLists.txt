# add_library(dhatable SHARED table.cpp)
# target_compile_features(dhatable PRIVATE cxx_std_14)
# target_link_libraries(dhatable "${TORCH_LIBRARIES}")


add_library(mycopy SHARED copy.cpp)
target_compile_features(mycopy PRIVATE cxx_std_14)
target_link_libraries(mycopy "${TORCH_LIBRARIES}")

add_library(myops SHARED ops.cpp)
target_compile_features(myops PRIVATE cxx_std_14)
target_link_libraries(myops "${TORCH_LIBRARIES}")