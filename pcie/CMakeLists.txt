add_library(pcie_monitor STATIC pcie_monitor.cpp)
target_link_libraries(pcie_monitor PCM_STATIC)
# set_target_properties(pcie_monitor PROPERTIES COMPILE_FLAGS "-save-temps") 