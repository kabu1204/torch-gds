file(GLOB LIBPCM_SRC 
    msr.cpp
    cpucounters.cpp
    pci.cpp
    mmio.cpp
    bw.cpp
    utils.cpp
    topology.cpp
    debug.cpp
    threadpool.cpp
    uncore_pmu_discovery.cpp
    # UNUX_SOURCES
    dashboard.cpp
    resctrl.cpp
)

add_library(PCM_STATIC STATIC ${LIBPCM_SRC})
set_target_properties(PCM_STATIC PROPERTIES OUTPUT_NAME pcm)