#ifndef TORCH_GDS_PCIE_MONITOR
#define TORCH_GDS_PCIE_MONITOR

#include "pcm/cpucounters.h"
#include "pcm/types.h"
#include <chrono>
#include <mutex>
#include <sys/types.h>
#include <vector>
#include <unordered_map>
#include <cstdint>

using namespace pcmm;

enum PCIeEventAggr {
    PCIRdCur,
    ItoM,
    ItoMCacheNear,
    UCRdF,
    WiL,
};

enum PCIeEventIdx {
    PCIRdCur_miss,
    PCIRdCur_hit,
    ItoM_miss,
    ItoM_hit,
    ItoMCacheNear_miss,
    ItoMCacheNear_hit,
    UCRdF_miss,
    WiL_miss,
    eventLast
};

class PCIeMonitor {
public:
    static PCIeMonitor* Instance();
    void StartGroup(uint32_t groupIdx);
    void StopGroup(uint32_t groupIdx);
    void Clear();
    uint64_t Event(PCIeEventIdx event);
    uint64_t EventAggr(PCIeEventAggr event);
    uint64_t Event(uint32_t socket, PCIeEventIdx event);
    uint64_t EventAggr(uint32_t socket, PCIeEventAggr event);
    uint64_t GetReadAccessCounter();
    uint64_t GetReadBW();
    uint64_t GetDurationNs() const;
private:
    PCIeMonitor();
    void getGroupEventRecords(uint32_t run, uint32 groupIdx, uint32_t offset);
    uint64_t getEventRecords(uint32_t socket, uint32_t eventIdx);
    
    enum {
        before,
        after,
        total,
    };

    using StdHiresClock = std::chrono::high_resolution_clock;

    // std::mutex mutex;   // protect instance
    // static PCIeMonitor* instance;
    PCM* m;
    std::vector<eventGroup_t> evGroups;
    uint32_t numSocket;
    std::array<std::vector<std::vector<uint64_t>>, total> evRecords;
    std::vector<std::vector<uint64_t>> evSamples;
    std::chrono::time_point<StdHiresClock> timeStart;
    uint64_t durationNs;
};


#endif