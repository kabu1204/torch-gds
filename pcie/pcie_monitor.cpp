#include "pcie_monitor.h"
#include "pcm/cpucounters.h"
#include "pcm/types.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ratio>
#include <sys/types.h>

// WhitleyPlatform
PCIeMonitor::PCIeMonitor()
    : evGroups({
        {0xC8F3FE00000435, 0xC8F3FD00000435, 0xCC43FE00000435, 0xCC43FD00000435},
        {0xCD43FE00000435, 0xCD43FD00000435, 0xC877DE00000135, 0xC87FDE00000135}
    })
{
    m = PCM::getInstance();
    numSocket = m->getNumSockets();
    evSamples.resize(numSocket);
    
    int evCount = 0;
    for (auto &g:evGroups) evCount += g.size();

    // for each CPU socket
    for (auto &evs: evSamples) evs.resize(evCount);

    for (auto &run : evRecords) {
        run.resize(numSocket);
        // for each CPU socket
        for (auto &ev : run)
            ev.resize(evCount);
    }
}

PCIeMonitor* PCIeMonitor::Instance() {
    static PCIeMonitor instance;
    return &instance;
}

void PCIeMonitor::Clear() {
    for(auto& socket : evSamples)
        std::fill(socket.begin(), socket.end(), 0);
}

void PCIeMonitor::getGroupEventRecords(uint32_t run, uint32 groupIdx, uint32_t offset) {
    if (run >= total) {
        return;
    }

    eventGroup_t &group = evGroups[groupIdx];
    if (run == before) {
        m->programPCIeEventGroup(group);
    }

    for (uint32_t skt = 0; skt < numSocket; ++skt)
        for (uint32_t ctr = 0; ctr < group.size(); ++ctr) {
            evRecords[run][skt][ctr + offset] = m->getPCIeCounterData(skt, ctr);
            // printf("%lu, %lu, %lu\n", ctr, offset, evRecords[run][skt][ctr + offset]);
        }

    if (run == after) {
        for(uint32_t skt = 0; skt < numSocket; ++skt)
            for (uint32_t idx = offset; idx < offset + group.size(); ++idx)
                evSamples[skt][idx] += getEventRecords(skt, idx);
    }
}

uint64_t PCIeMonitor::getEventRecords(uint32_t socket, uint32_t eventIdx) {
    return evRecords[after][socket][eventIdx] - evRecords[before][socket][eventIdx];
}

void PCIeMonitor::StartGroup(uint32_t groupIdx) {
    uint32_t offset = 0;
    if (groupIdx >= evGroups.size()) {
        return;
    }
    for (auto it = evGroups.begin(); it != evGroups.begin() + groupIdx; it++) {
        offset += it->size();
    }
    timeStart = StdHiresClock::now();
    getGroupEventRecords(before, groupIdx, offset);
}

void PCIeMonitor::StopGroup(uint32_t groupIdx) {
    uint32_t offset = 0;
    if (groupIdx >= evGroups.size()) {
        return;
    }
    for (auto it = evGroups.begin(); it != evGroups.begin() + groupIdx; it++) {
        offset += it->size();
    }
    getGroupEventRecords(after, groupIdx, offset);
    auto timeStop = StdHiresClock::now();
    durationNs = std::chrono::duration_cast<std::chrono::nanoseconds>(timeStop - timeStart).count();
}

uint64_t PCIeMonitor::GetReadAccessCounter() {
    return EventAggr(PCIRdCur);
}

uint64_t PCIeMonitor::GetReadBW() {
    return (GetReadAccessCounter() * 64ULL) / (double(durationNs)/1e9);
}

uint64_t PCIeMonitor::GetDurationNs() const {
    return durationNs;
}

uint64_t PCIeMonitor::Event(PCIeEventIdx event) {
    uint64_t count = 0;
    for (uint32_t socket = 0; socket<numSocket; ++socket) {
        count += Event(socket, event);
    }
    return count;
}

uint64_t PCIeMonitor::EventAggr(PCIeEventAggr event) {
    uint64_t count = 0;
    for (uint32_t socket = 0; socket < numSocket; ++socket) {
        count += EventAggr(socket, event);
    }
    return count;
}

uint64_t PCIeMonitor::Event(uint32_t socket, PCIeEventIdx event) {
    return evSamples[socket][event];
}

uint64_t PCIeMonitor::EventAggr(uint32_t socket, PCIeEventAggr event) {
    uint64 count = 0;
    switch (event)
    {
        case PCIRdCur:
            count = evSamples[socket][PCIRdCur_miss] +
                    evSamples[socket][PCIRdCur_hit];
            break;
        case ItoM:
            count = evSamples[socket][ItoM_miss] +
                    evSamples[socket][ItoM_hit];
            break;
        case ItoMCacheNear:
            count = evSamples[socket][ItoMCacheNear_miss] +
                    evSamples[socket][ItoMCacheNear_hit];
            break;
        case UCRdF:
            count = evSamples[socket][UCRdF_miss];
            break;
        case WiL:
            count = evSamples[socket][WiL_miss];
            break;
        default:
            break;
    }
    return count;
}