// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2020-2022, Intel Corporation

namespace pcmm {

namespace debug {
    int currentDebugLevel = 0;

    void dyn_debug_level( int debugLevel ) {
        debug::currentDebugLevel = debugLevel;
    }
}

} // namespace pcm
