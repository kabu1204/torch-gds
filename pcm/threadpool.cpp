// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2020-2022, Intel Corporation

#include "threadpool.h"

namespace pcmm {

void ThreadPool::execute( ThreadPool* tp ) {
    while( 1 ) {
        Work* w = tp->retrieveWork();
        if ( w == nullptr ) break;
        w->execute();
        delete w;
    }
}

} // namespace pcm
