#include "table.h"

std::unordered_set<void*> ToDHATensorSet;

TORCH_LIBRARY(dha, m) {
  m.def("to_dha", cuda_direct_host);
}