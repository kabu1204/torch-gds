#include <ATen/core/TensorBody.h>
#include <torch/detail/static.h>
#include <unordered_set>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include "c10/util/Exception.h"
#include "torch/library.h"
#include "torch/torch.h"
#include "torch/nn.h"
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include "dha_table.h"
#include <c10/util/Exception.h>
#include <stdexcept>
#include <torch/detail/static.h>
#include "allocator.h"

static inline bool is_pinned_cuda(const at::Tensor& self) {
    auto device = self.device();
    // TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.is_cpu());
    return at::detail::getCUDAHooks().isPinnedPtr(self.storage().data());
}

template<typename T>
static inline void AddDHASet(const at::Tensor& t) {
    auto p = t.data_ptr<T>();
    DHATensorSet.insert((void*)p);
}

bool isDHATensor(const at::Tensor& t) {
    bool r;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(t.scalar_type(), "is_dha_tensor", [&] {
        r = DHATensorSet.count((void*)t.data_ptr<scalar_t>());
    });
    return r;
}

at::Tensor cuda_direct_host(const at::Tensor& self) {
  if (self.is_pinned() && self.device().is_cuda()) {
    TORCH_WARN("Already '", self.toString(), "' is pinned");
    return self;
  }
  auto* allocator = at::cuda::getCachingHostAllocatorCUDA();
  auto storage = at::Storage(
      at::Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(
        self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false
      );
      storage.device();
  auto specified_options = self.options();
  specified_options = specified_options.device(at::kCUDA);

  auto tensor = at::empty({0}, specified_options).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
//   TORCH_WARN(tensor.dtype());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "to_dha", [&] {
        AddDHASet<scalar_t>(tensor);
    });
  return tensor;
}

void addTensorDHA(const at::Tensor& t){
    // if (!is_pinned_cuda(t)) {
    //     throw std::runtime_error("try to add un-pinned memory to DHA set");
    // }
    auto p = t.data_ptr<float>();
    ToDHATensorSet.insert(p);
    TORCH_WARN("DHA Set:" + std::to_string(ToDHATensorSet.size()));
}

void addModuleDHA(const torch::nn::Module& m) {
    for (auto& child : m.children()) {
        addModuleDHA(*child);
    }
    // Then move every parameter to the new dtype/device.
    for (auto& parameter : m.named_parameters(/*recurse=*/false)) {
        // TORCH_WARN("adding parameter <" + parameter.key() + "> to DHA set");
        addTensorDHA(parameter.value());
    }
    // Then move every buffer to the new dtype/device.
    for (auto& buffer : m.named_buffers(/*recurse=*/false)) {
        // TORCH_WARN("adding buffer <" + buffer.key() + "> to DHA set");
        addTensorDHA(buffer.value());
    }
}

void pinModule(torch::nn::Module& m) {
    for (auto& child : m.children()) {
        pinModule(*child);
    }
    // Then move every parameter to the new dtype/device.
    for (auto& parameter : m.named_parameters(/*recurse=*/false)) {
        // TORCH_WARN("pinning parameter <" + parameter.key() + "> at ", parameter->data_ptr());
        parameter->set_data(parameter->pin_memory());
    }
    // Then move every buffer to the new dtype/device.
    for (auto& buffer : m.named_buffers(/*recurse=*/false)) {
        // TORCH_WARN("pinning buffer <" + buffer.key() + ">");
        buffer->set_data(buffer->pin_memory());
    }
}

void pinModuleDHA(torch::nn::Module& m) {
    for (auto& child : m.children()) {
        pinModule(*child);
    }
    // Then move every parameter to the new dtype/device.
    for (auto& parameter : m.named_parameters(/*recurse=*/false)) {
        // TORCH_WARN("DHA pinning parameter <" + parameter.key() + "> at ", parameter->data_ptr());
        parameter->set_data(cuda_direct_host(parameter.value()));
    }
    // Then move every buffer to the new dtype/device.
    for (auto& buffer : m.named_buffers(/*recurse=*/false)) {
        // TORCH_WARN("DHA pinning buffer <" + buffer.key() + ">");
        buffer->set_data(cuda_direct_host(buffer.value()));
    }
}