#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Optional.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <c10/core/Storage.h>
#include <ATen/CPUFunctions.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_set>
#include "c10/util/Exception.h"

#define PER_OPS_HEADER

#ifndef PER_OPS_HEADER
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_autocast_to_full_precision_native.h>
#include <ATen/ops/_autocast_to_reduced_precision_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_sparse_bsc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_bsr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/_to_copy_native.h>
#include <ATen/ops/_to_cpu_native.h>
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/empty_strided_native.h>
#include <ATen/ops/to_dense_backward_native.h>
#include <ATen/ops/to_dense_native.h>
#include <ATen/ops/to_mkldnn_backward_native.h>
#include <ATen/ops/to_native.h>
#include <ATen/ops/to_sparse_bsc_native.h>
#include <ATen/ops/to_sparse_bsr_native.h>
#include <ATen/ops/to_sparse_csc_native.h>
#include <ATen/ops/to_sparse_csr_native.h>
#include <ATen/ops/to_sparse_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
// #include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorConversions.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <algorithm>
#include <numeric>
#include "dha_table.h"
#include "table.h"

std::unordered_set<void*> ToDHATensorSet;  // data pointers of tensors to be pinned to DHA
std::unordered_set<void*> DHATensorSet;    // data pointers of tensors already pinned DHA

namespace at {
namespace native {

static inline Device ensure_has_index(Device device) {
  if (device.is_cpu() || device.has_index()) {
    return device;
  }
  const c10::impl::DeviceGuardImplInterface* impl = c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

static inline optional<Device> ensure_has_index(optional<Device> device) {
  if (!device.has_value()) {
    return nullopt;
  }
  return ensure_has_index(device.value());
}

// Tensor _pin_memory_cuda(const Tensor& self, c10::optional<Device> device) {
//   TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!device.has_value() || device->is_cuda());
//   auto* allocator = at::cuda::getPinnedMemoryAllocator();
//   auto storage = Storage(
//       Storage::use_byte_size_t(),
//       detail::computeStorageNbytes(
//           self.sizes(), self.strides(), self.dtype().itemsize()),
//       allocator,
//       /*resizable=*/false);
//   auto tensor = at::cpu::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
//   tensor.copy_(self);
//   TORCH_WARN("pin_memory from ", self.data_ptr(), " to ", tensor.data_ptr());
//   return tensor;
// }

Tensor _to_copy(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(!layout.has_value() || self.layout() == layout.value(),
           "to(options) doesn't support converting to a different layout, "
           "but got self.layout being ", self.layout(),
           " and options.layout set as ", layout.value());
  auto options = TensorOptions()
    .dtype(dtype)
    .layout(layout)
    .device(device)
    .pinned_memory(pin_memory);

  if (options.has_device()) {
    options = options.device(ensure_has_index(options.device()));
  }
  // memory_format is handled separately due to MemoryFormat::Preserve logic
  options = self.options().merge_in(options).memory_format(c10::nullopt);
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);


  int is_dha = ToDHATensorSet.count(static_cast<float*>(self.data_ptr()));
  char fmt_buf[4096];
  snprintf(fmt_buf, sizeof(fmt_buf), 
          "[MY COPY] src_device=%s, dst_device=%s, data_ptr=%p, #DHASet=%zu, isDHA=%d", 
                self.device().str().c_str(),
                device.value().str().c_str(),
                static_cast<float*>(self.data_ptr()),
                ToDHATensorSet.size(),
                is_dha);
  TORCH_WARN(std::string(fmt_buf));

  if (is_dha) {
    TORCH_WARN("Using direct-host-access");
    return cuda_direct_host(self);
  }

  // TODO: Use the dispatcher for this.
  // Currently there are unenumerated extensibility issues preventing this.
  if (at::sparse_csr::is_sparse_compressed(self)) {
      TORCH_CHECK(
          memory_format == MemoryFormat::Preserve,
          "to(options): ", at::sparse_csr::layoutToString(self.layout()),
          " only supports memory format Preserve, but got ", memory_format,
          " instead.");

      Tensor compressed_indices, plain_indices;
      std::tie(compressed_indices, plain_indices) = at::sparse_csr::getCompressedPlainIndices(self);

      const auto new_values = at::native::to(
          self.values(),
          dtype,
          c10::kStrided,
          device,
          pin_memory,
          non_blocking,
          true, // force copy since we are in _to_copy
          memory_format);

      const auto new_compressed_indices = at::native::to(
          compressed_indices,
          compressed_indices.scalar_type(),
          c10::kStrided,
          device,
          pin_memory,
          non_blocking,
          true, // force copy since we are in _to_copy
          memory_format);

      const auto new_plain_indices = at::native::to(
          plain_indices,
          plain_indices.scalar_type(),
          c10::kStrided,
          device,
          pin_memory,
          non_blocking,
          true, // force copy since we are in _to_copy
          memory_format);

    return at::native::_sparse_compressed_tensor_unsafe(
        new_compressed_indices,
        new_plain_indices,
        new_values,
        self.sizes(),
        new_values.scalar_type(),
        self.layout(),
        new_values.device());
  }

  bool pin_out = (non_blocking && self.is_cuda() && options.device().is_cpu() &&
                  (options.layout() == c10::kStrided));

  if (memory_format == MemoryFormat::Preserve) {
    // TORCH_WARN("preserve");
    if (options.device().supports_as_strided()) {
      if (self.is_non_overlapping_and_dense()) {
        // likely this branch
        Tensor r;
        if (self.is_quantized()) {
          r = at::empty_quantized(self.sizes(), self, options);
          at::QuantizerPtr quantizer = r.quantizer();
          r.copy_(self, non_blocking);
          set_quantizer_(r, quantizer);
        } else {
          TensorOptions opt_ = options.pinned_memory(pin_out);
          // TORCH_WARN("non-quantized, pin_memory=", opt_.pinned_memory(), " device=", opt_.device());
          r = at::empty_strided(
              self.sizes(),
              self.strides(),
              opt_);
          r.copy_(self, non_blocking);
        }
        return r;
      } else if (!self.is_quantized() && self.layout() == kStrided) {
          Tensor r;
          auto strides = infer_dense_strides(self.sizes(), self.strides());
          r = at::empty_strided(
              self.sizes(),
              strides,
              options.pinned_memory(pin_out));
          r.copy_(self, non_blocking);
          return r;
      } else {
        memory_format = self.suggest_memory_format();
      }
    } else {
      // TORCH_WARN("un strided");
      memory_format = self.suggest_memory_format();
    }
  }
  // See Note [Explicit nullopt MemoryFormat argument]
  // TODO: empty_quantized does not work here. It raises an exception in CheckMemoryFormat.h prior to
  // empty_affine_quantizd/_empty_per_channel_affine_quantized calls
  // at::empty also does not work here because there is no proper at::empty support for quantized tensors
  // as it would return a quantized tensor with an UnknownQuantizer
  auto r = self.is_quantized() ? at::empty_like(self, memory_format)
                               : at::empty_symint(self.sym_sizes(),
                                 options.memory_format(memory_format).pinned_memory(pin_out), c10::nullopt);
  r.copy_(self, non_blocking);
  return r;
}

}
}

TORCH_LIBRARY(dha, m) {
  m.def("to_dha(Tensor self) -> Tensor");
  m.def("add_dha(Tensor self) -> ()");
  m.def("is_dha_tensor(Tensor self) -> bool");
}

TORCH_LIBRARY_IMPL(dha, CPU, m) {
  m.impl("to_dha", cuda_direct_host);
  m.impl("add_dha", addTensorDHA);
  m.impl("is_dha_tensor", isDHATensor);
}

TORCH_LIBRARY_IMPL(dha, CUDA, m) {
  m.impl("to_dha", cuda_direct_host);
  m.impl("add_dha", addTensorDHA);
  m.impl("is_dha_tensor", isDHATensor);
}

TORCH_LIBRARY_IMPL(dha, Autocast, m) {
  m.impl("to_dha", cuda_direct_host);
  m.impl("add_dha", addTensorDHA);
  m.impl("is_dha_tensor", isDHATensor);
}