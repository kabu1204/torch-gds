#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/detail/CUDAHooks.h>
#include <ATen/detail/CUDAHooksInterface.h>

#include <cuda_runtime_api.h>
#include <string>

namespace at::cuda {
namespace custom{

struct BlockSize {
  size_t size_{0};
  void* ptr_{nullptr};
};

struct Block {
  size_t size_{0};
  void* ptr_{nullptr};

  std::mutex mutex_;
  bool allocated_{false};
  size_t event_count_{0};
  std::unordered_set<at::cuda::CUDAStream> streams_;
};

class EventPool {
 public:
  using Event = std::unique_ptr<
      at::cuda::CUDAEvent,
      std::function<void(at::cuda::CUDAEvent*)>>;
  EventPool() : pools_(at::cuda::device_count()) {}

  Event get(DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](at::cuda::CUDAEvent* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<at::cuda::CUDAEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(
        std::make_unique<at::cuda::CUDAEvent>(cudaEventDisableTiming).release(),
        destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<at::cuda::CUDAEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

// Used for heterogenous lookup support in the free list.
struct BlockComparator {
  using is_transparent = void;
  bool operator()(const Block* a, const Block* b) const {
    if (a->size_ != b->size_) {
      return a->size_ < b->size_;
    }
    return (uintptr_t)a->ptr_ < (uintptr_t)b->ptr_;
  }

  // Transparent overloads
  bool operator()(const Block* a, BlockSize b) const {
    if (a->size_ != b.size_) {
      return a->size_ < b.size_;
    }
    return (uintptr_t)a->ptr_ < (uintptr_t)b.ptr_;
  }
  bool operator()(BlockSize a, const Block* b) const {
    if (a.size_ != b->size_) {
      return a.size_ < b->size_;
    }
    return (uintptr_t)a.ptr_ < (uintptr_t)b->ptr_;
  }
};

class CUDAHostAllocator {
 public:
  std::pair<void*, void*> allocate(size_t size) {
    if (size == 0) {
      return {nullptr, nullptr};
    }

    process_events();

    // First, try to allocate from the free list
    {
      std::lock_guard<std::mutex> g(free_list_mutex_);
      auto it = free_list_.lower_bound(BlockSize{size, nullptr});
      if (it != free_list_.end()) {
        auto block = *it;
        block->allocated_ = true;
        free_list_.erase(it);
        return {block->ptr_, reinterpret_cast<void*>(block)};
      }
    }
    // Then, create a new block.
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, since we assume unified addressing. So we grab any existing
    // primary context, if available. See pytorch/pytorch#21081.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index =
        c10::cuda::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(
          at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
    }

    // Round up the allocation to the nearest power of two to improve reuse.
    void* ptr = nullptr;
    C10_CUDA_CHECK(cudaHostAlloc(
        &ptr, c10::llvm::PowerOf2Ceil(size), cudaHostAllocDefault));
    auto block = new Block();
    block->size_ = c10::llvm::PowerOf2Ceil(size);
    block->ptr_ = ptr;
    block->allocated_ = true;

    {
      std::lock_guard<std::mutex> g(blocks_mutex_);
      blocks_.insert(block);
      ptr_to_block_.insert({block->ptr_, block});
    }
    return {block->ptr_, reinterpret_cast<void*>(block)};
  }

  void free(void* ctx) {
    if (!ctx) {
      return;
    }

    // Note: we can assume that free is correctly paired with alloc,
    // and thus we do not need to look up the ctx in blocks_.
    auto* block = reinterpret_cast<Block*>(ctx);
    // TORCH_WARN("DHA Free: "+std::to_string(block->size_));

    c10::optional<std::vector<EventPool::Event>> events;
    {
      std::lock_guard<std::mutex> g(block->mutex_);
      block->allocated_ = false;
      if (block->streams_.empty()) {
        TORCH_INTERNAL_ASSERT(block->event_count_ == 0);
      } else {
        events = std::vector<EventPool::Event>();
        events->reserve(block->streams_.size());
        for (auto stream : block->streams_) {
          auto event = event_pool_.get(stream.device_index());
          event->record(stream);
          events->push_back(std::move(event));
        }
        block->event_count_ += events->size();
        block->streams_.clear();
      }
    }

    if (!events) {
      std::lock_guard<std::mutex> g(free_list_mutex_);
      free_list_.insert(block);
    } else {
      std::lock_guard<std::mutex> g(cuda_events_mutex_);
      for (auto&& event : *events) {
        cuda_events_.emplace_front(std::move(event), block);
      }
    }
  }

  bool record_event(void* ptr, void* ctx, at::cuda::CUDAStream stream) {
    auto* block = reinterpret_cast<Block*>(ctx);

    // Note: we need to check if the passed-in `ctx` is valid. This is because
    // `record_event` (via `CachingHostAllocator_recordEvent`) can be invoked on
    // an arbitrary tensor, and is not guaranteed to correspond to a pinned
    // memory allocation. Therefore, we need to check that `ctx` is valid before
    // proceeding.
    {
      std::lock_guard<std::mutex> g(blocks_mutex_);
      if (blocks_.find(block) != blocks_.end()) {
        // Now we know this object is safe to access.
        std::lock_guard<std::mutex> gb(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
      auto it = ptr_to_block_.find(ptr);
      if (it != ptr_to_block_.end()) {
        block = it->second;
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
    }

    return false;
  }

  void empty_cache() {
    // Flush any available blocks into the free_list.
    process_events();

    // Release cached events from the event pool.
    event_pool_.empty_cache();

    // Remove all elements from the free list, remove them from the blocks
    // list, and free the associated pinned memory allocation. This requires
    // concurrently holding both the free list mutex and the blocks mutex, and
    // is the only function that concurrently holds multiple mutexes.
    std::lock(free_list_mutex_, blocks_mutex_);
    std::lock_guard<std::mutex> gf(free_list_mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

    std::vector<Block*> blocks_to_remove(free_list_.begin(), free_list_.end());
    free_list_.clear();
    for (auto* block : blocks_to_remove) {
      blocks_.erase(block);
      ptr_to_block_.erase(block->ptr_);
      AT_CUDA_CHECK(cudaFreeHost(block->ptr_));
      delete block;
    }
  }

 private:
  void process_events() {
    while (true) {
      // Avoid calling cudaEventDestroy while holding a mutex, so move
      // intermediate events out of the lock into this object.
      c10::optional<std::pair<EventPool::Event, Block*>> processed;

      {
        std::lock_guard<std::mutex> g(cuda_events_mutex_);
        if (!cuda_events_.empty()) {
          processed = std::move(cuda_events_.back());
          cuda_events_.pop_back();
        }
      }

      if (!processed) {
        return;
      }

      // otherwise, query the event
      {
        // now, see if we can handle this element
        auto& event = processed->first;
        cudaError_t err = cudaEventQuery(*event);
        if (err == cudaErrorNotReady) {
          (void)cudaGetLastError(); // clear CUDA error
          // push the event onto the back of the queue if it's not
          // ready. TODO: do we need some debouncing logic to avoid allocating
          // threads repeatedly spinning on an event?
          {
            std::lock_guard<std::mutex> g(cuda_events_mutex_);
            cuda_events_.push_back(std::move(*processed));
          }
          return;
        } else if (err != cudaSuccess) {
          C10_CUDA_CHECK(err);
        }
      }

      // Process the events.
      TORCH_INTERNAL_ASSERT(processed);
      auto* block = processed->second;
      bool available = false;
      {
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(!block->allocated_)
        block->event_count_--;
        if (block->event_count_ == 0) {
          available = true;
        }
      }

      if (available) {
        std::lock_guard<std::mutex> g(free_list_mutex_);
        free_list_.insert(block);
      }
    }
  }

  EventPool event_pool_;

  alignas(64) std::mutex blocks_mutex_;
  std::unordered_set<Block*> blocks_;
  std::unordered_map<void*, Block*> ptr_to_block_;
  // Note: sharding this mutex seems to be profitable in heavily multi-threaded
  // scenarios.
  alignas(64) std::mutex free_list_mutex_;
  // Note: an alternative datastructure can yield significant wins here in
  // microbenchmarks.
  std::set<Block*, BlockComparator> free_list_;

  alignas(64) std::mutex cuda_events_mutex_;
  std::deque<std::pair<EventPool::Event, Block*>> cuda_events_;
};

}

static custom::CUDAHostAllocator& getCustomCUDAHostAllocator() {
  // leak and don't worry about shutdown
  static auto* r = new custom::CUDAHostAllocator();
  return *r;
}

static void CustomCUDAHostAllocatorDeleter(void* ctx) {
  getCustomCUDAHostAllocator().free(ctx);
}

bool CustomCachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    at::cuda::CUDAStream stream) {
  return getCustomCUDAHostAllocator().record_event(ptr, ctx, stream);
}

// Releases cached pinned memory allocations via cudaHostFree
void CustomCachingHostAllocator_emptyCache() {
  getCustomCUDAHostAllocator().empty_cache();
}

struct CUDAHostAllocatorCUDA final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    // TORCH_WARN("DHA Alloc: "+std::to_string(size));
    auto ptr_and_ctx = getCustomCUDAHostAllocator().allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &CustomCUDAHostAllocatorDeleter,
        {at::DeviceType::CUDA, 0}};
  }
};

static CUDAHostAllocatorCUDA custom_cuda_host_allocator;

at::Allocator* getCachingHostAllocatorCUDA() {
  return &custom_cuda_host_allocator;
}

}