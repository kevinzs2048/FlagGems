// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <atomic>
#include <cstdint>
#include <exception>
#include <new>
#include <type_traits>
#include <utility>

namespace flag_gems::arm {
// CPU Triton ABI: one context pointer followed by program IDs and grid sizes.
using RawKernel = void (*)(void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
extern std::atomic<RawKernel> raw_kernel;
extern std::atomic<RawKernel> dense_raw_kernels[3];
struct RawInvocation {
  void (*invoke)(RawInvocation*) noexcept;
  std::exception_ptr error;
};

// Construct the return value in its final descriptor slot (C++17 copy elision).
// The value is destroyed only if construction completed, including error paths.
template <class Result>
class RawResult {
  union Storage {
    char empty;
    Result value;
    Storage() : empty {} {
    }
    ~Storage() {
    }
  } storage_;
  bool initialized_ = false;

 public:
  RawResult() = default;
  RawResult(const RawResult&) = delete;
  RawResult& operator=(const RawResult&) = delete;
  ~RawResult() {
    if (initialized_) storage_.value.~Result();
  }
  template <class F>
  void construct(F&& factory) {
    ::new (static_cast<void*>(&storage_.value)) Result(std::forward<F>(factory)());
    initialized_ = true;
  }
  Result take() {
    return std::move(storage_.value);
  }
};

// The descriptor and captured tensor references live on the caller's stack.
// One synchronous entry surrounds the complete operation, never each expert.
// No allocation, lock, shared_ptr, or JIT-cache lookup occurs on this path.
template <class F>
auto invoke_raw(F&& operation) -> decltype(operation()) {
  using Callable = std::remove_reference_t<F>;
  using Result = decltype(operation());
  auto kernel = raw_kernel.load(std::memory_order_acquire);
  TORCH_CHECK(kernel, "Initialize the CPU TLE Raw kernel before calling .raw");
  if constexpr (std::is_void_v<Result>) {
    struct Payload : RawInvocation {
      Callable operation;
    };
    Payload context {
        {[](RawInvocation* base) noexcept -> void {
           auto& data = *static_cast<Payload*>(base);
           try {
             data.operation();
           } catch (...) {
             data.error = std::current_exception();
           }
         }, {}},
        std::forward<F>(operation)
    };
    kernel(&context, 0, 0, 0, 1, 1, 1);
    if (context.error) std::rethrow_exception(context.error);
  } else {
    struct alignas(64) Payload : RawInvocation {
      Callable operation;
      RawResult<Result> result;
    };
    Payload context {
        {[](RawInvocation* base) noexcept -> void {
           auto& data = *static_cast<Payload*>(base);
           try {
             data.result.construct(data.operation);
           } catch (...) {
             data.error = std::current_exception();
           }
         }, {}},
        std::forward<F>(operation),
        {}
    };
    kernel(&context, 0, 0, 0, 1, 1, 1);
    if (context.error) std::rethrow_exception(context.error);
    return context.result.take();
  }
}

// Dense computation has no Tensor return value: preparation and ownership
// stay in the common host wrapper, and one Raw call runs the parallel plan.
template <class Plan>
void invoke_dense_compute(Plan& plan, unsigned index) {
  auto kernel = dense_raw_kernels[index].load(std::memory_order_acquire);
  TORCH_CHECK(kernel, "Initialize the dense CPU TLE Raw entries first");
  kernel(&plan, 0, 0, 0, 1, 1, 1);
}
}  // namespace flag_gems::arm
