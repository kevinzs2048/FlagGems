# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
  message(FATAL_ERROR "Arm native packing requires AArch64 NEON")
endif()
find_package(Python COMPONENTS Interpreter REQUIRED)
find_package(Torch REQUIRED)
add_library(flag_gems_arm_silu_native SHARED
  silu_mul_a8_pack.cpp silu_mul_a8_pack_op.cpp)
target_link_libraries(flag_gems_arm_silu_native PRIVATE Torch::Torch)
target_compile_definitions(flag_gems_arm_silu_native PRIVATE FLAGGEMS_ARM_SILU_OPERATOR_LIBRARY=1)
if(UNIX AND NOT APPLE)
  # Keep native and Raw arithmetic in this DSO when both Arm libraries load.
  target_link_options(flag_gems_arm_silu_native PRIVATE "LINKER:-Bsymbolic-functions")
endif()
# Preserve the reference FP32 schedule; required FMA sites use explicit NEON.
target_compile_options(flag_gems_arm_silu_native PRIVATE -O3 -ffp-contract=off)

# Use the LLVM toolchain matching FlagTree CPU, not necessarily Torch's C++
# compiler. Native-only builds do not require a bitcode compiler.
set(FLAGGEMS_ARM_BITCODE_COMPILER "" CACHE FILEPATH
  "clang++ from the LLVM toolchain used to build FlagTree CPU")
set(FLAGGEMS_ARM_BITCODE_CPU "generic" CACHE STRING
  "LLVM CPU target for SiLU bitcode (native restricts portability)")
set(FLAGGEMS_ARM_SILU_BITCODE_MODE "kernel" CACHE STRING
  "SiLU bitcode contents: kernel (inline arithmetic) or boundary (native leaf)")
set_property(CACHE FLAGGEMS_ARM_SILU_BITCODE_MODE PROPERTY STRINGS kernel boundary)
if(FLAGGEMS_ARM_BITCODE_COMPILER)
  if(FLAGGEMS_ARM_SILU_BITCODE_MODE STREQUAL "kernel")
    set(_silu_bitcode_source silu_mul_a8_pack.cpp)
  elseif(FLAGGEMS_ARM_SILU_BITCODE_MODE STREQUAL "boundary")
    set(_silu_bitcode_source silu_mul_a8_pack_boundary.cpp)
  else()
    message(FATAL_ERROR "FLAGGEMS_ARM_SILU_BITCODE_MODE must be kernel or boundary")
  endif()
  set(_silu_bitcode "${CMAKE_CURRENT_BINARY_DIR}/flag_gems_arm_silu.bc")
  set(_silu_sysroot_flags)
  if(APPLE AND CMAKE_OSX_SYSROOT)
    list(APPEND _silu_sysroot_flags -isysroot "${CMAKE_OSX_SYSROOT}")
  endif()
  if(APPLE AND CMAKE_OSX_DEPLOYMENT_TARGET)
    list(APPEND _silu_sysroot_flags
      "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  add_custom_command(OUTPUT "${_silu_bitcode}"
    COMMAND "${FLAGGEMS_ARM_BITCODE_COMPILER}" -O3 -std=c++17 -fPIC
      -ffp-contract=off -DFLAGGEMS_ARM_SILU_OPERATOR_LIBRARY=1
      -DFLAGGEMS_ARM_SILU_FUSE_PADDING=1
      "-mcpu=${FLAGGEMS_ARM_BITCODE_CPU}" ${_silu_sysroot_flags}
      -emit-llvm -c "${CMAKE_CURRENT_SOURCE_DIR}/${_silu_bitcode_source}"
      -o "${_silu_bitcode}"
    DEPENDS ${_silu_bitcode_source} silu_mul_a8_pack.h VERBATIM)
  add_custom_target(flag_gems_arm_silu_bitcode ALL DEPENDS "${_silu_bitcode}")
  if(FLAGGEMS_INSTALL)
    install(FILES "${_silu_bitcode}"
      DESTINATION flag_gems/runtime/backend/_arm/lib)
  endif()
endif()

option(FLAGGEMS_BUILD_ARM_KAI "Build Arm quantized GEMM and routed MoE" OFF)
if(FLAGGEMS_BUILD_ARM_KAI)
  set(KLEIDIAI_ROOT "" CACHE PATH "KleidiAI 1.29 source/include directory")
  set(KLEIDIAI_LIBRARY "" CACHE FILEPATH "KleidiAI static library")
  if(NOT EXISTS "${KLEIDIAI_ROOT}/kai/kai_common.h" OR
     NOT EXISTS "${KLEIDIAI_LIBRARY}")
    message(FATAL_ERROR "Arm KAI operators require KLEIDIAI_ROOT and KLEIDIAI_LIBRARY")
  endif()
  find_package(OpenMP REQUIRED)
  # Torch ships its own OpenMP runtime. Linking another pool can invalidate
  # the thread policy and nested-parallel guards used by these operators.
  get_target_property(_torch_includes Torch::Torch INTERFACE_INCLUDE_DIRECTORIES)
  list(GET _torch_includes 0 _torch_include)
  get_filename_component(_torch_directory "${_torch_include}" DIRECTORY)
  file(GLOB _torch_omp
    "${_torch_directory}/lib/libomp.dylib"
    "${_torch_directory}/lib/libgomp*.so*")
  list(LENGTH _torch_omp _torch_omp_count)
  if(NOT _torch_omp_count EQUAL 1)
    message(FATAL_ERROR "Expected one Torch OpenMP runtime, found: ${_torch_omp}")
  endif()
  separate_arguments(_omp_compile_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
  add_library(flag_gems_arm_native SHARED
    raw_dispatch.cpp g32.cpp w8a8.cpp moe_prepack.cpp moe_streamed.cpp moe_prefill.cpp
    gdn.cpp conv_prefill.cpp silu_mul_a8_pack.cpp)
  target_include_directories(flag_gems_arm_native PRIVATE "${KLEIDIAI_ROOT}")
  target_link_libraries(flag_gems_arm_native PRIVATE
    Torch::Torch "${_torch_omp}" "${KLEIDIAI_LIBRARY}" ${CMAKE_DL_LIBS})
  target_compile_options(flag_gems_arm_native PRIVATE ${_omp_compile_flags})
  target_compile_options(flag_gems_arm_native PRIVATE -O3 -ffp-contract=off)
  target_compile_features(flag_gems_arm_native PRIVATE cxx_std_17)
  # Bind the dense KAI implementation locally. Routed expert operations
  # explicitly resolve Torch's packed-weight ABI in runtime_kai.h.
  if(UNIX AND NOT APPLE)
    target_link_options(flag_gems_arm_native PRIVATE "LINKER:-Bsymbolic-functions")
  endif()
  if(FLAGGEMS_INSTALL)
    install(TARGETS flag_gems_arm_native
      LIBRARY DESTINATION flag_gems/runtime/backend/_arm/lib)
  endif()
endif()
if(FLAGGEMS_INSTALL)
  install(TARGETS flag_gems_arm_silu_native
    LIBRARY DESTINATION flag_gems/runtime/backend/_arm/lib)
  install(FILES LICENSE.sleef.txt
    DESTINATION flag_gems/runtime/backend/_arm/lib)
endif()
