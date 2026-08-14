cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(FlagGemsArmOperators LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../../cmake")
endif()

if(NOT TARGET Torch::Torch)
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
  find_package(Torch MODULE REQUIRED)
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(NOT TARGET triton_jit)
  if(NOT DEFINED TRITON_JIT_ROOT OR NOT DEFINED TRITON_JIT_BUILD)
    message(FATAL_ERROR
      "TRITON_JIT_ROOT and TRITON_JIT_BUILD are required for a standalone Arm build")
  endif()
  list(APPEND CMAKE_MODULE_PATH "${TRITON_JIT_ROOT}/cmake")
  set(fmt_DIR "${TRITON_JIT_BUILD}/_deps/fmt-build")
  find_package(TritonJIT CONFIG REQUIRED
    PATHS "${TRITON_JIT_BUILD}" NO_DEFAULT_PATH)
endif()

add_library(flag_gems_arm_ops SHARED q4_op.cpp w8_op.cpp gdn_op.cpp)
target_link_libraries(flag_gems_arm_ops PRIVATE
  Torch::Torch triton_jit OpenMP::OpenMP_CXX ffi)
target_compile_definitions(flag_gems_arm_ops PRIVATE
  FLAGGEMS_Q4_KERNEL_SOURCE_DEFAULT=""
  FLAGGEMS_W8_KERNEL_SOURCE_DEFAULT="")
set_target_properties(flag_gems_arm_ops PROPERTIES
  PREFIX "lib"
  BUILD_RPATH "${TRITON_JIT_BUILD}/src")

if(APPLE)
  set_target_properties(flag_gems_arm_ops PROPERTIES
    BUILD_RPATH "@loader_path;${TRITON_JIT_BUILD}/src"
    INSTALL_RPATH "@loader_path")
  if(OpenMP_libomp_LIBRARY)
    add_custom_command(TARGET flag_gems_arm_ops POST_BUILD
      COMMAND "${CMAKE_INSTALL_NAME_TOOL}" -change
        "${OpenMP_libomp_LIBRARY}" "@rpath/libomp.dylib"
        "$<TARGET_FILE:flag_gems_arm_ops>"
      VERBATIM)
  endif()
elseif(UNIX)
  set_target_properties(flag_gems_arm_ops PROPERTIES
    BUILD_RPATH "$ORIGIN;${TRITON_JIT_BUILD}/src"
    INSTALL_RPATH "$ORIGIN")
endif()

if(FLAGGEMS_INSTALL)
  install(TARGETS flag_gems_arm_ops LIBRARY DESTINATION "csrc/arm")
  install(FILES __init__.py kernel_sources.h q4_kernels.py w8_kernels.py README.md
          DESTINATION "csrc/arm")
endif()
