#pragma once

#include <cstdlib>

#ifndef FLAGGEMS_Q4_KERNEL_SOURCE_DEFAULT
#define FLAGGEMS_Q4_KERNEL_SOURCE_DEFAULT ""
#endif

#ifndef FLAGGEMS_W8_KERNEL_SOURCE_DEFAULT
#define FLAGGEMS_W8_KERNEL_SOURCE_DEFAULT ""
#endif

inline const char* flag_gems_kernel_source(const char* variable,
                                           const char* fallback) {
  if (const char* configured = std::getenv(variable)) {
    if (*configured != '\0') {
      return configured;
    }
  }
  TORCH_CHECK(fallback[0] != '\0', variable, " is not configured");
  return fallback;
}

#define Q4_KERNEL_SOURCE                                                   \
  flag_gems_kernel_source("FLAGGEMS_Q4_KERNEL_SOURCE",                   \
                          FLAGGEMS_Q4_KERNEL_SOURCE_DEFAULT)
#define W8_KERNEL_SOURCE                                                   \
  flag_gems_kernel_source("FLAGGEMS_W8_KERNEL_SOURCE",                   \
                          FLAGGEMS_W8_KERNEL_SOURCE_DEFAULT)
