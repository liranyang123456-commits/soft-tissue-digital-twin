#pragma once

// Windows RPC headers define `small` as a macro for `char`.  PyTorch 2.11
// uses `small` as a C++ parameter name in CUDACachingAllocator.h, so native
// CUDA extensions fail unless Windows headers are included once and the macro
// is removed before PyTorch headers are parsed.
#ifdef _WIN32
#include <windows.h>
#ifdef small
#undef small
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

