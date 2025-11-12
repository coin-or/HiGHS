#include "linalg.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#define CHECK_CUSPARSE(expr)