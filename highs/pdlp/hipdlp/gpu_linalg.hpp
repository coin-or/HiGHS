/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file pdlp/hipdlp/gpu_linalg.hpp
 * @brief GPU implementation of the linear algebra interface.
 */
#ifndef PDLP_HIPDLP_GPU_LINALG_HPP
#define PDLP_HIPDLP_GPU_LINALG_HPP
#include "linalg.hpp"
#include "Highs.h"

namespace highs {
namespace pdlp {

std::unique_ptr<LinearAlgebraBackend> createGpuBackend();


} // namespace pdlp
} // namespace highs
#endif  // PDLP_HIPDLP_GPU_LINALG_HPP
