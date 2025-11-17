#ifndef HIPO_AUTO_DETECT_H
#define HIPO_AUTO_DETECT_H

#include <string>

namespace hipo {
// Detect BLAS integer model
enum class BlasIntegerModel { not_set, unknown, lp64, ilp64 };
BlasIntegerModel getBlasIntegerModel();
std::string getBlasIntegerModelString();

// Detect Metis integer type
int getMetisIntegerType();

}  // namespace hipo

#endif