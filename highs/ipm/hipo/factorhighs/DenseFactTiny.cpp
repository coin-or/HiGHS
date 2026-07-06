
using Int = int;

namespace hipo {
Int denseFactTiny(Int n, Int k, double* A, double* B) {
  // ===========================================================================
  // Partial dense factorisation for supernodes with small size and front.
  // Assumes that both n and n-k are smaller than the block size.
  // Matrix A is in format FH
  // Matrix B is in format FH
  // BLAS calls: none
  // ===========================================================================

  if (n < 0 || k < 0 || !A || (k < n && !B)) return 1;
  if (n == 0) return 0;

  const Int ldb = n - k;

  // factorisation of top kxk block
  for (Int row = 0; row < k; ++row) {
    for (Int col = 0; col < row; ++col) {
      // off-diagonal
      for (Int l = 0; l < col; ++l) {
        A[col + row * k] -= A[l + row * k] * A[l + col * k] * A[l + l * k];
      }
      double temp = A[col + row * k];
      A[col + row * k] /= A[col + col * k];

      // contribution to diagonal
      A[row + row * k] -= A[col + row * k] * temp;
    }
  }

  for (Int row = k; row < n; ++row) {
    // update rows below
    for (Int col = 0; col < k; ++col) {
      for (Int l = 0; l < col; ++l) {
        A[col + row * k] -= A[l + row * k] * A[l + col * k] * A[l + l * k];
      }
      A[col + row * k] /= A[col + col * k];
    }

    // update schur complement
    const Int rowB = row - k;
    for (Int col = k; col <= row; ++col) {
      const Int colB = col - k;
      for (Int l = 0; l < k; ++l) {
        B[colB + rowB * ldb] -= A[l + row * k] * A[l + col * k] * A[l + l * k];
      }
    }
  }

  // store the reciprocal of the pivot
  for (Int row = k; row < n; ++row) {
    // A[row + k * row] = 1.0 / A[row + k * row];
  }
}

}  // namespace hipo
