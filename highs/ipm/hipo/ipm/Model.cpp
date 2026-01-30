#include "Model.h"

#include "Parameters.h"
#include "Status.h"
#include "ipm/IpxWrapper.h"
#include "ipm/hipo/auxiliary/Log.h"
#include "model/HighsHessianUtils.h"

namespace hipo {

Int Model::init(const HighsLp& lp, const HighsHessian& Q) {
  fillInIpxData(lp, n_, m_, offset_, c_, lower_, upper_, A_.start_, A_.index_,
                A_.value_, b_, constraints_);
  Q_ = Q;
  if (qp()) completeHessian(n_, Q_);
  sense_ = lp.sense_;

  if (checkData()) return kStatusBadModel;

  lp_orig_ = &lp;
  n_orig_ = n_;
  m_orig_ = m_;
  A_.num_col_ = n_;
  A_.num_row_ = m_;

  preprocess();
  n_preproc_ = n_;
  m_preproc_ = m_;

  scale();
  reformulate();
  denseColumns();
  computeNorms();

  // double transpose to sort indices of each column
  A_.ensureRowwise();
  A_.ensureColwise();

  if (checkData()) return kStatusBadModel;

  ready_ = true;

  return 0;
}

Int Model::checkData() const {
  // Check if model provided by the user is ok.
  // Return kStatusBadModel if something is wrong.

  // Dimensions are valid
  if (n_ <= 0 || m_ < 0) return kStatusBadModel;

  // Vectors are of correct size
  if (c_.size() != n_ || b_.size() != m_ || lower_.size() != n_ ||
      upper_.size() != n_ || constraints_.size() != m_ ||
      A_.start_.size() != n_ + 1 || A_.index_.size() != A_.start_.back() ||
      A_.value_.size() != A_.start_.back())
    return kStatusBadModel;

  // Hessian is ok, for QPs only
  if (qp() && (Q_.dim_ != n_ || Q_.format_ != HessianFormat::kTriangular))
    return kStatusBadModel;

  // Vectors are valid
  for (Int i = 0; i < n_; ++i)
    if (!std::isfinite(c_[i])) return kStatusBadModel;
  for (Int i = 0; i < m_; ++i)
    if (!std::isfinite(b_[i])) return kStatusBadModel;
  for (Int i = 0; i < n_; ++i) {
    if (!std::isfinite(lower_[i]) && lower_[i] != -INFINITY)
      return kStatusBadModel;
    if (!std::isfinite(upper_[i]) && upper_[i] != INFINITY)
      return kStatusBadModel;
    if (lower_[i] > upper_[i]) return kStatusBadModel;
  }
  for (Int i = 0; i < m_; ++i)
    if (constraints_[i] != '<' && constraints_[i] != '=' &&
        constraints_[i] != '>')
      return kStatusBadModel;

  // Matrix is valid
  for (Int i = 0; i < A_.start_[n_]; ++i)
    if (!std::isfinite(A_.value_[i])) return kStatusBadModel;

  return 0;
}

void Model::preprocess() {
  // Perform some basic preprocessing, in case the problem is run without
  // presolve

  // ==========================================
  // Remove empty rows
  // ==========================================

  // find empty rows
  std::vector<Int> entries_per_row(m_, 0);
  for (Int col = 0; col < n_; ++col) {
    for (Int el = A_.start_[col]; el < A_.start_[col + 1]; ++el) {
      const Int row = A_.index_[el];
      ++entries_per_row[row];
    }
  }

  empty_rows_ = 0;
  for (Int i : entries_per_row)
    if (i == 0) ++empty_rows_;

  if (empty_rows_ > 0) {
    rows_shift_.assign(m_, 0);
    for (Int i = 0; i < m_; ++i) {
      if (entries_per_row[i] == 0) {
        // count how many empty rows there are before a given row
        for (Int j = i + 1; j < m_; ++j) ++rows_shift_[j];
        rows_shift_[i] = -1;
      }
    }

    // shift each row index by the number of empty rows before it
    for (Int col = 0; col < n_; ++col) {
      for (Int el = A_.start_[col]; el < A_.start_[col + 1]; ++el) {
        const Int row = A_.index_[el];
        A_.index_[el] -= rows_shift_[row];
      }
    }
    A_.num_row_ -= empty_rows_;

    // shift entries in b and constraints
    for (Int i = 0; i < m_; ++i) {
      // ignore entries to be removed
      if (rows_shift_[i] == -1) continue;

      Int shifted_pos = i - rows_shift_[i];
      b_[shifted_pos] = b_[i];
      constraints_[shifted_pos] = constraints_[i];
    }
    b_.resize(A_.num_row_);
    constraints_.resize(A_.num_row_);

    m_ = A_.num_row_;
  }

  // ==========================================
  // Remove fixed variables
  // ==========================================
  // See "Preprocessing for quadratic programming", Gould, Toint, Math Program
  fixed_vars_ = 0;
  for (Int i = 0; i < n_; ++i)
    if (lower_[i] == upper_[i]) ++fixed_vars_;

  if (fixed_vars_ > 0) {
    fixed_at_.assign(n_, kHighsInf);
    std::vector<Int> index_to_remove{};
    for (Int j = 0; j < n_; ++j) {
      if (lower_[j] == upper_[j]) {
        fixed_at_[j] = lower_[j];
        index_to_remove.push_back(j);
        const double xcol = fixed_at_[j];

        offset_ += c_[j] * xcol + 0.5 * Q_.diag(j) * xcol * xcol;

        for (Int el = A_.start_[j]; el < A_.start_[j + 1]; ++el) {
          const Int row = A_.index_[el];
          const double val = A_.value_[el];
          b_[row] -= val * xcol;
        }

        for (Int colQ = 0; colQ < j; ++colQ) {
          for (Int el = Q_.start_[colQ]; el < Q_.start_[colQ + 1]; ++el) {
            const Int rowQ = Q_.index_[el];
            if (rowQ == j) {
              c_[colQ] += Q_.value_[el] * xcol;
            }
          }
        }
        for (Int el = Q_.start_[j]; el < Q_.start_[j + 1]; ++el) {
          const Int rowQ = Q_.index_[el];
          c_[rowQ] += Q_.value_[el] * xcol;
        }
      }
    }

    HighsIndexCollection index_collection;
    create(index_collection, index_to_remove.size(), index_to_remove.data(),
           n_);
    A_.deleteCols(index_collection);
    Q_.deleteCols(index_collection);

    Int next = 0;
    Int copy_to = 0;
    for (Int i = 0; i < n_; ++i) {
      if (next < index_to_remove.size() && i == index_to_remove[next]) {
        ++next;
        continue;
      } else {
        c_[copy_to] = c_[i];
        lower_[copy_to] = lower_[i];
        upper_[copy_to] = upper_[i];
        copy_to++;
      }
    }

    n_ -= fixed_vars_;
    assert(A_.num_col_ == n_);
    assert(Q_.dim_ == n_);
    c_.resize(n_);
    lower_.resize(n_);
    upper_.resize(n_);
  }
}

void Model::postprocess(std::vector<double>& x, std::vector<double>& xl,
                        std::vector<double>& xu, std::vector<double>& slack,
                        std::vector<double>& y, std::vector<double>& zl,
                        std::vector<double>& zu) const {
  if (fixed_vars_ > 0) {
    // Add primal and dual variables for fixed variables

    std::vector<double> new_x(fixed_at_.size(), 0.0);
    std::vector<double> new_xl(fixed_at_.size(), 0.0);
    std::vector<double> new_xu(fixed_at_.size(), 0.0);
    std::vector<double> new_zl(fixed_at_.size(), 0.0);
    std::vector<double> new_zu(fixed_at_.size(), 0.0);

    // compute c-A^T*y+Q*x
    std::vector<double> temp = c_;




    Int pos{};
    for (Int i = 0; i < fixed_at_.size(); ++i) {
      if (std::isfinite(fixed_at_[i])) {
        new_x[i] = fixed_at_[i];
        new_xl[i] = 0.0;
        new_xu[i] = 0.0;
        new_zl[i] = kHighsInf;
        new_zu[i] = kHighsInf;

      } else {
        new_x[i] = x[pos];
        new_xl[i] = xl[pos];
        new_xu[i] = xu[pos];
        new_zl[i] = zl[pos];
        new_zu[i] = zu[pos];
        ++pos;
      }
    }

    x = std::move(new_x);
    xl = std::move(new_xl);
    xu = std::move(new_xu);
    zl = std::move(new_zl);
    zu = std::move(new_zu);
  }

  if (empty_rows_ > 0) {
    // Add Lagrange multiplier for empty rows that were removed
    // Add slack for constraints that were removed

    std::vector<double> new_y(rows_shift_.size(), 0.0);
    std::vector<double> new_slack(rows_shift_.size(), 0.0);

    // position to read from y and slack
    Int pos = 0;

    for (Int i = 0; i < rows_shift_.size(); ++i) {
      // ignore shift of empty rows, they will receive a value of 0
      if (rows_shift_[i] == -1) continue;

      // re-align value of y and slack, considering empty rows
      new_y[pos + rows_shift_[i]] = y[pos];
      new_slack[pos + rows_shift_[i]] = slack[pos];
      ++pos;
    }

    y = std::move(new_y);
    slack = std::move(new_slack);
  }
}

void Model::reformulate() {
  // put the model into correct formulation

  Int Annz = A_.numNz();

  for (Int i = 0; i < m_; ++i) {
    if (constraints_[i] != '=') {
      // inequality constraint, add slack variable

      ++n_;

      // lower/upper bound for new slack
      if (constraints_[i] == '>') {
        lower_.push_back(-kHighsInf);
        upper_.push_back(0.0);
      } else {
        lower_.push_back(0.0);
        upper_.push_back(kHighsInf);
      }

      // cost for new slack
      c_.push_back(0.0);

      // add column of identity to A_
      std::vector<Int> temp_ind{i};
      std::vector<double> temp_val{1.0};
      A_.addVec(1, temp_ind.data(), temp_val.data());

      // set scaling to 1
      if (scaled()) colscale_.push_back(1.0);
    }
  }

  if (qp()) completeHessian(n_, Q_);
}

void Model::computeNorms() {
  norm_scaled_obj_ = infNorm(c_);

  norm_unscaled_obj_ = 0.0;
  for (Int i = 0; i < n_; ++i) {
    double val = std::abs(c_[i]);
    if (scaled()) val /= colscale_[i];
    norm_unscaled_obj_ = std::max(norm_unscaled_obj_, val);
  }

  norm_scaled_rhs_ = infNorm(b_);
  for (double d : lower_)
    if (std::isfinite(d))
      norm_scaled_rhs_ = std::max(norm_scaled_rhs_, std::abs(d));
  for (double d : upper_)
    if (std::isfinite(d))
      norm_scaled_rhs_ = std::max(norm_scaled_rhs_, std::abs(d));

  norm_unscaled_rhs_ = 0.0;
  for (Int i = 0; i < m_; ++i) {
    double val = std::abs(b_[i]);
    if (scaled()) val /= rowscale_[i];
    norm_unscaled_rhs_ = std::max(norm_unscaled_rhs_, val);
  }
  for (Int i = 0; i < n_; ++i) {
    if (std::isfinite(lower_[i])) {
      double val = std::abs(lower_[i]);
      if (scaled()) val *= colscale_[i];
      norm_unscaled_rhs_ = std::max(norm_unscaled_rhs_, val);
    }
    if (std::isfinite(upper_[i])) {
      double val = std::abs(upper_[i]);
      if (scaled()) val *= colscale_[i];
      norm_unscaled_rhs_ = std::max(norm_unscaled_rhs_, val);
    }
  }

  // norms of rows and cols of A
  one_norm_cols_.resize(n_);
  one_norm_rows_.resize(m_);
  inf_norm_cols_.resize(n_);
  inf_norm_rows_.resize(m_);
  for (Int col = 0; col < n_; ++col) {
    for (Int el = A_.start_[col]; el < A_.start_[col + 1]; ++el) {
      Int row = A_.index_[el];
      double val = A_.value_[el];
      one_norm_cols_[col] += std::abs(val);
      one_norm_rows_[row] += std::abs(val);
      inf_norm_rows_[row] = std::max(inf_norm_rows_[row], std::abs(val));
      inf_norm_cols_[col] = std::max(inf_norm_cols_[col], std::abs(val));
    }
  }
}

void Model::print(const LogHighs& log) const {
  std::stringstream log_stream;

  log_stream << textline("Rows:") << sci(m_, 0, 1) << '\n';
  log_stream << textline("Cols:") << sci(n_, 0, 1) << '\n';
  log_stream << textline("Nnz A:") << sci(A_.numNz(), 0, 1) << '\n';
  if (num_dense_cols_ > 0)
    log_stream << textline("Dense cols:") << integer(num_dense_cols_, 0)
               << '\n';
  if (empty_rows_ > 0)
    log_stream << "Removed " << empty_rows_ << " empty rows\n";
  if (fixed_vars_ > 0)
    log_stream << "Removed " << fixed_vars_ << " fixed variables\n";
  if (qp()) {
    log_stream << textline("Nnz Q:") << sci(Q_.numNz(), 0, 1);
    if (nonSeparableQp())
      log_stream << ", non-separable\n";
    else
      log_stream << ", separable\n";
  }

  // compute max and min entry of A in absolute value
  double Amin = kHighsInf;
  double Amax = 0.0;
  for (double val : A_.value_) {
    if (val != 0.0) {
      Amin = std::min(Amin, std::abs(val));
      Amax = std::max(Amax, std::abs(val));
    }
  }
  if (std::isinf(Amin)) Amin = 0.0;

  // compute max and min entry of c
  double cmin = kHighsInf;
  double cmax = 0.0;
  for (Int i = 0; i < n_; ++i) {
    if (c_[i] != 0.0) {
      cmin = std::min(cmin, std::abs(c_[i]));
      cmax = std::max(cmax, std::abs(c_[i]));
    }
  }
  if (std::isinf(cmin)) cmin = 0.0;

  // compute max and min entry of b
  double bmin = kHighsInf;
  double bmax = 0.0;
  for (Int i = 0; i < m_; ++i) {
    if (b_[i] != 0.0) {
      bmin = std::min(bmin, std::abs(b_[i]));
      bmax = std::max(bmax, std::abs(b_[i]));
    }
  }
  if (std::isinf(bmin)) bmin = 0.0;

  // compute max and min entry of Q in absolute value
  double Qmin = kHighsInf;
  double Qmax = 0.0;
  for (double val : Q_.value_) {
    if (val != 0.0) {
      Qmin = std::min(Qmin, std::abs(val));
      Qmax = std::max(Qmax, std::abs(val));
    }
  }
  if (std::isinf(Qmin)) Qmin = 0.0;

  // compute max and min for bounds
  double boundmin = kHighsInf;
  double boundmax = 0.0;
  for (Int i = 0; i < n_; ++i) {
    if (lower_[i] != 0.0 && std::isfinite(lower_[i])) {
      boundmin = std::min(boundmin, std::abs(lower_[i]));
      boundmax = std::max(boundmax, std::abs(lower_[i]));
    }
    if (upper_[i] != 0.0 && std::isfinite(upper_[i])) {
      boundmin = std::min(boundmin, std::abs(upper_[i]));
      boundmax = std::max(boundmax, std::abs(upper_[i]));
    }
  }
  if (std::isinf(boundmin)) boundmin = 0.0;

  // compute max and min scaling
  double scalemin = kHighsInf;
  double scalemax = 0.0;
  if (scaled()) {
    for (Int i = 0; i < n_; ++i) {
      scalemin = std::min(scalemin, colscale_[i]);
      scalemax = std::max(scalemax, colscale_[i]);
    }
    for (Int i = 0; i < m_; ++i) {
      scalemin = std::min(scalemin, rowscale_[i]);
      scalemax = std::max(scalemax, rowscale_[i]);
    }
  }
  if (std::isinf(scalemin)) scalemin = 0.0;

  // print ranges
  log_stream << textline("Range of A:") << "[" << sci(Amin, 5, 1) << ", "
             << sci(Amax, 5, 1) << "], ratio ";
  if (Amin != 0.0)
    log_stream << sci(Amax / Amin, 0, 1) << '\n';
  else
    log_stream << "-\n";

  log_stream << textline("Range of b:") << "[" << sci(bmin, 5, 1) << ", "
             << sci(bmax, 5, 1) << "], ratio ";
  if (bmin != 0.0)
    log_stream << sci(bmax / bmin, 0, 1) << '\n';
  else
    log_stream << "-\n";

  log_stream << textline("Range of c:") << "[" << sci(cmin, 5, 1) << ", "
             << sci(cmax, 5, 1) << "], ratio ";
  if (cmin != 0.0)
    log_stream << sci(cmax / cmin, 0, 1) << '\n';
  else
    log_stream << "-\n";

  if (qp()) {
    log_stream << textline("Range of Q:") << "[" << sci(Qmin, 5, 1) << ", "
               << sci(Qmax, 5, 1) << "], ratio ";
    if (Amin != 0.0)
      log_stream << sci(Qmax / Qmin, 0, 1) << '\n';
    else
      log_stream << "-\n";
  }

  log_stream << textline("Range of bounds:") << "[" << sci(boundmin, 5, 1)
             << ", " << sci(boundmax, 5, 1) << "], ratio ";
  if (boundmin != 0.0)
    log_stream << sci(boundmax / boundmin, 0, 1) << '\n';
  else
    log_stream << "-\n";

  log_stream << textline("Scaling coefficients:") << "[" << sci(scalemin, 5, 1)
             << ", " << sci(scalemax, 5, 1) << "], ratio ";
  if (scalemin != 0.0)
    log_stream << sci(scalemax / scalemin, 0, 1) << '\n';
  else
    log_stream << "-\n";

  if (log.debug(1)) {
    log_stream << textline("Scaling CG iterations:")
               << integer(CG_iter_scaling_) << '\n';
    log_stream << textline("Norm b unscaled") << sci(norm_unscaled_rhs_, 0, 1)
               << '\n';
    log_stream << textline("Norm b scaled") << sci(norm_scaled_rhs_, 0, 1)
               << '\n';
    log_stream << textline("Norm c unscaled") << sci(norm_unscaled_obj_, 0, 1)
               << '\n';
    log_stream << textline("Norm c scaled") << sci(norm_scaled_obj_, 0, 1)
               << '\n';
  }

  log.print(log_stream);
}

void Model::scale() {
  // Apply Curtis-Reid scaling and scale the problem accordingly

  // check if scaling is needed
  bool need_scaling = false;
  for (Int col = 0; col < n_; ++col) {
    for (Int el = A_.start_[col]; el < A_.start_[col + 1]; ++el) {
      if (std::abs(A_.value_[el]) != 1.0) {
        need_scaling = true;
        break;
      }
    }
  }

  if (!need_scaling) return;

  // *********************************************************************
  // Compute scaling
  // *********************************************************************
  // Transformation:
  // A -> R * A * C
  // b -> R * b
  // c -> C * c
  // x -> C^-1 * x
  // y -> R^-1 * y
  // z -> C * z
  // Q -> C * Q * C
  // where R is row scaling, C is col scaling.

  // Compute exponents for CR scaling of matrix A
  std::vector<Int> colexp(n_);
  std::vector<Int> rowexp(m_);
  CG_iter_scaling_ =
      CurtisReidScaling(A_.start_, A_.index_, A_.value_, rowexp, colexp);

  // Compute scaling from exponents
  colscale_.resize(n_);
  rowscale_.resize(m_);
  for (Int i = 0; i < n_; ++i) colscale_[i] = std::ldexp(1.0, colexp[i]);
  for (Int i = 0; i < m_; ++i) rowscale_[i] = std::ldexp(1.0, rowexp[i]);

  bool scaling_failed = isInfVector(colscale_) || isNanVector(colscale_) ||
                        isInfVector(rowscale_) || isNanVector(rowscale_);
  if (scaling_failed) {
    colscale_.clear();
    rowscale_.clear();
    return;
  }

  // *********************************************************************
  // Apply scaling
  // *********************************************************************

  // Column has been scaled up by colscale_[col], so cost is scaled up and
  // bounds are scaled down
  for (Int col = 0; col < n_; ++col) {
    c_[col] *= colscale_[col];
    lower_[col] /= colscale_[col];
    upper_[col] /= colscale_[col];
  }

  // Row has been scaled up by rowscale_[row], so b is scaled up
  for (Int row = 0; row < m_; ++row) b_[row] *= rowscale_[row];

  // Each entry of the matrix is scaled by the corresponding row and col
  // factor
  for (Int col = 0; col < n_; ++col) {
    for (Int el = A_.start_[col]; el < A_.start_[col + 1]; ++el) {
      Int row = A_.index_[el];
      A_.value_[el] *= rowscale_[row];
      A_.value_[el] *= colscale_[col];
    }
  }

  for (Int col = 0; col < Q_.dim_; ++col) {
    for (Int el = Q_.start_[col]; el < Q_.start_[col + 1]; ++el) {
      Int row = Q_.index_[el];
      Q_.value_[el] *= colscale_[row];
      Q_.value_[el] *= colscale_[col];
    }
  }
}

void Model::unscale(std::vector<double>& x, std::vector<double>& xl,
                    std::vector<double>& xu, std::vector<double>& slack,
                    std::vector<double>& y, std::vector<double>& zl,
                    std::vector<double>& zu) const {
  // Undo the scaling with internal format

  if (scaled()) {
    for (Int i = 0; i < n_preproc_; ++i) {
      x[i] *= colscale_[i];
      xl[i] *= colscale_[i];
      xu[i] *= colscale_[i];
      zl[i] /= colscale_[i];
      zu[i] /= colscale_[i];
    }
    for (Int i = 0; i < m_preproc_; ++i) {
      y[i] *= rowscale_[i];
      slack[i] /= rowscale_[i];
    }
  }

  // set variables that were ignored
  for (Int i = 0; i < n_preproc_; ++i) {
    if (!hasLb(i)) {
      xl[i] = kHighsInf;
      zl[i] = 0.0;
    }
    if (!hasUb(i)) {
      xu[i] = kHighsInf;
      zu[i] = 0.0;
    }
  }
}

void Model::unscale(std::vector<double>& x, std::vector<double>& slack,
                    std::vector<double>& y, std::vector<double>& z) const {
  // Undo the scaling with format for crossover

  if (scaled()) {
    for (Int i = 0; i < n_preproc_; ++i) {
      x[i] *= colscale_[i];
      z[i] /= colscale_[i];
    }
    for (Int i = 0; i < m_preproc_; ++i) {
      y[i] *= rowscale_[i];
      slack[i] /= rowscale_[i];
    }
  }
}

void Model::denseColumns() {
  // Compute the maximum density of any column of A and count the number of
  // dense columns.

  max_col_density_ = 0.0;
  num_dense_cols_ = 0;
  for (Int col = 0; col < n_; ++col) {
    Int col_nz = A_.start_[col + 1] - A_.start_[col];
    double col_density = (double)col_nz / m_;
    max_col_density_ = std::max(max_col_density_, col_density);
    if (A_.num_row_ > kMinRowsForDensity && col_density > kDenseColThresh)
      ++num_dense_cols_;
  }
}

Int Model::loadIntoIpx(ipx::LpSolver& lps) const {
  Int ipx_m, ipx_n;
  std::vector<double> ipx_b, ipx_c, ipx_lower, ipx_upper, ipx_A_vals;
  std::vector<Int> ipx_A_ptr, ipx_A_rows;
  std::vector<char> ipx_constraints;
  double ipx_offset;

  if (!lp_orig_) return kStatusError;

  fillInIpxData(*lp_orig_, ipx_n, ipx_m, ipx_offset, ipx_c, ipx_lower,
                ipx_upper, ipx_A_ptr, ipx_A_rows, ipx_A_vals, ipx_b,
                ipx_constraints);

  Int load_status = lps.LoadModel(
      ipx_n, ipx_offset, ipx_c.data(), ipx_lower.data(), ipx_upper.data(),
      ipx_m, ipx_A_ptr.data(), ipx_A_rows.data(), ipx_A_vals.data(),
      ipx_b.data(), ipx_constraints.data());

  return load_status;
}

void Model::multAWithoutSlack(double alpha, const std::vector<double>& x,
                              std::vector<double>& y, bool trans) const {
  assert(x.size() == (trans ? m_preproc_ : n_preproc_));
  assert(y.size() == (trans ? n_preproc_ : m_preproc_));

  if (trans) {
    for (Int col = 0; col < n_preproc_; ++col) {
      for (Int el = A_.start_[col]; el < A_.start_[col + 1]; ++el) {
        y[col] += alpha * A_.value_[el] * x[A_.index_[el]];
      }
    }
  } else {
    for (Int col = 0; col < n_preproc_; ++col) {
      for (Int el = A_.start_[col]; el < A_.start_[col + 1]; ++el) {
        y[A_.index_[el]] += alpha * A_.value_[el] * x[col];
      }
    }
  }
}

void Model::multQWithoutSlack(double alpha, const std::vector<double>& x,
                              std::vector<double>& y) const {
  assert(x.size() == n_preproc_);
  assert(Q_.format_ == HessianFormat::kTriangular);

  for (Int col = 0; col < n_preproc_; ++col) {
    for (Int el = Q_.start_[col]; el < Q_.start_[col + 1]; ++el) {
      const Int row = Q_.index_[el];
      if (row >= n_preproc_) continue;
      y[row] += alpha * Q_.value_[el] * x[col];
      if (row != col) y[col] += alpha * Q_.value_[el] * x[row];
    }
  }
}

void Model::printDense() const {
  std::vector<std::vector<double>> Adense(m_, std::vector<double>(n_, 0.0));
  std::vector<std::vector<double>> Qdense(n_, std::vector<double>(n_, 0.0));

  for (Int col = 0; col < n_; ++col) {
    for (Int el = A_.start_[col]; el < A_.start_[col + 1]; ++el) {
      const Int row = A_.index_[el];
      const double val = A_.value_[el];
      Adense[row][col] = val;
    }
  }
  for (Int col = 0; col < n_; ++col) {
    for (Int el = Q_.start_[col]; el < Q_.start_[col + 1]; ++el) {
      const Int row = Q_.index_[el];
      const double val = Q_.value_[el];
      Qdense[row][col] = val;
    }
  }

  printf("\nA\n");
  for (Int i = 0; i < m_; ++i) {
    for (Int j = 0; j < n_; ++j) printf("%6.2f ", Adense[i][j]);
    printf("\n");
  }
  printf("b\n");
  for (Int i = 0; i < m_; ++i) printf("%6.2f ", b_[i]);
  printf("\n");
  printf("c\n");
  for (Int i = 0; i < n_; ++i) printf("%6.2f ", c_[i]);
  printf("\n");
  printf("lb\n");
  for (Int i = 0; i < n_; ++i) printf("%6.2f ", lower_[i]);
  printf("\n");
  printf("ub\n");
  for (Int i = 0; i < n_; ++i) printf("%6.2f ", upper_[i]);
  printf("\n");
  printf("Q\n");
  for (Int i = 0; i < n_; ++i) {
    for (Int j = 0; j < n_; ++j) printf("%6.2f ", Qdense[i][j]);
    printf("\n");
  }
  printf("offset %6.2f\n", offset_);
}

}  // namespace hipo