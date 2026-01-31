#include "PrePostProcess.h"

#include "CurtisReidScaling.h"
#include "Iterate.h"
#include "Model.h"
#include "model/HighsHessianUtils.h"

namespace hipo {

void PrePostProcessPoint::assertConsistency(Int n, Int m) const {
  assert(x.size() == n);
  assert(xl.size() == n);
  assert(xu.size() == n);
  assert(slack.size() == m);
  assert(y.size() == m);
  assert(zl.size() == n);
  assert(zu.size() == n);
}

void RemoveEmptyRows::apply(Model& model) {
  Int& n = model.n_;
  Int& m = model.m_;
  HighsSparseMatrix& A = model.A_;
  std::vector<double>& b = model.b_;
  std::vector<char>& constraints = model.constraints_;

  n_pre = n;
  m_pre = m;

  // find empty rows
  std::vector<Int> entries_per_row(m, 0);
  for (Int col = 0; col < n; ++col) {
    for (Int el = A.start_[col]; el < A.start_[col + 1]; ++el) {
      const Int row = A.index_[el];
      ++entries_per_row[row];
    }
  }

  empty_rows = 0;
  for (Int i : entries_per_row)
    if (i == 0) ++empty_rows;

  if (empty_rows > 0) {
    rows_shift.assign(m, 0);
    for (Int i = 0; i < m; ++i) {
      if (entries_per_row[i] == 0) {
        // count how many empty rows there are before a given row
        for (Int j = i + 1; j < m; ++j) ++rows_shift[j];
        rows_shift[i] = -1;
      }
    }

    // shift each row index by the number of empty rows before it
    for (Int col = 0; col < n; ++col) {
      for (Int el = A.start_[col]; el < A.start_[col + 1]; ++el) {
        const Int row = A.index_[el];
        A.index_[el] -= rows_shift[row];
      }
    }
    A.num_row_ -= empty_rows;

    // shift entries in b and constraints
    for (Int i = 0; i < m; ++i) {
      // ignore entries to be removed
      if (rows_shift[i] == -1) continue;

      Int shifted_pos = i - rows_shift[i];
      b[shifted_pos] = b[i];
      constraints[shifted_pos] = constraints[i];
    }
    b.resize(A.num_row_);
    constraints.resize(A.num_row_);

    m = A.num_row_;
  }

  n_post = n;
  m_post = m;
}

void RemoveEmptyRows::undo(PrePostProcessPoint& point) const {
  point.assertConsistency(n_post, m_post);
  if (empty_rows > 0) {
    // Add Lagrange multiplier for empty rows that were removed
    // Add slack for constraints that were removed

    std::vector<double> new_y(m_pre, 0.0);
    std::vector<double> new_slack(m_pre, 0.0);

    // position to read from y and slack
    Int pos = 0;

    for (Int i = 0; i < m_pre; ++i) {
      // ignore shift of empty rows, they will receive a value of 0
      if (rows_shift[i] == -1) continue;

      // re-align value of y and slack, considering empty rows
      new_y[pos + rows_shift[i]] = point.y[pos];
      new_slack[pos + rows_shift[i]] = point.slack[pos];
      ++pos;
    }

    point.y = std::move(new_y);
    point.slack = std::move(new_slack);
  }
  point.assertConsistency(n_pre, m_pre);
}

void RemoveFixedVars::apply(Model& model) {
  Int& n = model.n_;
  Int& m = model.m_;
  HighsSparseMatrix& A = model.A_;
  std::vector<double>& b = model.b_;
  std::vector<double>& c = model.c_;
  std::vector<double>& lower = model.lower_;
  std::vector<double>& upper = model.upper_;
  HighsHessian& Q = model.Q_;
  double& offset = model.offset_;

  n_pre = n;
  m_pre = m;

  // See "Preprocessing for quadratic programming", Gould, Toint, Math Program
  fixed_vars_ = 0;
  for (Int i = 0; i < n; ++i)
    if (lower[i] == upper[i]) ++fixed_vars_;

  if (fixed_vars_ > 0) {
    fixed_at_.assign(n, kHighsInf);
    std::vector<Int> index_to_remove{};
    for (Int j = 0; j < n; ++j) {
      if (lower[j] == upper[j]) {
        fixed_at_[j] = lower[j];
        index_to_remove.push_back(j);
        const double xcol = fixed_at_[j];

        offset += c[j] * xcol + 0.5 * Q.diag(j) * xcol * xcol;

        for (Int el = A.start_[j]; el < A.start_[j + 1]; ++el) {
          const Int row = A.index_[el];
          const double val = A.value_[el];
          b[row] -= val * xcol;
        }

        for (Int colQ = 0; colQ < j; ++colQ) {
          for (Int el = Q.start_[colQ]; el < Q.start_[colQ + 1]; ++el) {
            const Int rowQ = Q.index_[el];
            if (rowQ == j) {
              c[colQ] += Q.value_[el] * xcol;
            }
          }
        }
        for (Int el = Q.start_[j]; el < Q.start_[j + 1]; ++el) {
          const Int rowQ = Q.index_[el];
          c[rowQ] += Q.value_[el] * xcol;
        }
      }
    }

    HighsIndexCollection index_collection;
    create(index_collection, index_to_remove.size(), index_to_remove.data(), n);
    A.deleteCols(index_collection);
    Q.deleteCols(index_collection);

    Int next = 0;
    Int copy_to = 0;
    for (Int i = 0; i < n; ++i) {
      if (next < index_to_remove.size() && i == index_to_remove[next]) {
        ++next;
        continue;
      } else {
        c[copy_to] = c[i];
        lower[copy_to] = lower[i];
        upper[copy_to] = upper[i];
        copy_to++;
      }
    }

    n -= fixed_vars_;
    assert(A.num_col_ == n);
    assert(Q.dim_ == n);
    c.resize(n);
    lower.resize(n);
    upper.resize(n);
  }

  n_post = n;
  m_post = m;
}

void RemoveFixedVars::undo(PrePostProcessPoint& point) const {
  point.assertConsistency(n_post, m_post);

  if (fixed_vars_ > 0) {
    // Add primal and dual variables for fixed variables

    std::vector<double> new_x(n_pre, 0.0);
    std::vector<double> new_xl(n_pre, 0.0);
    std::vector<double> new_xu(n_pre, 0.0);
    std::vector<double> new_zl(n_pre, 0.0);
    std::vector<double> new_zu(n_pre, 0.0);

    Int pos{};
    for (Int i = 0; i < n_pre; ++i) {
      if (std::isfinite(fixed_at_[i])) {
        new_x[i] = fixed_at_[i];
        new_xl[i] = 0.0;
        new_xu[i] = 0.0;
        new_zl[i] = kHighsInf;
        new_zu[i] = kHighsInf;

      } else {
        new_x[i] = point.x[pos];
        new_xl[i] = point.xl[pos];
        new_xu[i] = point.xu[pos];
        new_zl[i] = point.zl[pos];
        new_zu[i] = point.zu[pos];
        ++pos;
      }
    }

    point.x = std::move(new_x);
    point.xl = std::move(new_xl);
    point.xu = std::move(new_xu);
    point.zl = std::move(new_zl);
    point.zu = std::move(new_zu);
  }

  point.assertConsistency(n_pre, m_pre);
}

void Scale::apply(Model& model) {
  // Apply Curtis-Reid scaling and scale the problem accordingly

  Int& n = model.n_;
  Int& m = model.m_;
  HighsSparseMatrix& A = model.A_;
  std::vector<double>& b = model.b_;
  std::vector<double>& c = model.c_;
  std::vector<double>& lower = model.lower_;
  std::vector<double>& upper = model.upper_;
  HighsHessian& Q = model.Q_;

  n_pre = n;
  m_pre = m;
  n_post = n;
  m_post = m;

  // check if scaling is needed
  bool need_scaling = false;
  for (Int col = 0; col < n; ++col) {
    for (Int el = A.start_[col]; el < A.start_[col + 1]; ++el) {
      if (std::abs(A.value_[el]) != 1.0) {
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
  std::vector<Int> colexp(n);
  std::vector<Int> rowexp(m);
  CG_iter_scaling =
      CurtisReidScaling(A.start_, A.index_, A.value_, rowexp, colexp);

  // Compute scaling from exponents
  colscale.resize(n);
  rowscale.resize(m);
  for (Int i = 0; i < n; ++i) colscale[i] = std::ldexp(1.0, colexp[i]);
  for (Int i = 0; i < m; ++i) rowscale[i] = std::ldexp(1.0, rowexp[i]);

  bool scaling_failed = isInfVector(colscale) || isNanVector(colscale) ||
                        isInfVector(rowscale) || isNanVector(rowscale);
  if (scaling_failed) {
    colscale.clear();
    rowscale.clear();
    return;
  }

  // *********************************************************************
  // Apply scaling
  // *********************************************************************

  // Column has been scaled up by colscale_[col], so cost is scaled up and
  // bounds are scaled down
  for (Int col = 0; col < n; ++col) {
    c[col] *= colscale[col];
    lower[col] /= colscale[col];
    upper[col] /= colscale[col];
  }

  // Row has been scaled up by rowscale_[row], so b is scaled up
  for (Int row = 0; row < m; ++row) b[row] *= rowscale[row];

  // Each entry of the matrix is scaled by the corresponding row and col
  // factor
  for (Int col = 0; col < n; ++col) {
    for (Int el = A.start_[col]; el < A.start_[col + 1]; ++el) {
      Int row = A.index_[el];
      A.value_[el] *= rowscale[row];
      A.value_[el] *= colscale[col];
    }
  }

  for (Int col = 0; col < Q.dim_; ++col) {
    for (Int el = Q.start_[col]; el < Q.start_[col + 1]; ++el) {
      Int row = Q.index_[el];
      Q.value_[el] *= colscale[row];
      Q.value_[el] *= colscale[col];
    }
  }
}

void Scale::undo(PrePostProcessPoint& point, const Model& model) const {
  point.assertConsistency(n_post, m_post);
  if (scaled()) {
    for (Int i = 0; i < n_post; ++i) {
      point.x[i] *= colscale[i];
      point.xl[i] *= colscale[i];
      point.xu[i] *= colscale[i];
      point.zl[i] /= colscale[i];
      point.zu[i] /= colscale[i];
    }
    for (Int i = 0; i < m_post; ++i) {
      point.y[i] *= rowscale[i];
      point.slack[i] /= rowscale[i];
    }

    // set variables that were ignored
    for (Int i = 0; i < n_post; ++i) {
      if (!model.hasLb(i)) {
        point.xl[i] = kHighsInf;
        point.zl[i] = 0.0;
      }
      if (!model.hasUb(i)) {
        point.xu[i] = kHighsInf;
        point.zu[i] = 0.0;
      }
    }
  }
  point.assertConsistency(n_pre, m_pre);
}

void Reformulate::apply(Model& model, Scale& S) {
  Int& n = model.n_;
  Int& m = model.m_;
  HighsSparseMatrix& A = model.A_;
  std::vector<double>& b = model.b_;
  std::vector<double>& c = model.c_;
  std::vector<double>& lower = model.lower_;
  std::vector<double>& upper = model.upper_;
  std::vector<char>& constraints = model.constraints_;
  HighsHessian& Q = model.Q_;

  n_pre = n;
  m_pre = m;

  Int Annz = A.numNz();

  for (Int i = 0; i < m; ++i) {
    if (constraints[i] != '=') {
      // inequality constraint, add slack variable

      ++n;

      // lower/upper bound for new slack
      if (constraints[i] == '>') {
        lower.push_back(-kHighsInf);
        upper.push_back(0.0);
      } else {
        lower.push_back(0.0);
        upper.push_back(kHighsInf);
      }

      // cost for new slack
      c.push_back(0.0);

      // add column of identity to A_
      std::vector<Int> temp_ind{i};
      std::vector<double> temp_val{1.0};
      A.addVec(1, temp_ind.data(), temp_val.data());

      // set scaling to 1
      if (S.scaled()) S.colscale.push_back(1.0);
    }
  }

  if (model.qp()) completeHessian(n, Q);

  n_post = n;
  m_post = m;
}

void Reformulate::undo(PrePostProcessPoint& point, const Model& model,
                       const Iterate& it) const {
  // Copy x, xl, xu, zl, zu without slacks
  point.x = std::vector<double>(it.x.begin(), it.x.begin() + n_pre);
  point.xl = std::vector<double>(it.xl.begin(), it.xl.begin() + n_pre);
  point.xu = std::vector<double>(it.xu.begin(), it.xu.begin() + n_pre);
  point.zl = std::vector<double>(it.zl.begin(), it.zl.begin() + n_pre);
  point.zu = std::vector<double>(it.zu.begin(), it.zu.begin() + n_pre);

  // force unused entries to have correct value
  for (int i = 0; i < n_pre; ++i) {
    if (!model.hasLb(i)) {
      point.xl[i] = kHighsInf;
      point.zl[i] = 0.0;
    }
    if (!model.hasUb(i)) {
      point.xu[i] = kHighsInf;
      point.zu[i] = 0.0;
    }
  }

  // For the Lagrange multipliers, use slacks from zl and zu, to get correct
  // sign. NB: there is no explicit slack stored for equality constraints.
  point.y.resize(m_pre);
  Int slack_pos = 0;
  for (Int i = 0; i < m_pre; ++i) {
    switch (model.constraint(i)) {
      case '=':
        point.y[i] = it.y[i];
        break;
      case '>':
        point.y[i] = it.zu[n_pre + slack_pos];
        ++slack_pos;
        break;
      case '<':
        point.y[i] = -it.zl[n_pre + slack_pos];
        ++slack_pos;
        break;
    }
  }

  // For x-slacks, use slacks from xl and xu, to get correct sign.
  // NB: there is no explicit slack stored for equality constraints.
  point.slack.resize(m_pre);
  slack_pos = 0;
  for (Int i = 0; i < m_pre; ++i) {
    switch (model.constraint(i)) {
      case '=':
        point.slack[i] = 0.0;
        break;
      case '>':
        point.slack[i] = -it.xu[n_pre + slack_pos];
        ++slack_pos;
        break;
      case '<':
        point.slack[i] = it.xl[n_pre + slack_pos];
        ++slack_pos;
        break;
    }
  }
  point.assertConsistency(n_pre, m_pre);
}

void PrePostProcess::apply(Model& model) {
  RER.apply(model);
  stack.push(typeRemoveEmptyRows);

  RFV.apply(model);
  stack.push(typeRemoveFixedVars);

  S.apply(model);
  stack.push(typeScale);

  Ref.apply(model, S);
  stack.push(typeReformulate);
}
void PrePostProcess::undo(PrePostProcessPoint& point, const Model& model,
                          const Iterate& it) const {
  while (!stack.empty()) {
    Type type = stack.top();
    stack.pop();
    switch (type) {
      case typeRemoveEmptyRows:
        RER.undo(point);
        break;
      case typeRemoveFixedVars:
        RFV.undo(point);
        break;
      case typeScale:
        S.undo(point, model);
        break;
      case typeReformulate:
        Ref.undo(point, model, it);
        break;
    }
  }
}
}  // namespace hipo
