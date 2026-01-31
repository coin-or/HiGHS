#ifndef HIPO_PRE_POST_PROCESS
#define HIPO_PRE_POST_PROCESS

#include <stack>
#include <vector>

#include "ipm/hipo/auxiliary/IntConfig.h"

namespace hipo {

class Model;
class Iterate;

struct PrePostProcessPoint {
  std::vector<double>& x;
  std::vector<double>& xl;
  std::vector<double>& xu;
  std::vector<double>& slack;
  std::vector<double>& y;
  std::vector<double>& zl;
  std::vector<double>& zu;

  void assertConsistency(Int n, Int m) const;
};

struct RemoveEmptyRows {
  Int n_pre, m_pre, n_post, m_post;
  std::vector<Int> rows_shift;
  Int empty_rows{};

  void apply(Model& model);
  void undo(PrePostProcessPoint& point) const;
};

struct RemoveFixedVars {
  Int n_pre, m_pre, n_post, m_post;
  Int fixed_vars_{};
  std::vector<double> fixed_at_;

  void apply(Model& model);
  void undo(PrePostProcessPoint& point) const;
};

struct Scale {
  Int n_pre, m_pre, n_post, m_post;
  std::vector<double> colscale, rowscale;
  Int CG_iter_scaling;

  void apply(Model& model);
  void undo(PrePostProcessPoint& point, const Model& model) const;
  bool scaled() const { return !colscale.empty(); }
};

struct Reformulate {
  Int n_pre, m_pre, n_post, m_post;

  void apply(Model& model, Scale& scale);
  void undo(PrePostProcessPoint& point, const Model& model,
            const Iterate& it) const;
};

struct PrePostProcess {
  RemoveEmptyRows RER;
  RemoveFixedVars RFV;
  Scale S;
  Reformulate Ref;

  enum Type {
    typeRemoveEmptyRows,
    typeRemoveFixedVars,
    typeScale,
    typeReformulate
  };
  mutable std::stack<Type> stack;

  void apply(Model& model);
  void undo(PrePostProcessPoint& point, const Model& model,
            const Iterate& it) const;
};

}  // namespace hipo

#endif
