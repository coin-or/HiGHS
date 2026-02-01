#ifndef HIPO_PRE_POST_PROCESS
#define HIPO_PRE_POST_PROCESS

#include <cassert>
#include <memory>
#include <stack>
#include <vector>

#include "ipm/hipo/auxiliary/IntConfig.h"

namespace hipo {

class Model;
class Iterate;

struct PreprocessorPoint {
  std::vector<double>& x;
  std::vector<double>& xl;
  std::vector<double>& xu;
  std::vector<double>& slack;
  std::vector<double>& y;
  std::vector<double>& zl;
  std::vector<double>& zu;

  void assertConsistency(Int n, Int m) const;
};

struct PreprocessAction {
  virtual ~PreprocessAction() = default;
  virtual void apply(Model& model) = 0;
  virtual void undo(PreprocessorPoint& point, const Model& model,
                    const Iterate& it) const = 0;
};

struct PreprocessEmptyRows : public PreprocessAction {
  Int n_pre, m_pre, n_post, m_post;
  std::vector<Int> rows_shift;
  Int empty_rows{};

  void apply(Model& model) override;
  void undo(PreprocessorPoint& point, const Model& model,
            const Iterate& it) const override;
};

struct PreprocessFixedVars : public PreprocessAction {
  Int n_pre, m_pre, n_post, m_post;
  Int fixed_vars_{};
  std::vector<double> fixed_at_;

  void apply(Model& model) override;
  void undo(PreprocessorPoint& point, const Model& model,
            const Iterate& it) const override;
};

struct PreprocessScaling : public PreprocessAction {
  Int n_pre, m_pre, n_post, m_post;
  Int CG_iter_scaling;

  void apply(Model& model) override;
  void undo(PreprocessorPoint& point, const Model& model,
            const Iterate& it) const override;
};

struct PreprocessFormulation : public PreprocessAction {
  Int n_pre, m_pre, n_post, m_post;

  void apply(Model& model) override;
  void undo(PreprocessorPoint& point, const Model& model,
            const Iterate& it) const override;
};

struct Preprocessor {
  mutable std::stack<std::unique_ptr<PreprocessAction>> stack;

  void apply(Model& model);
  void undo(PreprocessorPoint& point, const Model& model,
            const Iterate& it) const;
};

}  // namespace hipo

#endif
