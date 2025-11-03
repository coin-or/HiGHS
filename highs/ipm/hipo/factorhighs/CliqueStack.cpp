#include "CliqueStack.h"

#include <cassert>
#include <cstring>

namespace hipo {

CliqueStack::CliqueStack(Int stack_size) {
  stack_.assign(stack_size, 0.0);
  top_ = 0;
  workspace_ = nullptr;
  worksize_ = 0;
}

double* CliqueStack::setup(Int clique_size, bool& reallocation) {
  // Clear workspace

  assert(!workspace_ && !worksize_);
  reallocation = false;

  // This should not trigger reallocation, because the reserve in init is done
  // with the maximum possible size of a clique.
  if (top_ + clique_size > stack_.size()) {
    reallocation = true;
    stack_.resize(top_ + clique_size, 0.0);
  }

  workspace_ = &stack_[top_];
  worksize_ = clique_size;

  // initialize workspace to zero
  std::memset(workspace_, 0, worksize_ * sizeof(double));

  return workspace_;
}

const double* CliqueStack::get() const { return stack_.data(); }
Int CliqueStack::getTop() const { return top_; }

const double* CliqueStack::getChild(Int& child_sn) const {
  // Get the top of the stack, in terms of supernode ID of the child and pointer
  // to its data.

  child_sn = sn_pushed_.top().first;
  Int child_size = sn_pushed_.top().second;
  const double* child = &stack_[top_ - child_size];

  return child;
}

void CliqueStack::popChild() {
  // Remove top child from the stack

  Int child_size = sn_pushed_.top().second;
  sn_pushed_.pop();

  top_ -= child_size;
}

void CliqueStack::pushWork(Int sn) {
  // Put the content of the workspace at the top of the stack

  // stack_[top_] has lower address than workspace, so no need to resize.
  // workspace_ and stack_[top_] do not overlap, so use memcpy
  std::memcpy(&stack_[top_], workspace_, worksize_ * sizeof(double));

  top_ += worksize_;

  // keep track of supernodes pushed
  sn_pushed_.push({sn, worksize_});

  worksize_ = 0;
  workspace_ = nullptr;
}

}  // namespace hipo