#ifndef FACTORHIGHS_TREE_SPLITTING_H
#define FACTORHIGHS_TREE_SPLITTING_H

#include <vector>
#include <memory>

#include "ipm/hipo/auxiliary/IntConfig.h"

namespace hipo {

enum NodeType { single, subtree };
struct NodeData {
  NodeType type;
  std::vector<Int> firstdesc;
  std::vector<Int> group;
};

class TreeSplitting {
  // Information to split the elimination tree.
  // If split_[sn] is not null, then a task is associated with the supernode sn.
  // - If type is single, then the task processes only the supernode sn.
  // - If type is subtree, then the task processes each subtree rooted at
  //   group[i]. Each subtree requires processing supernodes j,
  //    firstdesc[i] <= j <= group[i].
  std::vector<std::unique_ptr<NodeData>> split_;

 private:
  NodeData& insert(Int sn);
  Int task_count_{};

 public:
  void resize(Int sn_count);

  NodeData& insertSingle(Int sn);
  NodeData& insertSubtree(Int sn);

  const NodeData* find(Int sn) const;

  Int tasks() const { return task_count_; }
};

// Consider a supernode p, with children {a,b,c,d,e,f,g}.
// p is NodeType::single.
// c,f are NodeType::single.
// {a,b,d} is a group of small subtrees.
// {e,g} is a group of small subtrees.
//
// Information NodeData::firstdesc and NodeData::group is only used if type is
// subtree. So, split_ looks like this:
// p -> {single, -, -}
// a -> {subtree, {Fa,Fb,Fd}, {a,b,d}}
// b -> null
// c -> {single, -, -}
// d -> null
// e -> {subtree, {Fe, Fg}, {e,g}}
// f -> {single, -, -}
// g -> null
// where Fj is the first descendant of node j.
//
// - When a is run, a task is executed that executes the whole subtree of nodes
//   a, b and d.
// - When b is run, nothing happens, since it was already executed as part of
//   the task that executed a.
// - When c is run, a task is created that executes only that supernode.
// - When d is run, nothing happens, since it was already executed as part of
//   the task that executed a.
// - When e is run, a task is executed that executes the whole subtree of nodes
//   e and g.
// - When f is run, a task is created that executes only that supernode.
// - When g is run, nothing happens, since it was already executed as part of
//   the task that executed e.
//
// It is important that node a is processed before b and d, so that when b or d
// are synced, their operations are already performed. Otherwise, syncing node b
// would complete even though the operations for node b have not been performed.
// When the factorisation is run in serial using a CliqueStack, the children are
// processed in reverse order. To obtain the same result when running the code
// in parallel, the children must be synced in reverse order, and so spawned in
// forward order. To make the TreeSplitting work, the groups of supernodes
// should be formed in reverse order, so that in the group {a,b,d}, d has the
// lowest ordering and a the highest one. In this way, syncing in reverse order
// produces the correct result.

}  // namespace hipo
#endif