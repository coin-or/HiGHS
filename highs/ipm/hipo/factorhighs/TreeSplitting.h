#ifndef FACTORHIGHS_TREE_SPLITTING_H
#define FACTORHIGHS_TREE_SPLITTING_H

#include <map>
#include <vector>

#include "ipm/hipo/auxiliary/IntConfig.h"

namespace hipo {

enum NodeType { single, subtree };
struct NodeData {
  NodeType type;
  std::vector<Int> firstdesc;
  std::vector<Int> group;
};

class TreeSplitting {
  // Information to split the elimination tree. Each entry in split_
  // correspond to a task that is executed in parallel.
  // split_ contains pairs (sn, data):
  // - If data.type is single, then the task processes only the supernode sn.
  // - If data.type is subtree, then the task processes each subtree rooted at
  //    data.group[i]. Each subtree requires processing supernodes j,
  //    data.firstdesc[i] <= j <= data.group[i].
  std::map<Int, NodeData> split_;

  // For each supernode, belong[sn] is true if sn is found in the split_ data
  // structure. Avoids too many lookups into the map.
  std::vector<bool> belong_;

 public:
  void resize(Int sn_count);

  NodeData& insert(Int sn);
  NodeData& insertSingle(Int sn);
  NodeData& insertSubtree(Int sn);

  const NodeData* find(Int sn) const;

  bool belong(Int sn) const { return belong_[sn]; }
  Int tasks() const { return split_.size(); }
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
// c -> {single, -, -}
// e -> {subtree, {Fe, Fg}, {e,g}}
// f -> {single, -, -}
// where Fj is the first descendant of node j.
// belong_[j] is true for j=p,a,c,e,f and false for j=b,d,g.
//
// - When a is ran, a task is executed that executes the whole subtree of nodes
//   a, b and d.
// - When b is ran, nothing happens, since it was already executed as part of
//   the task that executed a.
// - When c is ran, a task is created that executes only that supernode.
// - When d is ran, nothing happens, since it was already executed as part of
//   the task that executed a.
// - When e is ran, a task is executed that executes the whole subtree of nodes
//   e and f.
// - When f is ran, a task is created that executes only that supernode.
// - When g is ran, nothing happens, since it was already executed as part of
//   the task that executed e.
//
// It is important that node a is processed before b and d, so that when b or d
// are synced, their operations are already performed. Otherwise, syncing node b
// would complete even though the operations for node b have not been performed.
// In other words, children should be synced in forward order, and thus spawned
// in reverse order.
//

}  // namespace hipo
#endif