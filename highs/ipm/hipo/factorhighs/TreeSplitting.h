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

}  // namespace hipo
#endif