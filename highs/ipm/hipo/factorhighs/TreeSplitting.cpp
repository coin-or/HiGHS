#include "TreeSplitting.h"

namespace hipo {

void TreeSplitting::resize(Int sn_count) { split_.resize(sn_count); }

NodeData& TreeSplitting::insert(Int sn) {
  if (!split_[sn]) task_count_++;
  split_[sn].reset(new NodeData);
  return *split_[sn];
}

NodeData& TreeSplitting::insertSingle(Int sn) {
  NodeData& data = insert(sn);
  data.type = NodeType::single;
  return data;
}

NodeData& TreeSplitting::insertSubtree(Int sn) {
  NodeData& data = insert(sn);
  data.type = NodeType::subtree;
  return data;
}

const NodeData* TreeSplitting::find(Int sn) const { return split_[sn].get(); }

}  // namespace hipo