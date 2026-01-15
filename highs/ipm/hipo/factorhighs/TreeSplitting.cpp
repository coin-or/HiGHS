#include "TreeSplitting.h"

namespace hipo {

void TreeSplitting::resize(Int sn_count) { belong_.resize(sn_count, false); }

NodeData& TreeSplitting::insert(Int sn) {
  auto res_insert = split_.insert({sn, {}});
  NodeData& data = res_insert.first->second;
  belong_[sn] = true;
  return data;
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

const NodeData* TreeSplitting::find(Int sn) const {
  const NodeData* ptr = nullptr;
  if (belong_[sn]) ptr = &(split_.find(sn)->second);
  return ptr;
}

}  // namespace hipo