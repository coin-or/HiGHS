#include "Auxiliary.h"

#include <stack>

namespace hipo {

void counts2Ptr(std::vector<Int64>& ptr, std::vector<Int64>& w) {
  // Given the column counts in the vector w (of size n),
  // compute the column pointers in the vector ptr (of size n+1),
  // and copy the first n pointers back into w.

  Int64 temp_nz{};
  Int64 n = w.size();
  for (Int64 j = 0; j < n; ++j) {
    ptr[j] = temp_nz;
    temp_nz += w[j];
    w[j] = ptr[j];
  }
  ptr[n] = temp_nz;
}

void inversePerm(const std::vector<Int64>& perm, std::vector<Int64>& iperm) {
  // Given the permutation perm, produce the inverse permutation iperm.
  // perm[i] : i-th entry to use in the new order.
  // iperm[i]: where entry i is located in the new order.

  for (Int64 i = 0; i < perm.size(); ++i) {
    iperm[perm[i]] = i;
  }
}

void subtreeSize(const std::vector<Int64>& parent, std::vector<Int64>& sizes) {
  // Compute sizes of subtrees of the tree given by parent

  Int64 n = parent.size();
  sizes.assign(n, 1);

  for (Int64 i = 0; i < n; ++i) {
    Int64 k = parent[i];
    if (k != -1) sizes[k] += sizes[i];
  }
}

void transpose(const std::vector<Int64>& ptr, const std::vector<Int64>& rows,
               std::vector<Int64>& ptrT, std::vector<Int64>& rowsT) {
  // Compute the transpose of the matrix and return it in rowsT and ptrT

  Int64 n = ptr.size() - 1;

  std::vector<Int64> work(n);

  // count the entries in each row into work
  for (Int64 i = 0; i < ptr.back(); ++i) {
    ++work[rows[i]];
  }

  // sum row sums to obtain pointers
  counts2Ptr(ptrT, work);

  for (Int64 j = 0; j < n; ++j) {
    for (Int64 el = ptr[j]; el < ptr[j + 1]; ++el) {
      Int64 i = rows[el];

      // entry (i,j) becomes entry (j,i)
      Int64 pos = work[i]++;
      rowsT[pos] = j;
    }
  }
}

void transpose(const std::vector<Int64>& ptr, const std::vector<Int64>& rows,
               const std::vector<double>& val, std::vector<Int64>& ptrT,
               std::vector<Int64>& rowsT, std::vector<double>& valT) {
  // Compute the transpose of the matrix and return it in rowsT, ptrT and valT

  Int64 n = ptr.size() - 1;

  std::vector<Int64> work(n);

  // count the entries in each row into work
  for (Int64 i = 0; i < ptr.back(); ++i) {
    ++work[rows[i]];
  }

  // sum row sums to obtain pointers
  counts2Ptr(ptrT, work);

  for (Int64 j = 0; j < n; ++j) {
    for (Int64 el = ptr[j]; el < ptr[j + 1]; ++el) {
      Int64 i = rows[el];

      // entry (i,j) becomes entry (j,i)
      Int64 pos = work[i]++;
      rowsT[pos] = j;
      valT[pos] = val[el];
    }
  }
}

void childrenLinkedList(const std::vector<Int64>& parent,
                        std::vector<Int64>& head, std::vector<Int64>& next) {
  // Create linked lists of children in elimination tree.
  // parent gives the dependencies of the tree,
  // head[node] is the first child of node,
  // next[head[node]] is the second child,
  // next[next[head[node]]] is the third child...
  // until -1 is reached.

  Int64 n = parent.size();
  head.assign(n, -1);
  next.assign(n, -1);
  for (Int64 node = n - 1; node >= 0; --node) {
    if (parent[node] == -1) continue;
    next[node] = head[parent[node]];
    head[parent[node]] = node;
  }
}

void reverseLinkedList(std::vector<Int64>& head, std::vector<Int64>& next) {
  // Reverse the linked list of children of each node.
  // If a node has children (a -> b -> c -> -1), the reverse list contains
  // children (c -> b -> a -> -1).

  const Int64 n = head.size();

  for (Int64 node = 0; node < n; ++node) {
    Int64 prev_node = -1;
    Int64 curr_node = head[node];
    Int64 next_node = -1;

    while (curr_node != -1) {
      next_node = next[curr_node];
      next[curr_node] = prev_node;
      prev_node = curr_node;
      curr_node = next_node;
    }

    head[node] = prev_node;
  }
}

void dfsPostorder(Int64 node, Int64& start, std::vector<Int64>& head,
                  const std::vector<Int64>& next, std::vector<Int64>& order) {
  // Perform depth first search starting from root node and order the nodes
  // starting from the value start. head and next contain the linked list of
  // children.

  std::stack<Int64> stack;
  stack.push(node);

  while (!stack.empty()) {
    const Int64 current = stack.top();
    const Int64 child = head[current];

    if (child == -1) {
      // no children left to order,
      // remove from the stack and order
      stack.pop();
      order[start++] = current;
    } else {
      // at least one child left to order,
      // add it to the stack and remove it from the list of children
      stack.push(child);
      head[current] = next[child];
    }
  }
}

void processEdge(Int64 j, Int64 i, const std::vector<Int64>& first,
                 std::vector<Int64>& maxfirst, std::vector<Int64>& delta,
                 std::vector<Int64>& prevleaf, std::vector<Int64>& ancestor) {
  // Process edge of skeleton matrix.
  // Taken from Tim Davis "Direct Methods for Sparse Linear Systems".

  // j not a leaf of ith row subtree
  if (i <= j || first[j] <= maxfirst[i]) {
    return;
  }

  // max first[j] so far
  maxfirst[i] = first[j];

  // previous leaf of ith row subtree
  Int64 jprev = prevleaf[i];

  // A(i,j) is in the skeleton matrix
  delta[j]++;

  if (jprev != -1) {
    // find least common ancestor of jprev and j
    Int64 q = jprev;
    while (q != ancestor[q]) {
      q = ancestor[q];
    }

    // path compression
    Int64 sparent;
    for (Int64 s = jprev; s != q; s = sparent) {
      sparent = ancestor[s];
      ancestor[s] = q;
    }

    // consider overlap
    delta[q]--;
  }

  // previous leaf of ith subtree set to j
  prevleaf[i] = j;
}

Int64 getDiagStart(Int64 n, Int64 k, Int64 nb, Int64 n_blocks,
                   std::vector<Int64>& start, bool triang) {
  // start position of diagonal blocks for blocked dense formats
  start.assign(n_blocks, 0);
  for (Int64 i = 1; i < n_blocks; ++i) {
    start[i] = start[i - 1] + nb * (n - (i - 1) * nb);
    if (triang) start[i] -= nb * (nb - 1) / 2;
  }

  Int64 jb = std::min(nb, k - (n_blocks - 1) * nb);
  Int64 result = start.back() + (n - (n_blocks - 1) * nb) * jb;
  if (triang) result -= jb * (jb - 1) / 2;
  return result;
}

Clock::Clock() { start(); }
void Clock::start() { t0 = std::chrono::high_resolution_clock::now(); }
double Clock::stop() const {
  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> d = t1 - t0;
  return d.count();
}

}  // namespace hipo
