/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2021 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file util/HSet.cpp
 * @brief
 * @author Julian Hall, Ivet Galabova, Qi Huangfu and Michael Feldmeier
 */
#include "util/HSet.h"

#include <cassert>

bool HSet::setup(const int size, const int max_entry, const bool output_flag,
                 FILE* log_file, const bool debug, const bool allow_assert) {
  setup_ = false;
  if (size <= 0) return false;
  if (max_entry < min_entry) return false;
  max_entry_ = max_entry;
  debug_ = debug;
  allow_assert_ = allow_assert;
  output_flag_ = output_flag;
  log_file_ = log_file;
  entry_.resize(size);
  pointer_.assign(max_entry_ + 1, no_pointer);
  count_ = 0;
  setup_ = true;
  return true;
}

void HSet::clear() {
  if (!setup_) setup(1, 0);
  pointer_.assign(max_entry_ + 1, no_pointer);
  count_ = 0;
  if (debug_) debug();
}

bool HSet::add(const int entry) {
  if (entry < min_entry) return false;
  if (!setup_) setup(1, entry);
  if (entry > max_entry_) {
    // Entry exceeds what's allowable so far so can't be in the list
    pointer_.resize(entry + 1);
    for (int ix = max_entry_ + 1; ix < entry; ix++) pointer_[ix] = no_pointer;
    max_entry_ = entry;
  } else if (pointer_[entry] > no_pointer) {
    // Duplicate
    if (debug_) debug();
    return false;
  }
  // New entry
  int size = entry_.size();
  if (count_ == size) {
    size++;
    entry_.resize(size);
  }
  pointer_[entry] = count_;
  entry_[count_++] = entry;
  if (debug_) debug();
  return true;
}

bool HSet::remove(const int entry) {
  if (!setup_) {
    setup(1, 0);
    if (debug_) debug();
    return false;
  }
  if (entry < min_entry) return false;
  if (entry > max_entry_) return false;
  int pointer = pointer_[entry];
  if (pointer == no_pointer) return false;
  pointer_[entry] = no_pointer;
  if (pointer < count_ - 1) {
    int last_entry = entry_[count_ - 1];
    entry_[pointer] = last_entry;
    pointer_[last_entry] = pointer;
  }
  count_--;
  if (debug_) debug();
  return true;
}

bool HSet::in(const int entry) const {
  if (entry < min_entry) return false;
  if (entry > max_entry_) return false;
  return pointer_[entry] != no_pointer;
}

bool HSet::debug() const {
  if (!setup_) {
    if (output_flag_) fprintf(log_file_, "HSet: ERROR setup_ not called\n");
    if (allow_assert_) assert(setup_);
    return false;
  }
  bool max_entry_ok = max_entry_ >= min_entry;
  if (!max_entry_ok) {
    if (output_flag_) {
      fprintf(log_file_, "HSet: ERROR max_entry_ = %d < %d\n", max_entry_,
              min_entry);
      print();
    }
    if (allow_assert_) assert(max_entry_ok);
    return false;
  }
  int size = entry_.size();
  bool size_count_ok = size >= count_;
  if (!size_count_ok) {
    if (output_flag_) {
      fprintf(log_file_,
              "HSet: ERROR entry_.size() = %d is less than count_ = %d\n", size,
              count_);
      print();
    }
    if (allow_assert_) assert(size_count_ok);
    return false;
  }
  // Check pointer_ is consistent with count_ and entry_
  int count = 0;
  for (int ix = 0; ix <= max_entry_; ix++) {
    int pointer = pointer_[ix];
    if (pointer == no_pointer) continue;
    bool pointer_ok = pointer >= 0 && pointer < count_;
    if (!pointer_ok) {
      if (output_flag_) {
        fprintf(log_file_, "HSet: ERROR pointer_[%d] = %d is not in [0, %d]\n",
                ix, pointer, count_);
        print();
      }
      if (allow_assert_) assert(pointer_ok);
      return false;
    }
    count++;
    int entry = entry_[pointer];
    bool entry_ok = entry == ix;
    if (!entry_ok) {
      if (output_flag_) {
        fprintf(log_file_, "HSet: ERROR entry_[%d] is %d, not %d\n", pointer,
                entry, ix);
        print();
      }
      if (allow_assert_) assert(entry_ok);
      return false;
    }
  }
  bool count_ok = count == count_;
  if (!count_ok) {
    if (output_flag_) {
      fprintf(log_file_, "HSet: ERROR pointer_ has %d pointers, not %d\n",
              count, count_);
      print();
    }
    if (allow_assert_) assert(count_ok);
    return false;
  }
  return true;
}

void HSet::print() const {
  if (!setup_) return;
  if (log_file_ == NULL) return;
  int size = entry_.size();
  fprintf(log_file_, "\nSet(%d, %d):\n", size, max_entry_);
  fprintf(log_file_, "Pointers: Pointers|");
  for (int ix = 0; ix <= max_entry_; ix++) {
    if (pointer_[ix] != no_pointer) fprintf(log_file_, " %4d", pointer_[ix]);
  }
  fprintf(log_file_, "\n");
  fprintf(log_file_, "          Entries |");
  for (int ix = 0; ix <= max_entry_; ix++) {
    if (pointer_[ix] != no_pointer) fprintf(log_file_, " %4d", ix);
  }
  fprintf(log_file_, "\n");
  fprintf(log_file_, "Entries:  Indices |");
  for (int ix = 0; ix < count_; ix++) fprintf(log_file_, " %4d", ix);
  fprintf(log_file_, "\n");
  fprintf(log_file_, "          Entries |");
  for (int ix = 0; ix < count_; ix++) fprintf(log_file_, " %4d", entry_[ix]);
  fprintf(log_file_, "\n");
}
