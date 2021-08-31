// Copyright [2021] Optimus Ride Inc.

#include "altro/common/timer.hpp"
#include "altro/common/profile_entry.hpp"
#include "altro/utils/assert.hpp"
#include "fmt/chrono.h"

namespace altro {

Timer::~Timer() { 
  if (!printed_summary_ && active_) {
    PrintSummary(); 
  }
  // Close the file if the Timer has ownship
  if (using_file_) {
    fclose(io_);
  }
}

void Timer::PrintSummary() {
  PrintSummary(&times_);
}

void Timer::PrintSummary(std::map<std::string, std::chrono::microseconds>* const times_ptr) {
  std::map<std::string, std::chrono::microseconds>& times = *times_ptr;

  using EntryPtr = std::shared_ptr<ProfileEntry>;
  std::vector<EntryPtr> entries;
  std::vector<EntryPtr> parents;

  // Add first (top level) entry
  // This is just temporary to provide the total time
  auto times_iter = times.begin();
  entries.emplace_back(std::make_shared<ProfileEntry>("top", std::chrono::microseconds(0)));
  parents.emplace_back(entries.front());

  // Add all the logged entries, assigning parents
  // Each ProfileEntry object splits the "stack" of names (e.g. "al/ilqr/cost" => ["al", "ilqr", "cost"])
  // and the total time taken by the function. 
  // Each entry also stores a pointer to it's parent (e.g. "al/ilqr" for "al/ilqr/cost")
  // This builds out a call "tree" with the root being a temporary entry that will 
  // calculate the total time recorded by the "base" level functions (e.g. "al").
  int max_width = 0;
  for (; times_iter != times.end(); ++times_iter) {
    // Create a new entry from the data in the Timer
    entries.emplace_back(std::make_shared<ProfileEntry>(times_iter->first, times_iter->second));
    EntryPtr entry = entries.back();

    // Get how "deep" the profile stack is.
    // e.g. level of "al/ilqr/cost" is 3.
    size_t curlevel = entry->NumLevels(); 

    // Add another level if needed
    // The "parents" vector contains the current parent for each level
    // parents[0] is equal to the base level, so it's length is equal to the
    // maximum "curlevel" seen so far +1.
    if (curlevel >= parents.size()) {
      parents.resize(curlevel + 1);  // the +1 accounts for the 0-level.
    }

    // Always set the current entry as the parent for it's level
    parents[curlevel] = entry;

    // Assign the parent to be the last entry in the previous level
    // This is ensured by the fact that the entries are sorted alphabetically
    entry->parent = parents[curlevel - 1];

    // Add up all the base level entries to get the total time
    if (curlevel == 1) {
      entries.front()->time += entry->time;
    }

    // Keep track of the total width needed to print the field names
    // Gets the length of e.g. "al/ilqr/cost" without the "/" delimiters.
    int name_width = 0;
    for (const std::string& part : entry->name) {
      name_width += part.length();
    }
    max_width = std::max(max_width, name_width);
  }

  // Print the header
  const int pad = 2;
  max_width += pad;
  std::string header = fmt::format("{1:<{0}}  {2:>8}  {3:>7}  {4:>7}", max_width, "Description",
                                   "Time (us)", "%Total", "%Parent");
  fmt::print(io_, "{}\n", header);
  fmt::print(io_, "{:-^{}}\n", "", header.length());
  for (size_t i = 1; i < entries.size(); ++i) {
    entries[i]->CalcStats();
    entries[i]->Print(io_, max_width);
  }
  printed_summary_ = true;
}

Stopwatch Timer::Start(const std::string& name) {
  if (IsActive()) {
    stack_.emplace_back(name);
    std::string fullname = stack_.front();
    for (size_t i = 1; i < stack_.size(); ++i) {
      fullname += "/" + stack_[i];
    }
    return Stopwatch(std::move(fullname), shared_from_this());
  }
  return Stopwatch();
}

void Timer::SetOutput(const std::string& filename) {
  FILE* io = fopen(filename.c_str(), "w");
  if (io == nullptr) {
    std::string errmsg = fmt::format("Error opening profiler file \"{}\". Got errno {}.", filename, errno);
    throw std::runtime_error(errmsg);
  }
  SetOutput(io);

  // Set a flag that ensures the file will be closed when the Timer is destroyed
  using_file_ = true;
}

Stopwatch::Stopwatch(std::string name, std::shared_ptr<Timer> timer) 
    : name_(std::move(name)), 
      start_(std::chrono::high_resolution_clock::now()), 
      parent_(std::move(timer)) {}

Stopwatch::~Stopwatch() {
  // NOTE: Parent will be nullptr if the timer is deactivated and Stopwatch is default-constructed.
  if (parent_) {  
    // Record duration and log the time with the parent timer.
    auto duration = std::chrono::high_resolution_clock::now() - start_;
    microseconds us = std::chrono::duration_cast<microseconds>(duration);
    parent_->stack_.pop_back();  // pop the name off the top of the stack (FILO stack)
    parent_->times_[name_] += us;
  }
}


}  // namespace altro