#include <memory>
#include <vector>

#include "altro/common/threadpool.hpp"

namespace altro {

class Work {
 public:
  void Kernel() {
    count_ += 1;
    usleep(1000);
  }
  int GetCount() const { return count_; }
 private:
  int count_ = 0;
};

class Launcher {
 public:
  Launcher(int N, int nthreads) {
    for (int i = 0; i < N; ++i) {
      workset_.emplace_back(std::make_unique<Work>());
    }
    double step = N / static_cast<double>(nthreads);
    for (double val = 0.0; val <= N; val += step) {
      inds_.emplace_back(static_cast<int>(round(val)));
    }
  }

  void WorkBlock(int start, int stop) {
    for (int i = start; i < stop; ++i) {
      workset_[i]->Kernel();
    }
  }

  void CreateTasks() {
    for (int i = 0; i < NumThreads(); ++i) {
      tasks_.emplace_back(std::bind(&Launcher::WorkBlock, this, inds_[i], inds_[i+1]));
    }
  }

  void CreateTasks(ThreadPool& pool) {
    for (int i = 0; i < NumThreads(); ++i) {
      int start = inds_[i];
      int stop = inds_[i + 1];
      auto work = [this, start, stop]() { this->WorkBlock(start, stop); };
      pool.AddTask(work);
    }
  }

  int NumThreads() const { return inds_.size() - 1; }
  int NumWork() const { return workset_.size(); }

  void PrintCounts() {
    for (int i = 0; i < NumWork(); ++i) {
      fmt::print("{}\n", workset_[i]->GetCount());
    }
  }

  int SumCounts() {
    int count = 0;
    for (int i = 0; i < NumWork(); ++i) {
      count += workset_[i]->GetCount();
    }
    return count;
  }

 private:
  std::vector<std::unique_ptr<Work>> workset_;
  std::vector<int> inds_;
  std::vector<std::packaged_task<void()>> tasks_;
};

}  // namespace altro