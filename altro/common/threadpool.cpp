#include <fmt/format.h>
#include <fmt/chrono.h>

#include "altro/common/threadpool.hpp"

namespace altro {

constexpr int ThreadPool::kTaskTimeout;

ThreadPool::ThreadPool(ThreadPool&& other) noexcept
    : is_running_(static_cast<bool>(other.is_running_)),
      threads_(std::move(other.threads_)),
      futures_(std::move(other.futures_)),
      queue_(std::move(other.queue_)),
      timeout_(other.timeout_) {
  ALTRO_ASSERT(!is_running_, "Cannot move a pool while it's running");
}

ThreadPool& ThreadPool::operator=(ThreadPool&& other) noexcept {
  ALTRO_ASSERT(!other.IsRunning(), "Cannot move a pool while it's running");
  is_running_ = static_cast<bool>(other.is_running_);
  threads_ = std::move(other.threads_);
  futures_ = std::move(other.futures_);
  queue_ = std::move(other.queue_);
  timeout_ = other.timeout_;
  return *this;
}

void ThreadPool::Wait() {
  for (const std::future<void>& future : futures_) {
    std::future_status status = future.wait_for(timeout_);
    if (status == std::future_status::timeout) {
      fmt::print("Task timed out after {}\n", timeout_);
    }
  }
  futures_.clear();
}

void ThreadPool::LaunchThreads(int nthreads) {
  ALTRO_ASSERT(!IsRunning(), "Cannot launch threads when they're already running.");
  if (IsRunning()) {
    return;
  }

  is_running_ = true;
  threads_.clear();
  try {
    for (int i = 0; i < nthreads; ++i) {
      auto kernel = [this, i]() -> void { this->WorkerThread(i); };
      threads_.emplace_back(kernel);
    }
  } catch (...) {
    is_running_ = false;
    throw;
  }
}

void ThreadPool::StopThreads() {
  is_running_ = false;
  for (std::thread& thread : threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

void ThreadPool::WorkerThread(int id) {
  (void)id;
  while (is_running_) {
    std::packaged_task<void()> task;
    if (queue_.TryPop(task)) {
      task();
    } else {
      std::this_thread::yield();
    }
  }
}

}  // namespace altro