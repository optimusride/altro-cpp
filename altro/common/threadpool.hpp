// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "altro/utils/assert.hpp"
#include "altro/common/threadsafe_queue.hpp"

namespace altro {

/**
 * @brief Basic threadpool that uses a single work queue.
 *
 * The threadpool is created via `LaunchThreads(nthreads)`, which creates 
 * `nthreads` "spinning" threads. Tasks can be added to the work queue via `AddTask`,
 * which accepts anything that can be converted to a `std::packaged_task<void()>`.
 * The user can wait for all the submitted jobs to finish via `Wait`.
 * 
 * The pool is not copyable, but is moveable.
 *
 * # Usage
 * @code {.cpp}
 * ThreadPool pool;
 *
 * # Create the tasks
 * pool.AddTask(std::packaged_task<void()>(...));
 *
 * # Launch the threads once all tasks have been created
 * pool.LaunchThreads();
 *
 * # Wait for the threads to finish
 * pool.Wait();
 * 
 * # Close the thread pool
 * pool.StopThreads();
 * @endcode
 *
 * Based off of Listing 9.1 in C++ Concurrency in Action by Anthony Williams.
 */
class ThreadPool {
 public:
  ThreadPool() : is_running_(false) {}

  // Disallow copying
  ThreadPool(const ThreadPool& other) = delete;
  ThreadPool& operator=(const ThreadPool& other) = delete;

  // Allow moving
  ThreadPool(ThreadPool&& other) noexcept;
  ThreadPool& operator=(ThreadPool&& other) noexcept;

  ~ThreadPool() {
    if (IsRunning()) {
      StopThreads();
    }
  }

  /**
   * @brief Add a new task to the pool.
   *
   * @param task A void function that takes no inputs. Can be anything that can 
   * be converted to a `std::packaged_task<void()>`.
   */
  template <class Task>
  void AddTask(const Task& task) {
    std::packaged_task<void()> ptask(task);
    futures_.emplace_back(ptask.get_future());
    queue_.Push(std::move(ptask));
  }


  /**
   * @brief Number of tasks in the work queue.
   *
   */
  size_t NumTasks() const { return queue_.Size(); }

  /**
   * @brief Number of threads in the threadpool
   * 
   */
  size_t NumThreads() const {return threads_.size(); }

  /**
   * @brief Wait for all the tasks in the queue to finish.
   *
   *
   */
  void Wait();

  /**
   * @brief Launch the threadpool
   * 
   * Any threads in the queue will immediately start executing.
   *
   */
  void LaunchThreads(int nthreads);

  /**
   * @brief Stop the threads that are currently running
   * 
   */
  void StopThreads();

  /**
   * @brief Checks if the threadpool is currently launched and running
   * 
   * Does not indicate if a task is running. The threads may just be spinning.
   * 
   */
  bool IsRunning() const { return is_running_; }

  /**
   * @brief Set the length of the timeout for each task after calling Wait().
   * 
   * Default is 10 seconds.
   * 
   * Note that this is not the total amount of time that `Wait` may take, but rather 
   * the time per task.
   *
   * @tparam Rep
   * @tparam Period
   * @param timeout Length of the final timeout for each of the threads.
   */
  template <class Rep, class Period>
  void SetTimeoutPerTask(std::chrono::duration<Rep, Period> timeout) {
    timeout_ = timeout;
  }

 private:
  /**
   * @brief Function executed by each of the threads
   *
   * @param id Thread id. Corresponds to the task index.
   */
  void WorkerThread(int id);

  static constexpr int kTaskTimeout = 10;
  std::atomic_bool is_running_;             // Are the threads running 
  std::vector<std::thread> threads_;        // Spinning thread pool
  std::vector<std::future<void>> futures_;  // Futures for each of the submitted tasks
  ThreadSafeQueue<std::packaged_task<void()>> queue_;  // Tasks to complete
  std::chrono::nanoseconds timeout_ = std::chrono::seconds(kTaskTimeout);  // Timeout for each task
};

}  // namespace altro