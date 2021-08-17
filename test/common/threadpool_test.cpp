#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <fmt/format.h>
#include <fmt/chrono.h>
#include <thread>

#include "altro/common/threadsafe_queue.hpp"
#include "altro/common/threadpool.hpp"
#include "perf/task_launcher.hpp"

namespace altro {

TEST(ThreadSafeQueueTest, Construction) {
  ThreadSafeQueue<int> queue;
  (void) queue;
}

TEST(ThreadSafeQueueTest, PushAndPop) {
  ThreadSafeQueue<int> q;
  const int val = 3;
  q.Push(val);
  EXPECT_EQ(q.Size(), 1);
  int res;
  bool success = q.TryPop(res);
  EXPECT_EQ(res, val);
  EXPECT_TRUE(success);
  success = q.TryPop(res);
  EXPECT_EQ(res, val);
  EXPECT_FALSE(success);
  EXPECT_EQ(q.Size(), 0);
}

TEST(ThreadSafeQueueTest, Move) {
  ThreadSafeQueue<int> q1;
  q1.Push(2);
  q1.Push(1);
  EXPECT_EQ(q1.Size(), 2);
  ThreadSafeQueue<int> q2(std::move(q1));
  EXPECT_EQ(q2.Size(), 2);
  int val;
  bool success = q2.TryPop(val);
  EXPECT_EQ(val, 2);
  EXPECT_TRUE(success);

  ThreadSafeQueue<int> q3;
  q3.Push(1);
}

TEST(ThreadPool, Move) {
  const int N = 16;
  const int nthreads = 4;
  Launcher launcher(N, nthreads);
  ThreadPool pool;
  launcher.CreateTasks(pool);

  ThreadPool pool2(std::move(pool));
  EXPECT_EQ(pool2.NumTasks(), 4);
  EXPECT_FALSE(pool2.IsRunning());
  pool2.LaunchThreads(nthreads);
  EXPECT_TRUE(pool2.IsRunning());
  pool2.Wait();
  EXPECT_EQ(launcher.SumCounts(), N);
  
  pool2.StopThreads(); // Must stop threads before moving
  ThreadPool pool3 = std::move(pool2);
  pool3.LaunchThreads(nthreads);
  EXPECT_TRUE(pool3.IsRunning());
  launcher.CreateTasks(pool3);
  pool3.Wait();
  EXPECT_EQ(launcher.SumCounts(), 2 * N);
}

TEST(ThreadPool, LaunchOnly) {
  const int N = 16;
  const int nthreads = 4;
  Launcher launcher(N, nthreads);
  ThreadPool pool;
  launcher.CreateTasks(pool);
  EXPECT_EQ(pool.NumTasks(), nthreads);
  EXPECT_EQ(launcher.SumCounts(), 0);
  EXPECT_EQ(pool.NumThreads(), 0);
  pool.LaunchThreads(nthreads);
  EXPECT_EQ(pool.NumThreads(), nthreads);
  // EXPECT_EQ(launcher.SumCounts(), 0);
}

TEST(ThreadPool, RunOnce) {
  const int N = 16;
  const int nthreads = 4;
  Launcher launcher(N, nthreads);
  ThreadPool pool;
  pool.LaunchThreads(nthreads);
  launcher.CreateTasks(pool);
  pool.Wait();
  EXPECT_EQ(launcher.SumCounts(), N);
}

TEST(ThreadPool, RunTwice) {
  const int N = 16;
  const int nthreads = 4;
  Launcher launcher(N, nthreads);
  ThreadPool pool;
  launcher.CreateTasks(pool);
  pool.LaunchThreads(nthreads);
  pool.Wait();
  EXPECT_EQ(launcher.SumCounts(), N);
  launcher.CreateTasks(pool);
  pool.Wait();
  EXPECT_EQ(launcher.SumCounts(), 2 * N);
}

TEST(ThreadPool, CheckRunning) {
  const int N = 16;
  const int nthreads = 4;
  Launcher launcher(N, nthreads);
  ThreadPool pool;
  launcher.CreateTasks(pool);
  EXPECT_FALSE(pool.IsRunning());
  pool.LaunchThreads(nthreads);
  EXPECT_TRUE(pool.IsRunning());
  EXPECT_EQ(pool.NumThreads(), nthreads);
  pool.StopThreads();
  EXPECT_FALSE(pool.IsRunning());
  EXPECT_EQ(pool.NumThreads(), 0);
}

TEST(ThreadPool, StopAndAdd) {
  const int N = 16;
  const int nthreads = 4;
  Launcher launcher(N, nthreads);
  ThreadPool pool;
  launcher.CreateTasks(pool);
  EXPECT_EQ(pool.NumTasks(), nthreads);
  pool.LaunchThreads(nthreads);
  pool.Wait();
  pool.StopThreads();
  EXPECT_EQ(pool.NumTasks(), 0);

  // Add the tasks back again
  launcher.CreateTasks(pool);
  EXPECT_EQ(pool.NumTasks(), nthreads);
  pool.LaunchThreads(nthreads / 2);
  pool.Wait();
  EXPECT_EQ(launcher.SumCounts(), 2 * N);
}


} // namespace altro
