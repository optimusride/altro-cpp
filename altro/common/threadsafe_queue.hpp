#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <atomic>

namespace altro {

/**
 * @brief A basic thread-safe queue.
 * 
 * Based off of example in C++ Concurrency in Action by Anthony Williams.
 * 
 * @tparam T Type of the queue elements.
 */
template <class T>
class ThreadSafeQueue {
  struct Node {
    std::shared_ptr<T> data;
    std::unique_ptr<Node> next;
  };

 public:
  ThreadSafeQueue()
      : head_mutex_(std::make_unique<std::mutex>()),
        head_(std::make_unique<Node>()),
        tail_mutex_(std::make_unique<std::mutex>()),
        tail_(head_.get()),
        data_cond_(std::make_unique<std::condition_variable>()) {}

  // Disallow copying
  ThreadSafeQueue(const ThreadSafeQueue& other) = delete;
  ThreadSafeQueue& operator=(const ThreadSafeQueue& other) = delete;

  // Allow Moving
  ThreadSafeQueue(ThreadSafeQueue&& other) noexcept;
  ThreadSafeQueue<T>& operator=(ThreadSafeQueue<T>&& other) noexcept;

  bool TryPop(T& value);
  void Push(T new_value);
  void Clear();
  bool IsEmpty();
  size_t Size() const { return *size_; };

 private:
  Node* GetTail() {
    std::lock_guard<std::mutex> tail_lock(*tail_mutex_);
    return tail_;
  }

  std::unique_ptr<Node> PopHead() {
    // NOTE: Lock must be acquired on the head before calling this function
    std::unique_ptr<Node> old_head = std::move(head_);
    head_ = std::move(old_head->next);
    return old_head;
  }

  std::unique_ptr<Node> TryPopHead(T& value) {
    std::lock_guard<std::mutex> head_lock(*head_mutex_);
    if (head_.get() == GetTail()) {  // Empty queue
      return std::unique_ptr<Node>();
    }
    --(*size_);
    value = std::move(*head_->data);
    return PopHead();
  }

  std::unique_ptr<std::atomic_size_t> size_ = std::make_unique<std::atomic_size_t>(0);
  std::unique_ptr<std::mutex> head_mutex_;  // pointer so it can be moved
  std::unique_ptr<Node> head_;
  std::unique_ptr<std::mutex> tail_mutex_;  // pointer so it can be moved
  Node* tail_;
  std::unique_ptr<std::condition_variable> data_cond_;  // pointer so it can be moved
};

template <class T>
ThreadSafeQueue<T>::ThreadSafeQueue(ThreadSafeQueue&& other) noexcept
    : size_(std::move(other.size_)),
      head_mutex_(std::move(other.head_mutex_)),
      head_(std::move(other.head_)),
      tail_mutex_(std::move(other.tail_mutex_)),
      tail_(other.tail_),
      data_cond_(std::move(other.data_cond_)) {
}

template <class T>
ThreadSafeQueue<T>& ThreadSafeQueue<T>::operator=(ThreadSafeQueue<T>&& other) noexcept {
  Clear(); // delete all data currently in queue safely
  size_ = std::move(other.size_);
  head_mutex_ = std::move(other.head_mutex_);
  head_ = std::move(other.head_);
  tail_mutex_ = std::move(other.tail_mutex_);
  tail_ = other.tail_;
  data_cond_ = std::move(other.data_cond_);
  return *this;
}

template <class T>
void ThreadSafeQueue<T>::Push(T new_value) {
  std::shared_ptr<T> new_data(std::make_shared<T>(std::move(new_value)));
  std::unique_ptr<Node> new_tail_node(std::make_unique<Node>());
  {
    std::lock_guard<std::mutex> tail_lock(*tail_mutex_);  // lock the tail
    tail_->data = new_data;                               // assign data into current tail

    // assign new tail
    Node* const new_tail = new_tail_node.get();
    tail_->next = std::move(new_tail_node);
    tail_ = new_tail;
  }
  ++(*size_);
  data_cond_->notify_one();  // wake up 1 thread to process the new data
}

template <class T>
bool ThreadSafeQueue<T>::TryPop(T& value) {
  std::unique_ptr<Node> const old_head = TryPopHead(value);
  return static_cast<bool>(old_head);
}

template <class T>
bool ThreadSafeQueue<T>::IsEmpty() {
  std::lock_guard<std::mutex> head_lock(*head_mutex_);
  return (head_.get() == GetTail());
}

template <class T>
void ThreadSafeQueue<T>::Clear() {
  std::lock_guard<std::mutex> head_lock(*head_mutex_);
  while (!IsEmpty()) {
    PopHead();
  }
}

}  // namespace altro
