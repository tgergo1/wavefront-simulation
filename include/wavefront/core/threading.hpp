#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

namespace wavefront {

inline std::size_t normalized_thread_count(std::size_t requested) {
  const std::size_t hw = std::max<std::size_t>(1, std::thread::hardware_concurrency());
  if (requested == 0) {
    return hw;
  }
  return std::min(requested, hw);
}

template <typename Func>
void deterministic_parallel_for(
    std::size_t total_items,
    std::size_t requested_threads,
    bool deterministic,
    Func&& fn) {
  if (total_items == 0) {
    return;
  }

  std::size_t thread_count = deterministic ? normalized_thread_count(requested_threads) : requested_threads;
  if (thread_count == 0) {
    thread_count = normalized_thread_count(0);
  }
  thread_count = std::max<std::size_t>(1, std::min(thread_count, total_items));

  if (thread_count == 1) {
    fn(0, total_items);
    return;
  }

  const std::size_t block = total_items / thread_count;
  const std::size_t remainder = total_items % thread_count;

  std::vector<std::thread> workers;
  workers.reserve(thread_count - 1);

  std::size_t begin = 0;
  for (std::size_t i = 0; i < thread_count; ++i) {
    const std::size_t length = block + (i < remainder ? 1 : 0);
    const std::size_t end = begin + length;

    if (i + 1 == thread_count) {
      fn(begin, end);
    } else {
      workers.emplace_back([begin, end, &fn]() { fn(begin, end); });
    }

    begin = end;
  }

  for (auto& worker : workers) {
    worker.join();
  }
}

}  // namespace wavefront
