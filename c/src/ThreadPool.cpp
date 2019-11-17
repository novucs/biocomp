#include "ThreadPool.h"
#include <functional>
#include <mutex>

ThreadPool::ThreadPool() {
    int thread_count = std::thread::hardware_concurrency();
    for (int i = 0; i < thread_count; i++) {
        threads.push_back(std::thread(std::bind(&ThreadPool::loop, this)));
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mutex);
        running = false;
    }

    for (std::thread &thread : threads) {
        thread.join();
    }
}

void ThreadPool::loop() {
    std::function<void()> job;
    bool found_job;

    while (true) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (!running) {
                break;
            }

            found_job = !jobs.empty();

            if (found_job) {
                job = jobs.front();
                jobs.pop();
            }
        }

        if (found_job) {
            job();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void ThreadPool::submit(std::function<void()> job) {
    std::unique_lock<std::mutex> lock(mutex);
    jobs.push(job);
}
