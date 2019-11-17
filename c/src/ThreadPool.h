#ifndef C_THREADPOOL_H
#define C_THREADPOOL_H

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>

class ThreadPool {
private:
    std::mutex mutex;
    std::vector<std::thread> threads = std::vector<std::thread>();
    std::queue<std::function<void()>> jobs = std::queue<std::function<void()>>();
    bool running = true;

    void loop();

public:
    ThreadPool();

    virtual ~ThreadPool();

    void submit(std::function<void()> job);
};

#endif //C_THREADPOOL_H
