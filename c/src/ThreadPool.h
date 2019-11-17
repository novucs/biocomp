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

    template<typename T>
    std::vector<T> map(std::vector<std::function<T()>> submitted) {
        std::vector<T> results;
        results.reserve(submitted.size());
        std::mutex map_mutex;

        for (int i = 0; i < submitted.size(); i++) {
            std::function<T()> function = submitted.at(i);
            submit([&function, &map_mutex, &results, &i] {
                T result = function();
                {
                    std::unique_lock<std::mutex> lock(map_mutex);
                    results.at(i) = result;
                }
            });
        }

        return results;
    }
};

#endif //C_THREADPOOL_H
