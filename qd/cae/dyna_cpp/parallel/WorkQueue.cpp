
#include <dyna_cpp/parallel/WorkQueue.hpp>

namespace qd {

/**  Constructors a new work queue object
 *
 * @param num_workers : number of workers. If <1 all cores are used.
 */
WorkQueue::WorkQueue(int64_t num_workers)
{
  if (num_workers < 1) {
    num_workers = std::thread::hardware_concurrency() + 1;
  }
  while (num_workers--) {
    m_workers.emplace_back(std::thread(&WorkQueue::do_work, this));
  }
}

/** Will abort all pending jobs and run any in-progress jobs to completion
 * upon destruction.
 *
 */
WorkQueue::~WorkQueue()
{
  abort();
}

/** worker thread function, for picking jobs.
 *
 */
void
WorkQueue::do_work()
{
  std::unique_lock<std::mutex> ul(m_mutex);
  while (!m_exit || (m_finish_work && !m_work.empty())) {

    if (!m_work.empty()) {
      std::function<void()> work(std::move(m_work.front()));
      m_work.pop_front();
      ul.unlock();
      work();
      ul.lock();
    } else {
      m_signal.wait(ul);
    }
  }
}

/** Waits for all threads to finish
 *
 */
void
WorkQueue::join_all()
{
  for (auto& thread : m_workers) {
    thread.join();
  }
  m_workers.clear();
}

/** Stops queue and jobs
 *
 * Stops work queue and finishes jobs currently being executed.
 * Queued jobs that have not begun execution will have their promises
 * broken.
 *
 */
void
WorkQueue::abort()
{
  m_exit = true;
  m_finish_work = false;
  m_signal.notify_all();
  join_all();
  {
    std::lock_guard<std::mutex> lg(m_mutex);
    m_work.clear();
  }
}

/** Stops new work from being submitted to this work queue
 *
 */
void
WorkQueue::stop()
{
  m_exit = true;
  m_finish_work = true;
  m_signal.notify_all();
}

/**  Wait for completion of all running jobs. No more work will done.
 *
 */
void
WorkQueue::wait_for_completion()
{
  stop();
  join_all();
  assert(m_work.empty());
}

} // namespace:qd