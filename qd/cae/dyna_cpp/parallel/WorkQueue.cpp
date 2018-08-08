
#include <dyna_cpp/parallel/WorkQueue.hpp>

#ifdef QD_DEBUG
#include <iostream>
#endif

namespace qd {

/**  Constructors a new work queue object
 *
 * @param num_workers : number of workers. If <1 all cores are used.
 */
WorkQueue::WorkQueue() {}

/** Will abort all pending jobs and run any in-progress jobs to completion
 * upon destruction.
 *
 */
WorkQueue::~WorkQueue()
{
#ifdef QD_DEBUG
  std::cout << "Killing " << m_workers.size() << " unterminated threads.\n";
#endif
  abort();
}

/** Reset the WorkQueue (required for restart)
 *
 */
void WorkQueue::reset() {
	m_exit = false;
	m_finish_work = true;
}

/** Initialize a certain amount of workers
 *
 * @param num_workers : number of workers to spawn
 *
 * By default one more thread than cores is allocated to
 * ensure a high workload. If there are already enough workers running,
 * nothing happens.
 */
void
WorkQueue::init_workers(size_t num_workers)
{
  std::lock_guard<std::mutex> lg(m_mutex);

  if (num_workers == 0) {
    return;
    // num_workers = std::thread::hardware_concurrency() + 1;
  }

  reset();

  for (size_t iThread = m_workers.size(); iThread < num_workers; ++iThread)
    m_workers.emplace_back(std::thread(&WorkQueue::do_work, this));

#ifdef QD_DEBUG
  std::cout << "Initialized " << m_workers.size() << " threads.\n";
#endif
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
#ifdef QD_DEBUG
  std::cout << "Joining " << m_workers.size() << " threads.\n";
#endif
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