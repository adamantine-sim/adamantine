/* Copyright (c) 2017, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef TIMER_H
#define TIMER_H

#include <boost/chrono/include.hpp>
#include <boost/mpi.hpp>
#include <string>

namespace adamantine
{
/**
 * This class measures the time spend in a given section by the rank 0 process.
 * This class does not use any MPI_Barrier to synchronize the timer among all
 * the processors.
 */
class Timer
{
public:
  /**
   * Constructor. The string @p section is used when the timing is output.
   */
  Timer(boost::mpi::communicator communicator, std::string const &section);

  /**
   * Start the clock.
   */
  void start();

  /**
   * Stop the clock. If the clock has already been started and stopped before
   * the new duration will be added to the previous one.
   */
  void stop();

  /**
   * Reset to zero the store duration.
   */
  void reset();

  /**
   * Print the name of the section and the elapsed time.
   */
  void print();

  /**
   * Return the current elapsed time.
   */
  boost::chrono::process_real_cpu_clock::duration get_elapsed_time();

private:
  boost::mpi::communicator _communicator;
  std::string _section;
  boost::chrono::process_cpu_clock _clock;
  boost::chrono::process_cpu_clock::time_point _t_start;
  /**
   * Store the elapsed time in milliseconds nds
   */
  boost::chrono::process_cpu_clock::duration _elapsed_time;
};
}
#endif
