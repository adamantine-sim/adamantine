/* SPDX-FileCopyrightText: Copyright (c) 2017 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef TIMER_H
#define TIMER_H

#include <boost/chrono/include.hpp>

#include <string>

#include <mpi.h>

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
   * Default constructor.
   */
  Timer() = default;

  /**
   * Constructor. The string @p section is used when the timing is output.
   */
  Timer(MPI_Comm communicator, std::string const &section);

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
  MPI_Comm _communicator;
  std::string _section;
  boost::chrono::process_cpu_clock _clock;
  boost::chrono::process_cpu_clock::time_point _t_start;
  /**
   * Store the elapsed time in milliseconds nds
   */
  boost::chrono::process_cpu_clock::duration _elapsed_time;
};
} // namespace adamantine
#endif
