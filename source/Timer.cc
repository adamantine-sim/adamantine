/* SPDX-FileCopyrightText: Copyright (c) 2017 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <Timer.hh>

#include <iostream>

namespace adamantine
{
Timer::Timer(MPI_Comm communicator, std::string const &section)
    : _communicator(communicator), _section(section), _clock(), _t_start(),
      _elapsed_time(boost::chrono::milliseconds(0))
{
}

void Timer::start() { _t_start = _clock.now(); }

void Timer::stop() { _elapsed_time += _clock.now() - _t_start; }

void Timer::reset() { _elapsed_time = boost::chrono::milliseconds(0); }

void Timer::print()
{
  int rank = -1;
  MPI_Comm_rank(_communicator, &rank);
  if (rank == 0)
  {
    boost::chrono::milliseconds ms =
        boost::chrono::duration_cast<boost::chrono::milliseconds>(
            _elapsed_time);
    std::cout << "Time elapsed in " + _section + ": " << ms << std::endl;
  }
}

boost::chrono::process_real_cpu_clock::duration Timer::get_elapsed_time()
{
  return _elapsed_time;
}
} // namespace adamantine
