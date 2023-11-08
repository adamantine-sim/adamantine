/* Copyright (c) 2016 - 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef UTILS_HH
#define UTILS_HH

#include <deal.II/base/exceptions.h>

#include <cassert>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace adamantine
{
/**
 * Wait for the file to appear.
 */
inline void wait_for_file(std::string const &filename,
                          std::string const &message)
{
  unsigned int counter = 1;
  while (!std::filesystem::exists(filename))
  {
    // Spin loop waiting for the file to appear (message printed if counter
    // overflows)
    if (counter == 0)
      std::cout << message << std::endl;
    ++counter;
  }
}

#define ASSERT(condition, message) assert((condition) && (message))

inline void ASSERT_THROW(bool cond, std::string const &message)
{
  if (cond == false)
    throw std::runtime_error(message);
}

// ----------- Custom Exceptions --------------//
class NotImplementedExc : public std::exception
{
  virtual const char *what() const throw() override
  {
    return "The function is not implemented";
  }
};

inline void ASSERT_THROW_NOT_IMPLEMENTED()
{
  NotImplementedExc exception;
  throw exception;
}

} // namespace adamantine

#endif
