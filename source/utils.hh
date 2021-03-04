/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef UTILS_HH
#define UTILS_HH

#include <cassert>
#include <exception>
#include <stdexcept>
#include <string>

namespace adamantine
{

inline void ASSERT(bool cond, std::string const &message)
{
  assert((cond) && (message.c_str()));
  (void)cond;
  (void)message;
}

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
