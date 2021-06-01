/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef UTILS_HH
#define UTILS_HH

#include <deal.II/base/cuda.h>
#include <deal.II/base/exceptions.h>

#include <cassert>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

namespace adamantine
{
#ifdef __CUDACC__
#define ADAMANTINE_HOST_DEV __host__ __device__
#else
#define ADAMANTINE_HOST_DEV
#endif

template <typename Number, typename MemorySpaceType>
struct Memory
{
  static Number *allocate_data(std::size_t const size);

  static void delete_data(Number *data_ptr) noexcept;

  static void set_zero(Number *data_ptr, std::size_t const size);
};

template <typename Number>
struct Memory<Number, dealii::MemorySpace::Host>
{
  static Number *allocate_data(std::size_t const size)
  {
    Number *data_ptr = new Number[size];
    return data_ptr;
  }

  static void delete_data(Number *data_ptr) noexcept { delete[] data_ptr; }

  static void set_zero(Number *data_ptr, std::size_t const size)
  {
    std::memset(data_ptr, 0, size * sizeof(Number));
  }
};

#ifdef __CUDACC__
template <typename Number>
struct Memory<Number, dealii::MemorySpace::CUDA>
{
  static Number *allocate_data(std::size_t const size)
  {
    Number *data_ptr;
    dealii::Utilities::CUDA::malloc(data_ptr, size);
    return data_ptr;
  }

  static void delete_data(Number *data_ptr) noexcept
  {
    cudaError_t const error_code = cudaFree(data_ptr);
    AssertNothrowCuda(error_code);
  }

  static void set_zero(Number *data_ptr, std::size_t const size)
  {
    cudaError_t const error_code =
        cudaMemset(data_ptr, 0, size * sizeof(Number));
    AssertCuda(error_code);
  }
};
#endif

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
