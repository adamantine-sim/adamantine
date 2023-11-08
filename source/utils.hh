/* Copyright (c) 2016 - 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef UTILS_HH
#define UTILS_HH

#include <deal.II/base/cuda.h>
#include <deal.II/base/cuda_size.h>
#include <deal.II/base/exceptions.h>

#include <cassert>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace adamantine
{
template <typename Functor>
void for_each(dealii::MemorySpace::Host, unsigned int const size, Functor f)
{
  for (unsigned int i = 0; i < size; ++i)
    f(i);
}

#ifdef __CUDACC__
template <typename Functor>
__global__ void for_each_impl(int size, Functor f)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size)
  {
    f(i);
  }
}

template <typename Functor>
void for_each(dealii::MemorySpace::Default, unsigned int const size,
              Functor const &f)
{
  const int n_blocks = 1 + size / dealii::CUDAWrappers::block_size;
  for_each_impl<<<n_blocks, dealii::CUDAWrappers::block_size>>>(size, f);
  AssertCudaKernel();
}
#endif

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
