/* Copyright (c) 2016 - 2021, the adamantine authors.
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

#ifdef ADAMANTINE_WITH_DEALII_WEAK_FORMS
#include <weak_forms/symbolic_decorations.h>
#endif

#include <cassert>
#include <cstring>
#include <exception>
#include <execution>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace adamantine
{
#ifdef __CUDACC__
#define ADAMANTINE_HOST_DEV __host__ __device__
#else
#define ADAMANTINE_HOST_DEV
#endif

template <typename Number>
inline void deep_copy(Number *output, dealii::MemorySpace::Host const &,
                      Number const *input, dealii::MemorySpace::Host const &,
                      unsigned int size)
{
  std::memcpy(output, input, size * sizeof(Number));
}

#ifdef __CUDACC__
template <typename Number>
inline void deep_copy(Number *output, dealii::MemorySpace::Host const &,
                      Number const *input, dealii::MemorySpace::CUDA const &,
                      unsigned int size)
{
  cudaError_t const error_code =
      cudaMemcpy((void *)output, (void const *)input, size * sizeof(Number),
                 cudaMemcpyDeviceToHost);
  AssertCuda(error_code);
}

template <typename Number>
inline void deep_copy(Number *output, dealii::MemorySpace::CUDA const &,
                      Number const *input, dealii::MemorySpace::Host const &,
                      unsigned int size)
{
  cudaError_t const error_code =
      cudaMemcpy((void *)output, (void const *)input, size * sizeof(Number),
                 cudaMemcpyHostToDevice);
  AssertCuda(error_code);
}

template <typename Number>
inline void deep_copy(Number *output, dealii::MemorySpace::CUDA const &,
                      Number const *input, dealii::MemorySpace::CUDA const &,
                      unsigned int size)
{
  cudaError_t const error_code =
      cudaMemcpy((void *)output, (void const *)input, size * sizeof(Number),
                 cudaMemcpyDeviceToDevice);
  AssertCuda(error_code);
}
#endif

template <typename ArrayType1, typename ArrayType2>
void deep_copy(ArrayType1 &output, ArrayType2 const &input)
{
  deep_copy(output.data(), typename ArrayType1::memory_space{}, input.data(),
            typename ArrayType2::memory_space{}, input.size());
}

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

template <typename Functor>
void for_each(dealii::MemorySpace::Host, unsigned int const size, Functor f)
{
  for (unsigned int i = 0; i < size; ++i)
    f(i);
}

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
void for_each(dealii::MemorySpace::CUDA, unsigned int const size,
              Functor const &f)
{
  const int n_blocks = 1 + size / dealii::CUDAWrappers::block_size;
  for_each_impl<<<n_blocks, dealii::CUDAWrappers::block_size>>>(size, f);
  AssertCudaKernel();
}
#endif

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

#ifdef ADAMANTINE_WITH_DEALII_WEAK_FORMS
/**
 * Customize latex output created by dealii-weak_forms.
 */
inline dealiiWeakForms::WeakForms::Decorations::Discretization
symbolic_names(std::string const &test_function,
               std::string const &trial_solution = "\\tilde")
{
  std::string const solution_field = "\\varphi";
  std::string const shape_function = "N";
  std::string const dof_value = "c";
  std::string const JxW = "\\int";

  return dealiiWeakForms::WeakForms::Decorations::Discretization(
      solution_field, test_function, trial_solution, shape_function, dof_value,
      JxW);
}
#endif

} // namespace adamantine

#endif
