/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "utils.hh"
#define BOOST_TEST_MODULE ArrayMDDevice

#include <ArrayMD.hh>

#include <deal.II/base/cuda_size.h>

#include "main.cc"

template <int dim_0, int dim_1>
__global__ void
fill_array2d(adamantine::Array2D<dim_0, dim_1, dealii::MemorySpace::CUDA>
                 array2d_template,
             adamantine::Array2D<-1, dim_1, dealii::MemorySpace::CUDA>
                 array2d_partial_template)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < dim_0 * dim_1)
  {
    int i = id / dim_1;
    int j = id - i * dim_1;
    array2d_template(i, j) = id;
    array2d_partial_template(i, j) = id;
  }
}

template <int dim_0, int dim_1>
__global__ void
check_array2d(adamantine::Array2D<dim_0, dim_1, dealii::MemorySpace::CUDA>
                  array2d_template,
              adamantine::Array2D<-1, dim_1, dealii::MemorySpace::CUDA>
                  array2d_partial_template,
              int *n_errors)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < dim_0 * dim_1)
  {
    int i = id / dim_1;
    int j = id - i * dim_1;
    if (array2d_template(i, j) != id)
      atomicAdd(&n_errors[0], 1);
    if (array2d_partial_template(i, j) != id)
      atomicAdd(&n_errors[1], 1);
  }
}

template <int dim_0, int dim_1, int dim_2, int dim_3>
__global__ void fill_array4d(
    adamantine::Array4D<dim_0, dim_1, dim_2, dim_3, dealii::MemorySpace::CUDA>
        array4d_template,
    adamantine::Array4D<dim_0, dim_1, dim_2, -1, dealii::MemorySpace::CUDA>
        array4d_partial_template_1,
    adamantine::Array4D<-1, dim_1, dim_2, -1, dealii::MemorySpace::CUDA>
        array4d_partial_template_2)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < dim_0 * dim_1 * dim_2 * dim_3)
  {
    int i = id / (dim_1 * dim_2 * dim_3);
    int j = (id - i * (dim_1 * dim_2 * dim_3)) / (dim_2 * dim_3);
    int k = (id - i * (dim_1 * dim_2 * dim_3) - j * (dim_2 * dim_3)) / dim_3;
    int l = id - i * (dim_1 * dim_2 * dim_3) - j * (dim_2 * dim_3) - k * dim_3;
    array4d_template(i, j, k, l) = id;
    array4d_partial_template_1(i, j, k, l) = id;
    array4d_partial_template_2(i, j, k, l) = id;
  }
}

template <int dim_0, int dim_1, int dim_2, int dim_3>
__global__ void check_array4d(
    adamantine::Array4D<dim_0, dim_1, dim_2, dim_3, dealii::MemorySpace::CUDA>
        array4d_template,
    adamantine::Array4D<dim_0, dim_1, dim_2, -1, dealii::MemorySpace::CUDA>
        array4d_partial_template_1,
    adamantine::Array4D<-1, dim_1, dim_2, -1, dealii::MemorySpace::CUDA>
        array4d_partial_template_2,
    int *n_errors)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < dim_0 * dim_1 * dim_2 * dim_3)
  {
    int i = id / (dim_1 * dim_2 * dim_3);
    int j = (id - i * (dim_1 * dim_2 * dim_3)) / (dim_2 * dim_3);
    int k = (id - i * (dim_1 * dim_2 * dim_3) - j * (dim_2 * dim_3)) / dim_3;
    int l = id - i * (dim_1 * dim_2 * dim_3) - j * (dim_2 * dim_3) - k * dim_3;
    if (array4d_template(i, j, k, l) != id)
      atomicAdd(&n_errors[0], 1);
    if (array4d_partial_template_1(i, j, k, l) != id)
      atomicAdd(&n_errors[1], 1);
    if (array4d_partial_template_2(i, j, k, l) != id)
      atomicAdd(&n_errors[2], 1);
  }
}

template <int dim_0, int dim_1, int dim_2, int dim_3, int dim_4>
__global__ void fill_array5d(
    adamantine::Array5D<dim_0, dim_1, dim_2, dim_3, dim_4,
                        dealii::MemorySpace::CUDA>
        array5d_template,
    adamantine::Array5D<-1, dim_1, dim_2, -1, dim_4, dealii::MemorySpace::CUDA>
        array5d_partial_template)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < dim_0 * dim_1 * dim_2 * dim_3 * dim_4)
  {
    int i = id / (dim_1 * dim_2 * dim_3 * dim_4);
    int j =
        (id - i * (dim_1 * dim_2 * dim_3 * dim_4)) / (dim_2 * dim_3 * dim_4);
    int k = (id - i * (dim_1 * dim_2 * dim_3 * dim_4) -
             j * (dim_2 * dim_3 * dim_4)) /
            (dim_3 * dim_4);
    int l = (id - i * (dim_1 * dim_2 * dim_3 * dim_4) -
             j * (dim_2 * dim_3 * dim_4) - k * (dim_3 * dim_4)) /
            dim_4;
    int m = id - i * (dim_1 * dim_2 * dim_3 * dim_4) -
            j * (dim_2 * dim_3 * dim_4) - k * (dim_3 * dim_4) - l * dim_4;
    array5d_template(i, j, k, l, m) = id;
    array5d_partial_template(i, j, k, l, m) = id;
  }
}

template <int dim_0, int dim_1, int dim_2, int dim_3, int dim_4>
__global__ void check_array5d(
    adamantine::Array5D<dim_0, dim_1, dim_2, dim_3, dim_4,
                        dealii::MemorySpace::CUDA>
        array5d_template,
    adamantine::Array5D<-1, dim_1, dim_2, -1, dim_4, dealii::MemorySpace::CUDA>
        array5d_partial_template,
    int *n_errors)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < dim_0 * dim_1 * dim_2 * dim_3 * dim_4)
  {
    int i = id / (dim_1 * dim_2 * dim_3 * dim_4);
    int j =
        (id - i * (dim_1 * dim_2 * dim_3 * dim_4)) / (dim_2 * dim_3 * dim_4);
    int k = (id - i * (dim_1 * dim_2 * dim_3 * dim_4) -
             j * (dim_2 * dim_3 * dim_4)) /
            (dim_3 * dim_4);
    int l = (id - i * (dim_1 * dim_2 * dim_3 * dim_4) -
             j * (dim_2 * dim_3 * dim_4) - k * (dim_3 * dim_4)) /
            dim_4;
    int m = id - i * (dim_1 * dim_2 * dim_3 * dim_4) -
            j * (dim_2 * dim_3 * dim_4) - k * (dim_3 * dim_4) - l * dim_4;
    if (array5d_template(i, j, k, l, m) != id)
      atomicAdd(&n_errors[0], 1);
    if (array5d_partial_template(i, j, k, l, m) != id)
      atomicAdd(&n_errors[1], 1);
  }
}

BOOST_AUTO_TEST_CASE(array2d)
{
  int constexpr dim_0 = 2;
  int constexpr dim_1 = 3;
  int constexpr size = dim_0 * dim_1;

  adamantine::Array2D<dim_0, dim_1, dealii::MemorySpace::CUDA> array2d_template;
  adamantine::Array2D<-1, dim_1, dealii::MemorySpace::CUDA>
      array2d_partial_template(dim_0);

  const int n_blocks = 1 + size / dealii::CUDAWrappers::block_size;
  fill_array2d<<<n_blocks, dealii::CUDAWrappers::block_size>>>(
      array2d_template, array2d_partial_template);

  int *n_errors_dev;
  dealii::Utilities::CUDA::malloc(n_errors_dev, 2);
  adamantine::Memory<int, dealii::MemorySpace::CUDA>::set_zero(n_errors_dev, 2);

  check_array2d<<<n_blocks, dealii::CUDAWrappers::block_size>>>(
      array2d_template, array2d_partial_template, n_errors_dev);

  std::vector<int> n_errors_host(2);
  dealii::Utilities::CUDA::copy_to_host(n_errors_dev, n_errors_host);

  for (unsigned int i = 0; i < 2; ++i)
    BOOST_CHECK(n_errors_host[i] == 0);

  dealii::Utilities::CUDA::free(n_errors_dev);
}

BOOST_AUTO_TEST_CASE(array4d)
{
  int constexpr dim_0 = 2;
  int constexpr dim_1 = 3;
  int constexpr dim_2 = 4;
  int constexpr dim_3 = 5;
  int constexpr size = dim_0 * dim_1 * dim_2 * dim_3;

  adamantine::Array4D<dim_0, dim_1, dim_2, dim_3, dealii::MemorySpace::CUDA>
      array4d_template;
  adamantine::Array4D<dim_0, dim_1, dim_2, -1, dealii::MemorySpace::CUDA>
      array4d_partial_template_1(dim_3);
  adamantine::Array4D<-1, dim_1, dim_2, -1, dealii::MemorySpace::CUDA>
      array4d_partial_template_2(dim_0, dim_3);

  const int n_blocks = 1 + size / dealii::CUDAWrappers::block_size;
  fill_array4d<<<n_blocks, dealii::CUDAWrappers::block_size>>>(
      array4d_template, array4d_partial_template_1, array4d_partial_template_2);

  int *n_errors_dev;
  dealii::Utilities::CUDA::malloc(n_errors_dev, 3);
  adamantine::Memory<int, dealii::MemorySpace::CUDA>::set_zero(n_errors_dev, 3);

  check_array4d<<<n_blocks, dealii::CUDAWrappers::block_size>>>(
      array4d_template, array4d_partial_template_1, array4d_partial_template_2,
      n_errors_dev);

  std::vector<int> n_errors_host(3);
  dealii::Utilities::CUDA::copy_to_host(n_errors_dev, n_errors_host);

  for (unsigned int i = 0; i < 3; ++i)
    BOOST_CHECK(n_errors_host[i] == 0);

  dealii::Utilities::CUDA::free(n_errors_dev);
}

BOOST_AUTO_TEST_CASE(array5d)
{
  int constexpr dim_0 = 2;
  int constexpr dim_1 = 3;
  int constexpr dim_2 = 4;
  int constexpr dim_3 = 5;
  int constexpr dim_4 = 6;
  int constexpr size = dim_0 * dim_1 * dim_2 * dim_3 * dim_4;

  adamantine::Array5D<dim_0, dim_1, dim_2, dim_3, dim_4,
                      dealii::MemorySpace::CUDA>
      array5d_template;
  adamantine::Array5D<-1, dim_1, dim_2, -1, dim_4, dealii::MemorySpace::CUDA>
      array5d_partial_template(dim_0, dim_3);

  const int n_blocks = 1 + size / dealii::CUDAWrappers::block_size;
  fill_array5d<<<n_blocks, dealii::CUDAWrappers::block_size>>>(
      array5d_template, array5d_partial_template);

  int *n_errors_dev;
  dealii::Utilities::CUDA::malloc(n_errors_dev, 2);
  adamantine::Memory<int, dealii::MemorySpace::CUDA>::set_zero(n_errors_dev, 2);

  check_array5d<<<n_blocks, dealii::CUDAWrappers::block_size>>>(
      array5d_template, array5d_partial_template, n_errors_dev);

  std::vector<int> n_errors_host(2);
  dealii::Utilities::CUDA::copy_to_host(n_errors_dev, n_errors_host);

  for (unsigned int i = 0; i < 2; ++i)
    BOOST_CHECK(n_errors_host[i] == 0);

  dealii::Utilities::CUDA::free(n_errors_dev);
}
