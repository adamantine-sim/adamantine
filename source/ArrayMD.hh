/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MDARRAY_HH
#define MDARRAY_HH

#include <utils.hh>

// This file contains multiple arrays of different dimensions. The size of the
// arrays can be set at compile-time or at runtime, similar to a Kokkos::View.
// Note that the use case of these arrays is very different than Kokkos::View.
// The goal here is to store data on the host or the device but not to loop over
// the data store. This means that the memory layout is not as important.
// Another difference with Kokkos::View is that copies of the arrays are not
// owning, i.e. the lifetime of the underlying data is determined by the
// original object.
namespace adamantine
{
/**
 * Two-dimensional array with both dimensions set at compile time.
 */
template <int dim_0, int dim_1, typename MemorySpaceType>
class Array2D
{
public:
  static_assert(dim_0 > 0, "dim_0 should be greater than 0.");
  static_assert(dim_1 > 0, "dim_1 should be greater than 0.");

  /**
   * Default constructor.
   */
  Array2D();

  /**
   * Copy constructor. The copy is non-owning, i.e., the lifetime of the
   * underlying data is determined by the original object.
   */
  Array2D(Array2D<dim_0, dim_1, MemorySpaceType> const &other);

  /**
   * Destructor.
   */
  ~Array2D();

  /**
   * Set the data to zero.
   */
  void set_zero();

  /**
   * Dimension `i` of the array.
   */
  ADAMANTINE_HOST_DEV unsigned int extent(unsigned int i) const;

  /**
   * Access element `(i,j)` of the array.
   */
  ADAMANTINE_HOST_DEV double &operator()(unsigned int i, unsigned int j);

  /**
   * Access element `(i,j)` of the array.
   */
  ADAMANTINE_HOST_DEV double const &operator()(unsigned int i,
                                               unsigned int j) const;

private:
  /**
   * Owning flag true if owning and false otherwise.
   */
  bool _owning = true;
  /**
   * Pointer to data.
   */
  double *_values = nullptr;
};

template <int dim_0, int dim_1, typename MemorySpaceType>
Array2D<dim_0, dim_1, MemorySpaceType>::Array2D()
    : _values(Memory<double, MemorySpaceType>::allocate_data(dim_0 * dim_1))
{
}

template <int dim_0, int dim_1, typename MemorySpaceType>
Array2D<dim_0, dim_1, MemorySpaceType>::Array2D(
    Array2D<dim_0, dim_1, MemorySpaceType> const &other)
    : _owning(false), _values(other._values)
{
}

template <int dim_0, int dim_1, typename MemorySpaceType>
Array2D<dim_0, dim_1, MemorySpaceType>::~Array2D()
{
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }
  _values = nullptr;
}

template <int dim_0, int dim_1, typename MemorySpaceType>
void Array2D<dim_0, dim_1, MemorySpaceType>::set_zero()
{

  Memory<double, MemorySpaceType>::set_zero(_values, dim_0 * dim_1);
}

template <int dim_0, int dim_1, typename MemorySpaceType>
ADAMANTINE_HOST_DEV unsigned int
Array2D<dim_0, dim_1, MemorySpaceType>::extent(unsigned int i) const
{
  if (i == 0)
    return dim_0;
  else if (i == 1)
    return dim_1;
  else
    return 0;
}

template <int dim_0, int dim_1, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double &
Array2D<dim_0, dim_1, MemorySpaceType>::operator()(unsigned int i,
                                                   unsigned int j)
{
  return _values[i * dim_1 + j];
}

template <int dim_0, int dim_1, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double const &
Array2D<dim_0, dim_1, MemorySpaceType>::operator()(unsigned int i,
                                                   unsigned int j) const
{
  return _values[i * dim_1 + j];
}

/**
 * Two-dimensional array with the first dimension set at runtime and the second
 * dimension set at compile-time.
 */
template <int dim_1, typename MemorySpaceType>
class Array2D<-1, dim_1, MemorySpaceType>
{
public:
  static_assert(dim_1 > 0, "dim_1 should be greater than 0.");

  /**
   * Default constructor.
   */
  Array2D() = default;

  /**
   * Constructor. Set the first dimension to `size_0`.
   */
  Array2D(unsigned int size_0);

  /**
   * Copy constructor. The copy is non-owning, i.e., the lifetime of the
   * underlying data is determined by the original object.
   */
  Array2D(Array2D<-1, dim_1, MemorySpaceType> const &other);

  /**
   * Destructor.
   */
  ~Array2D();

  /**
   * Reinitialize the data using `size_0` for the first dimension. The initial
   * data is cleared.
   */
  void reinit(unsigned int size_0);

  /**
   * Set the data to zero.
   */
  void set_zero();

  /**
   * Dimension `i` of the array.
   */
  ADAMANTINE_HOST_DEV unsigned int extent(unsigned int i) const;

  /**
   * Access element `(i,j)` of the array.
   */
  ADAMANTINE_HOST_DEV double &operator()(unsigned int i, unsigned int j);

  /**
   * Access element `(i,j)` of the array.
   */
  ADAMANTINE_HOST_DEV double const &operator()(unsigned int i,
                                               unsigned int j) const;

private:
  /**
   * Owning flag true if owning and false otherwise.
   */
  bool _owning = true;
  /**
   * Dimension zero of the array.
   */
  unsigned int _dim_0 = 0;
  /**
   * Pointer to data.
   */
  double *_values = nullptr;
};

template <int dim_1, typename MemorySpaceType>
Array2D<-1, dim_1, MemorySpaceType>::Array2D(unsigned int size_0)
    : _dim_0(size_0),
      _values(Memory<double, MemorySpaceType>::allocate_data(_dim_0 * dim_1))
{
}

template <int dim_1, typename MemorySpaceType>
Array2D<-1, dim_1, MemorySpaceType>::Array2D(
    Array2D<-1, dim_1, MemorySpaceType> const &other)
    : _owning(false), _dim_0(other._dim_0), _values(other._values)
{
}

template <int dim_1, typename MemorySpaceType>
Array2D<-1, dim_1, MemorySpaceType>::~Array2D()
{
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }
  _values = nullptr;
}

template <int dim_1, typename MemorySpaceType>
void Array2D<-1, dim_1, MemorySpaceType>::reinit(unsigned int size_0)
{
  // Free memory if necessary
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }

  _dim_0 = size_0;
  _values = Memory<double, MemorySpaceType>::allocate_data(_dim_0 * dim_1);
}

template <int dim_1, typename MemorySpaceType>
void Array2D<-1, dim_1, MemorySpaceType>::set_zero()
{

  Memory<double, MemorySpaceType>::set_zero(_values, _dim_0 * dim_1);
}

template <int dim_1, typename MemorySpaceType>
ADAMANTINE_HOST_DEV unsigned int
Array2D<-1, dim_1, MemorySpaceType>::extent(unsigned int i) const
{
  if (i == 0)
    return _dim_0;
  else if (i == 1)
    return dim_1;
  else
    return 0;
}

template <int dim_1, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double &
Array2D<-1, dim_1, MemorySpaceType>::operator()(unsigned int i, unsigned int j)
{
  return _values[i * dim_1 + j];
}

template <int dim_1, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double const &
Array2D<-1, dim_1, MemorySpaceType>::operator()(unsigned int i,
                                                unsigned int j) const
{
  return _values[i * dim_1 + j];
}

/**
 * Four-dimensional array with all the dimension set at compile-time.
 */
template <int dim_0, int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
class Array4D
{
public:
  static_assert(dim_0 > 0, "dim_0 should be greater than 0.");
  static_assert(dim_1 > 0, "dim_1 should be greater than 0.");
  static_assert(dim_2 > 0, "dim_2 should be greater than 0.");
  static_assert(dim_3 > 0, "dim_3 should be greater than 0.");

  /**
   * Default constructor.
   */
  Array4D();

  /**
   * Copy constructor. The copy is non-owning, i.e., the lifetime of the
   * underlying data is determined by the original object.
   */
  Array4D(Array4D<dim_0, dim_1, dim_2, dim_3, MemorySpaceType> const &other);

  /**
   * Destructor.
   */
  ~Array4D();

  /**
   * Set the data to zero.
   */
  void set_zero();

  /**
   * Dimension `i` of the array.
   */
  ADAMANTINE_HOST_DEV unsigned int extent(unsigned int i) const;

  /**
   * Access element `(i,j,k,l)` of the array.
   */
  ADAMANTINE_HOST_DEV double &operator()(unsigned int i, unsigned int j,
                                         unsigned int k, unsigned int l);

  /**
   * Access element `(i,j,k,l)` of the array.
   */
  ADAMANTINE_HOST_DEV double const &operator()(unsigned int i, unsigned int j,
                                               unsigned int k,
                                               unsigned int l) const;

private:
  /**
   * Owning flag true if owning and false otherwise.
   */
  bool _owning = true;
  /**
   * Pointer to data.
   */
  double *_values = nullptr;
};

template <int dim_0, int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
Array4D<dim_0, dim_1, dim_2, dim_3, MemorySpaceType>::Array4D()
    : _values(Memory<double, MemorySpaceType>::allocate_data(dim_0 * dim_1 *
                                                             dim_2 * dim_3))
{
}

template <int dim_0, int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
Array4D<dim_0, dim_1, dim_2, dim_3, MemorySpaceType>::Array4D(
    Array4D<dim_0, dim_1, dim_2, dim_3, MemorySpaceType> const &other)
    : _owning(false), _values(other._values)
{
}

template <int dim_0, int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
Array4D<dim_0, dim_1, dim_2, dim_3, MemorySpaceType>::~Array4D()
{
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }
  _values = nullptr;
}

template <int dim_0, int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
void Array4D<dim_0, dim_1, dim_2, dim_3, MemorySpaceType>::set_zero()
{

  Memory<double, MemorySpaceType>::set_zero(_values,
                                            dim_0 * dim_1 * dim_2 * dim_3);
}

template <int dim_0, int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
ADAMANTINE_HOST_DEV unsigned int
Array4D<dim_0, dim_1, dim_2, dim_3, MemorySpaceType>::extent(
    unsigned int i) const
{
  if (i == 0)
    return dim_0;
  else if (i == 1)
    return dim_1;
  else if (i == 2)
    return dim_2;
  else if (i == 3)
    return dim_3;
  else
    return 0;
}

template <int dim_0, int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double &
Array4D<dim_0, dim_1, dim_2, dim_3, MemorySpaceType>::operator()(unsigned int i,
                                                                 unsigned int j,
                                                                 unsigned int k,
                                                                 unsigned int l)
{
  return _values[i * (dim_1 * dim_2 * dim_3) + j * (dim_2 * dim_3) + k * dim_3 +
                 l];
}

template <int dim_0, int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double const &
Array4D<dim_0, dim_1, dim_2, dim_3, MemorySpaceType>::operator()(
    unsigned int i, unsigned int j, unsigned int k, unsigned int l) const
{
  return _values[i * (dim_1 * dim_2 * dim_3) + j * (dim_2 * dim_3) + k * dim_3 +
                 l];
}

/**
 * Four-dimensional array with the first dimension set at runtime and the other
 * dimensions set at compile-time.
 */
template <int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
class Array4D<-1, dim_1, dim_2, dim_3, MemorySpaceType>
{
public:
  static_assert(dim_1 > 0, "dim_0 should be greater than 0.");
  static_assert(dim_2 > 0, "dim_1 should be greater than 0.");
  static_assert(dim_3 > 0, "dim_2 should be greater than 0.");

  /**
   * Default constructor.
   */
  Array4D() = default;

  /**
   * Constructor. Set the first dimension to `size_0`.
   */
  Array4D(unsigned int size_0);

  /**
   * Copy constructor. The copy is non-owning, i.e., the lifetime of the
   * underlying data is determined by the original object.
   */
  Array4D(Array4D<-1, dim_1, dim_2, dim_3, MemorySpaceType> const &other);

  /**
   * Destructor.
   */
  ~Array4D();

  /**
   * Reinitialize the data using `size_0` for the first dimension. The initial
   * data is cleared.
   */
  void reinit(unsigned int size_0);

  /**
   * Set the data to zero.
   */
  void set_zero();

  /**
   * Dimension `i` of the array.
   */
  ADAMANTINE_HOST_DEV unsigned int extent(unsigned int i) const;

  /**
   * Access element `(i,j,k,l)` of the array.
   */
  ADAMANTINE_HOST_DEV double &operator()(unsigned int i, unsigned int j,
                                         unsigned int k, unsigned int l);

  /**
   * Access element `(i,j,k,l)` of the array.
   */
  ADAMANTINE_HOST_DEV double const &operator()(unsigned int i, unsigned int j,
                                               unsigned int k,
                                               unsigned int l) const;

private:
  /**
   * Owning flag true if owning and false otherwise.
   */
  bool _owning = true;
  /**
   * Dimension zero of the array.
   */
  unsigned int _dim_0 = 0;
  /**
   * Pointer to data.
   */
  double *_values = nullptr;
};

template <int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
Array4D<-1, dim_1, dim_2, dim_3, MemorySpaceType>::Array4D(unsigned int size_0)
    : _dim_0(size_0), _values(Memory<double, MemorySpaceType>::allocate_data(
                          _dim_0 * dim_1 * dim_2 * dim_3))
{
}

template <int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
Array4D<-1, dim_1, dim_2, dim_3, MemorySpaceType>::Array4D(
    Array4D<-1, dim_1, dim_2, dim_3, MemorySpaceType> const &other)
    : _owning(false), _dim_0(other._dim_0), _values(other._values)
{
}

template <int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
Array4D<-1, dim_1, dim_2, dim_3, MemorySpaceType>::~Array4D()
{
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }
  _values = nullptr;
}

template <int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
void Array4D<-1, dim_1, dim_2, dim_3, MemorySpaceType>::reinit(
    unsigned int size_0)
{
  // Free memory if necessary
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }

  _dim_0 = size_0;
  _values = Memory<double, MemorySpaceType>::allocate_data(_dim_0 * dim_1 *
                                                           dim_2 * dim_3);
}

template <int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
void Array4D<-1, dim_1, dim_2, dim_3, MemorySpaceType>::set_zero()
{

  Memory<double, MemorySpaceType>::set_zero(_values,
                                            _dim_0 * dim_1 * dim_2 * dim_3);
}

template <int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
ADAMANTINE_HOST_DEV unsigned int
Array4D<-1, dim_1, dim_2, dim_3, MemorySpaceType>::extent(unsigned int i) const
{
  if (i == 0)
    return _dim_0;
  else if (i == 1)
    return dim_1;
  else if (i == 2)
    return dim_2;
  else if (i == 3)
    return dim_3;
  else
    return 0;
}

template <int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double &
Array4D<-1, dim_1, dim_2, dim_3, MemorySpaceType>::operator()(unsigned int i,
                                                              unsigned int j,
                                                              unsigned int k,
                                                              unsigned int l)
{
  return _values[i * (dim_1 * dim_2 * dim_3) + j * (dim_2 * dim_3) + k * dim_3 +
                 l];
}

template <int dim_1, int dim_2, int dim_3, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double const &
Array4D<-1, dim_1, dim_2, dim_3, MemorySpaceType>::operator()(
    unsigned int i, unsigned int j, unsigned int k, unsigned int l) const
{
  return _values[i * (dim_1 * dim_2 * dim_3) + j * (dim_2 * dim_3) + k * dim_3 +
                 l];
}

/**
 * Four-dimensional array with the last dimension set at runtime and the other
 * dimensions set at compile-time.
 */
template <int dim_0, int dim_1, int dim_2, typename MemorySpaceType>
class Array4D<dim_0, dim_1, dim_2, -1, MemorySpaceType>
{
public:
  static_assert(dim_0 > 0, "dim_0 should be greater than 0.");
  static_assert(dim_1 > 0, "dim_1 should be greater than 0.");
  static_assert(dim_2 > 0, "dim_2 should be greater than 0.");

  /**
   * Default constructor.
   */
  Array4D() = default;

  /**
   * Constructor. Set the last dimension to `size_3`.
   */
  Array4D(unsigned int size_3);

  /**
   * Copy constructor. The copy is non-owning, i.e., the lifetime of the
   * underlying data is determined by the original object.
   */
  Array4D(Array4D<dim_0, dim_1, dim_2, -1, MemorySpaceType> const &other);

  /**
   * Destructor.
   */
  ~Array4D();

  /**
   * Reinitialize the data using `size_3` for the last dimension. The initial
   * data is cleared.
   */
  void reinit(unsigned int size_3);

  /**
   * Set the data to zero.
   */
  void set_zero();

  /**
   * Dimension `i` of the array.
   */
  ADAMANTINE_HOST_DEV unsigned int extent(unsigned int i) const;

  /**
   * Access element `(i,j,k,l)` of the array.
   */
  ADAMANTINE_HOST_DEV double &operator()(unsigned int i, unsigned int j,
                                         unsigned int k, unsigned int l);

  /**
   * Access element `(i,j,k,l)` of the array.
   */
  ADAMANTINE_HOST_DEV double const &operator()(unsigned int i, unsigned int j,
                                               unsigned int k,
                                               unsigned int l) const;

private:
  /**
   * Owning flag true if owning and false otherwise.
   */
  bool _owning = true;
  /**
   * Dimension three of the array.
   */
  unsigned int _dim_3 = 0;
  /**
   * Pointer to data.
   */
  double *_values = nullptr;
};

template <int dim_0, int dim_1, int dim_2, typename MemorySpaceType>
Array4D<dim_0, dim_1, dim_2, -1, MemorySpaceType>::Array4D(unsigned int size_3)
    : _dim_3(size_3), _values(Memory<double, MemorySpaceType>::allocate_data(
                          dim_0 * dim_1 * dim_2 * _dim_3))
{
}

template <int dim_0, int dim_1, int dim_2, typename MemorySpaceType>
Array4D<dim_0, dim_1, dim_2, -1, MemorySpaceType>::Array4D(
    Array4D<dim_0, dim_1, dim_2, -1, MemorySpaceType> const &other)
    : _owning(false), _dim_3(other._dim_3), _values(other._values)
{
}

template <int dim_0, int dim_1, int dim_2, typename MemorySpaceType>
Array4D<dim_0, dim_1, dim_2, -1, MemorySpaceType>::~Array4D()
{
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }
  _values = nullptr;
}

template <int dim_0, int dim_1, int dim_2, typename MemorySpaceType>
void Array4D<dim_0, dim_1, dim_2, -1, MemorySpaceType>::reinit(
    unsigned int size_3)
{
  // Free memory if necessary
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }

  _dim_3 = size_3;
  _values = Memory<double, MemorySpaceType>::allocate_data(dim_0 * dim_1 *
                                                           dim_2 * _dim_3);
}

template <int dim_0, int dim_1, int dim_2, typename MemorySpaceType>
void Array4D<dim_0, dim_1, dim_2, -1, MemorySpaceType>::set_zero()
{

  Memory<double, MemorySpaceType>::set_zero(_values,
                                            dim_0 * dim_1 * dim_2 * _dim_3);
}

template <int dim_0, int dim_1, int dim_2, typename MemorySpaceType>
ADAMANTINE_HOST_DEV unsigned int
Array4D<dim_0, dim_1, dim_2, -1, MemorySpaceType>::extent(unsigned int i) const
{
  if (i == 0)
    return dim_0;
  else if (i == 1)
    return dim_1;
  else if (i == 2)
    return dim_2;
  else if (i == 3)
    return _dim_3;
  else
    return 0;
}
template <int dim_0, int dim_1, int dim_2, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double &
Array4D<dim_0, dim_1, dim_2, -1, MemorySpaceType>::operator()(unsigned int i,
                                                              unsigned int j,
                                                              unsigned int k,
                                                              unsigned int l)
{
  return _values[i * (dim_1 * dim_2 * _dim_3) + j * (dim_2 * _dim_3) +
                 k * _dim_3 + l];
}

template <int dim_0, int dim_1, int dim_2, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double const &
Array4D<dim_0, dim_1, dim_2, -1, MemorySpaceType>::operator()(
    unsigned int i, unsigned int j, unsigned int k, unsigned int l) const
{
  return _values[i * (dim_1 * dim_2 * _dim_3) + j * (dim_2 * _dim_3) +
                 k * _dim_3 + l];
}

/**
 * Four-dimensional array with the first and the last dimensions set at runtime
 * and the other dimensions set at compile-time.
 */
template <int dim_1, int dim_2, typename MemorySpaceType>
class Array4D<-1, dim_1, dim_2, -1, MemorySpaceType>
{
public:
  static_assert(dim_1 > 0, "dim_1 should be greater than 0.");
  static_assert(dim_2 > 0, "dim_2 should be greater than 0.");

  /**
   * Default constructor.
   */
  Array4D() = default;

  /**
   * Constructor. Set the first dimension to `size_0` and the last dimension to
   * `size_3`.
   */
  Array4D(unsigned int size_0, unsigned int size_3);

  /**
   * Copy constructor. The copy is non-owning, i.e., the lifetime of the
   * underlying data is determined by the original object.
   */
  Array4D(Array4D<-1, dim_1, dim_2, -1, MemorySpaceType> const &other);

  /**
   * Destructor.
   */
  ~Array4D();

  /**
   * Reinitialize the data using `size_0` for the first dimension. The initial
   * data is cleared.
   */
  void reinit(unsigned int size_0, unsigned int size_3);

  /**
   * Set the data to zero.
   */
  void set_zero();

  /**
   * Dimension `i` of the array.
   */
  ADAMANTINE_HOST_DEV unsigned int extent(unsigned int i) const;

  /**
   * Access element `(i,j,k,l)` of the array.
   */
  ADAMANTINE_HOST_DEV double &operator()(unsigned int i, unsigned int j,
                                         unsigned int k, unsigned int l);

  /**
   * Access element `(i,j,k,l)` of the array.
   */
  ADAMANTINE_HOST_DEV double const &operator()(unsigned int i, unsigned int j,
                                               unsigned int k,
                                               unsigned int l) const;

private:
  /**
   * Owning flag true if owning and false otherwise.
   */
  bool _owning = true;
  /**
   * Dimension zero of the array.
   */
  unsigned int _dim_0 = 0;
  /**
   * Dimension three of the array.
   */
  unsigned int _dim_3 = 0;
  /**
   * Pointer to data.
   */
  double *_values = nullptr;
};

template <int dim_1, int dim_2, typename MemorySpaceType>
Array4D<-1, dim_1, dim_2, -1, MemorySpaceType>::Array4D(unsigned int size_0,
                                                        unsigned int size_3)
    : _dim_0(size_0), _dim_3(size_3),
      _values(Memory<double, MemorySpaceType>::allocate_data(_dim_0 * dim_1 *
                                                             dim_2 * _dim_3))
{
}

template <int dim_1, int dim_2, typename MemorySpaceType>
Array4D<-1, dim_1, dim_2, -1, MemorySpaceType>::Array4D(
    Array4D<-1, dim_1, dim_2, -1, MemorySpaceType> const &other)
    : _owning(false), _dim_0(other._dim_0), _dim_3(other._dim_3),
      _values(other._values)
{
}

template <int dim_1, int dim_2, typename MemorySpaceType>
Array4D<-1, dim_1, dim_2, -1, MemorySpaceType>::~Array4D()
{
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }
  _values = nullptr;
}

template <int dim_1, int dim_2, typename MemorySpaceType>
void Array4D<-1, dim_1, dim_2, -1, MemorySpaceType>::reinit(unsigned int size_0,
                                                            unsigned int size_3)
{
  // Free memory if necessary
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }

  _dim_0 = size_0;
  _dim_3 = size_3;
  _values = Memory<double, MemorySpaceType>::allocate_data(_dim_0 * dim_1 *
                                                           dim_2 * _dim_3);
}

template <int dim_1, int dim_2, typename MemorySpaceType>
void Array4D<-1, dim_1, dim_2, -1, MemorySpaceType>::set_zero()
{

  Memory<double, MemorySpaceType>::set_zero(_values,
                                            _dim_0 * dim_1 * dim_2 * _dim_3);
}

template <int dim_1, int dim_2, typename MemorySpaceType>
ADAMANTINE_HOST_DEV unsigned int
Array4D<-1, dim_1, dim_2, -1, MemorySpaceType>::extent(unsigned int i) const
{
  if (i == 0)
    return _dim_0;
  else if (i == 1)
    return dim_1;
  else if (i == 2)
    return dim_2;
  else if (i == 3)
    return _dim_3;
  else
    return 0;
}
template <int dim_1, int dim_2, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double &
Array4D<-1, dim_1, dim_2, -1, MemorySpaceType>::operator()(unsigned int i,
                                                           unsigned int j,
                                                           unsigned int k,
                                                           unsigned int l)
{
  return _values[i * (dim_1 * dim_2 * _dim_3) + j * (dim_2 * _dim_3) +
                 k * _dim_3 + l];
}

template <int dim_1, int dim_2, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double const &
Array4D<-1, dim_1, dim_2, -1, MemorySpaceType>::operator()(unsigned int i,
                                                           unsigned int j,
                                                           unsigned int k,
                                                           unsigned int l) const
{
  return _values[i * (dim_1 * dim_2 * _dim_3) + j * (dim_2 * _dim_3) +
                 k * _dim_3 + l];
}

/**
 * Five-dimensional array with all dimension set at compile-time.
 */
template <int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,
          typename MemorySpaceType>
class Array5D
{
public:
  static_assert(dim_0 > 0, "dim_0 should be greater than 0.");
  static_assert(dim_1 > 0, "dim_1 should be greater than 0.");
  static_assert(dim_2 > 0, "dim_2 should be greater than 0.");
  static_assert(dim_3 > 0, "dim_3 should be greater than 0.");
  static_assert(dim_4 > 0, "dim_4 should be greater than 0.");

  /**
   * Default constructor.
   */
  Array5D();

  /**
   * Copy constructor. The copy is non-owning, i.e., the lifetime of the
   * underlying data is determined by the original object.
   */
  Array5D(
      Array5D<dim_0, dim_1, dim_2, dim_3, dim_4, MemorySpaceType> const &other);

  /**
   * Destructor.
   */
  ~Array5D();

  /**
   * Set the data to zero.
   */
  void set_zero();

  /**
   * Dimension `i` of the array.
   */
  ADAMANTINE_HOST_DEV unsigned int extent(unsigned int i) const;

  /**
   * Access element `(i,j,k,l,m)` of the array.
   */
  ADAMANTINE_HOST_DEV double &operator()(unsigned int i, unsigned int j,
                                         unsigned int k, unsigned int l,
                                         unsigned int m);

  /**
   * Access element `(i,j,k,l,m)` of the array.
   */
  ADAMANTINE_HOST_DEV double const &operator()(unsigned int i, unsigned int j,
                                               unsigned int k, unsigned int l,
                                               unsigned int m) const;

private:
  /**
   * Owning flag true if owning and false otherwise.
   */
  bool _owning = true;
  /**
   * Pointer to data.
   */
  double *_values = nullptr;
};

template <int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,
          typename MemorySpaceType>
Array5D<dim_0, dim_1, dim_2, dim_3, dim_4, MemorySpaceType>::Array5D()
    : _values(Memory<double, MemorySpaceType>::allocate_data(
          dim_0 * dim_1 * dim_2 * dim_3 * dim_4))
{
}

template <int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,
          typename MemorySpaceType>
Array5D<dim_0, dim_1, dim_2, dim_3, dim_4, MemorySpaceType>::Array5D(
    Array5D<dim_0, dim_1, dim_2, dim_3, dim_4, MemorySpaceType> const &other)
    : _owning(false), _values(other._values)
{
}

template <int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,
          typename MemorySpaceType>
Array5D<dim_0, dim_1, dim_2, dim_3, dim_4, MemorySpaceType>::~Array5D()
{
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }
  _values = nullptr;
}

template <int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,
          typename MemorySpaceType>
void Array5D<dim_0, dim_1, dim_2, dim_3, dim_4, MemorySpaceType>::set_zero()
{

  Memory<double, MemorySpaceType>::set_zero(_values, dim_0 * dim_1 * dim_2 *
                                                         dim_3 * dim_4);
}

template <int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,
          typename MemorySpaceType>
ADAMANTINE_HOST_DEV unsigned int
Array5D<dim_0, dim_1, dim_2, dim_3, dim_4, MemorySpaceType>::extent(
    unsigned int i) const
{
  if (i == 0)
    return dim_0;
  else if (i == 1)
    return dim_1;
  else if (i == 2)
    return dim_2;
  else if (i == 3)
    return dim_3;
  else if (i == 4)
    return dim_4;
  else
    return 0;
}

template <int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,
          typename MemorySpaceType>
ADAMANTINE_HOST_DEV double &
Array5D<dim_0, dim_1, dim_2, dim_3, dim_4, MemorySpaceType>::operator()(
    unsigned int i, unsigned int j, unsigned int k, unsigned int l,
    unsigned int m)
{
  return _values[i * (dim_1 * dim_2 * dim_3 * dim_4) +
                 j * (dim_2 * dim_3 * dim_4) + k * (dim_3 * dim_4) + l * dim_4 +
                 m];
}

template <int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,
          typename MemorySpaceType>
ADAMANTINE_HOST_DEV double const &
Array5D<dim_0, dim_1, dim_2, dim_3, dim_4, MemorySpaceType>::operator()(
    unsigned int i, unsigned int j, unsigned int k, unsigned int l,
    unsigned int m) const
{
  return _values[i * (dim_1 * dim_2 * dim_3 * dim_4) +
                 j * (dim_2 * dim_3 * dim_4) + k * (dim_3 * dim_4) + l * dim_4 +
                 m];
}

/**
 * Five-dimensional array with the first and the fourth dimensions set at
 * runtime and the other dimensions set at compile-time.
 */
template <int dim_1, int dim_2, int dim_4, typename MemorySpaceType>
class Array5D<-1, dim_1, dim_2, -1, dim_4, MemorySpaceType>
{
public:
  static_assert(dim_1 > 0, "dim_1 should be greater than 0.");
  static_assert(dim_2 > 0, "dim_2 should be greater than 0.");
  static_assert(dim_4 > 0, "dim_4 should be greater than 0.");

  /**
   * Default constructor.
   */
  Array5D() = default;

  /**
   * Constructor. Set the first dimension to `size_0` and the fourth dimension
   * to `size_3`.
   */
  Array5D(unsigned int size_0, unsigned int size_3);

  /**
   * Copy constructor. The copy is non-owning, i.e., the lifetime of the
   * underlying data is determined by the original object.
   */
  Array5D(Array5D<-1, dim_1, dim_2, -1, dim_4, MemorySpaceType> const &other);

  /**
   * Destructor.
   */
  ~Array5D();

  /**
   * Reinitialize the data using `size_0` for the first dimension and `size_3`
   * for the fourth dimensio. The initial data is cleared.
   */
  void reinit(unsigned int size_0, unsigned int size_3);

  /**
   * Set the data to zero.
   */
  void set_zero();

  /**
   * Dimension `i` of the array.
   */
  ADAMANTINE_HOST_DEV unsigned int extent(unsigned int i) const;

  /**
   * Access element `(i,j,k,l,m)` of the array.
   */
  ADAMANTINE_HOST_DEV double &operator()(unsigned int i, unsigned int j,
                                         unsigned int k, unsigned int l,
                                         unsigned int m);
  /**
   * Access element `(i,j,k,l,m)` of the array.
   */
  ADAMANTINE_HOST_DEV double const &operator()(unsigned int i, unsigned int j,
                                               unsigned int k, unsigned int l,
                                               unsigned int m) const;

private:
  /**
   * Owning flag true if owning and false otherwise.
   */
  bool _owning = true;
  /**
   * Dimension zero of the array.
   */
  unsigned int _dim_0 = 0;
  /**
   * Dimension three of the array.
   */
  unsigned int _dim_3 = 0;
  /**
   * Pointer to data.
   */
  double *_values = nullptr;
};

template <int dim_1, int dim_2, int dim_4, typename MemorySpaceType>
Array5D<-1, dim_1, dim_2, -1, dim_4, MemorySpaceType>::Array5D(
    unsigned int size_0, unsigned int size_3)
    : _dim_0(size_0), _dim_3(size_3),
      _values(Memory<double, MemorySpaceType>::allocate_data(
          _dim_0 * dim_1 * dim_2 * _dim_3 * dim_4))
{
}

template <int dim_1, int dim_2, int dim_4, typename MemorySpaceType>
Array5D<-1, dim_1, dim_2, -1, dim_4, MemorySpaceType>::Array5D(
    Array5D<-1, dim_1, dim_2, -1, dim_4, MemorySpaceType> const &other)
    : _owning(false), _dim_0(other._dim_0), _dim_3(other._dim_3),
      _values(other._values)
{
}

template <int dim_1, int dim_2, int dim_4, typename MemorySpaceType>
void Array5D<-1, dim_1, dim_2, -1, dim_4, MemorySpaceType>::reinit(
    unsigned int size_0, unsigned int size_3)
{
  // Free memory if necessary
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }

  _dim_0 = size_0;
  _dim_3 = size_3;
  _values = Memory<double, MemorySpaceType>::allocate_data(
      _dim_0 * dim_1 * dim_2 * _dim_3 * dim_4);
}

template <int dim_1, int dim_2, int dim_4, typename MemorySpaceType>
Array5D<-1, dim_1, dim_2, -1, dim_4, MemorySpaceType>::~Array5D()
{
  if (_owning && _values)
  {
    Memory<double, MemorySpaceType>::delete_data(_values);
  }
  _values = nullptr;
}

template <int dim_1, int dim_2, int dim_4, typename MemorySpaceType>
void Array5D<-1, dim_1, dim_2, -1, dim_4, MemorySpaceType>::set_zero()
{

  Memory<double, MemorySpaceType>::set_zero(_values, _dim_0 * dim_1 * dim_2 *
                                                         _dim_3 * dim_4);
}

template <int dim_1, int dim_2, int dim_4, typename MemorySpaceType>
ADAMANTINE_HOST_DEV unsigned int
Array5D<-1, dim_1, dim_2, -1, dim_4, MemorySpaceType>::extent(
    unsigned int i) const
{
  if (i == 0)
    return _dim_0;
  else if (i == 1)
    return dim_1;
  else if (i == 2)
    return dim_2;
  else if (i == 3)
    return _dim_3;
  else if (i == 4)
    return dim_4;
  else
    return 0;
}

template <int dim_1, int dim_2, int dim_4, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double &
Array5D<-1, dim_1, dim_2, -1, dim_4, MemorySpaceType>::operator()(
    unsigned int i, unsigned int j, unsigned int k, unsigned int l,
    unsigned int m)
{
  return _values[i * (dim_1 * dim_2 * _dim_3 * dim_4) +
                 j * (dim_2 * _dim_3 * dim_4) + k * (_dim_3 * dim_4) +
                 l * dim_4 + m];
}

template <int dim_1, int dim_2, int dim_4, typename MemorySpaceType>
ADAMANTINE_HOST_DEV double const &
Array5D<-1, dim_1, dim_2, -1, dim_4, MemorySpaceType>::operator()(
    unsigned int i, unsigned int j, unsigned int k, unsigned int l,
    unsigned int m) const
{
  return _values[i * (dim_1 * dim_2 * _dim_3 * dim_4) +
                 j * (dim_2 * _dim_3 * dim_4) + k * (_dim_3 * dim_4) +
                 l * dim_4 + m];
}

} // namespace adamantine

#endif
