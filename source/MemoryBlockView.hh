/* Copyright (c) 2021-2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MEMORY_BLOCK_VIEW_HH
#define MEMORY_BLOCK_VIEW_HH

#include <MemoryBlock.hh>

namespace adamantine
{
/**
 * This class is a view on a MemoryBlock. It does not own the memory, it can
 * only access it.
 */
template <typename Number, typename MemorySpaceType>
class MemoryBlockView
{
public:
  using memory_space = MemorySpaceType;

  /**
   * Default constructor.
   */
  KOKKOS_FUNCTION
  MemoryBlockView() = default;

  /**
   * Constructor.
   */
  KOKKOS_FUNCTION
  MemoryBlockView(MemoryBlock<Number, MemorySpaceType> const &memory_block);

  /**
   * Copy constructor.
   */
  KOKKOS_FUNCTION MemoryBlockView(
      MemoryBlockView<Number, MemorySpaceType> const &memory_block_view);

  /**
   * Reinitialize the pointer to the underlying data and the size.
   */
  KOKKOS_FUNCTION
  void reinit(MemoryBlock<Number, MemorySpaceType> const &memory_block);

  /**
   * Assignment operator.
   */
  KOKKOS_FUNCTION
  MemoryBlockView<Number, MemorySpaceType> &
  operator=(MemoryBlockView<Number, MemorySpaceType> const &memory_block_view);

  /**
   * Access operator for a 1D MemoryBlock.
   */
  KOKKOS_FUNCTION Number &operator()(unsigned int i) const;

  /**
   * Access operator for a 2D MemoryBlock.
   */
  KOKKOS_FUNCTION Number &operator()(unsigned int i, unsigned int j) const;

  /**
   * Access operator for a 3D MemoryBlock.
   */
  KOKKOS_FUNCTION Number &operator()(unsigned int i, unsigned int j,
                                     unsigned int k) const;

  /**
   * Access operator for a 4D MemoryBlock.
   */
  KOKKOS_FUNCTION Number &operator()(unsigned int i, unsigned int j,
                                     unsigned int k, unsigned int l) const;

  /**
   * Access operator for a 5D MemoryBlock.
   */
  KOKKOS_FUNCTION Number &operator()(unsigned int i, unsigned int j,
                                     unsigned int k, unsigned int l,
                                     unsigned int m) const;

  /**
   * Return the number of accessible elements.
   */
  KOKKOS_FUNCTION unsigned int size() const;

  /**
   * Return the @p i dimension.
   */
  KOKKOS_FUNCTION unsigned int extent(unsigned int i) const;

  /**
   * Return the pointer to the underlying data.
   */
  KOKKOS_FUNCTION Number *data() const;

private:
  unsigned int _size = 0;
  unsigned int _dim_0 = 0;
  unsigned int _dim_1 = 0;
  unsigned int _dim_2 = 0;
  unsigned int _dim_3 = 0;
  unsigned int _dim_4 = 0;
  Number *_data = nullptr;
};

template <typename Number, typename MemorySpaceType>
KOKKOS_FUNCTION MemoryBlockView<Number, MemorySpaceType>::MemoryBlockView(
    MemoryBlock<Number, MemorySpaceType> const &memory_block)
{
  _size = memory_block._size;
  _dim_0 = memory_block._extent[0];
  _dim_1 = memory_block._extent[1];
  _dim_2 = memory_block._extent[2];
  _dim_3 = memory_block._extent[3];
  _dim_4 = memory_block._extent[4];
  _data = memory_block._data;
}

template <typename Number, typename MemorySpaceType>
KOKKOS_FUNCTION MemoryBlockView<Number, MemorySpaceType>::MemoryBlockView(
    MemoryBlockView<Number, MemorySpaceType> const &memory_block_view)
{
  _size = memory_block_view._size;
  _dim_0 = memory_block_view._dim_0;
  _dim_1 = memory_block_view._dim_1;
  _dim_2 = memory_block_view._dim_2;
  _dim_3 = memory_block_view._dim_3;
  _dim_4 = memory_block_view._dim_4;
  _data = memory_block_view._data;
}

template <typename Number, typename MemorySpaceType>
KOKKOS_FUNCTION void MemoryBlockView<Number, MemorySpaceType>::reinit(
    MemoryBlock<Number, MemorySpaceType> const &memory_block)
{
  _size = memory_block._size;
  _dim_0 = memory_block._extent[0];
  _dim_1 = memory_block._extent[1];
  _dim_2 = memory_block._extent[2];
  _dim_3 = memory_block._extent[3];
  _dim_4 = memory_block._extent[4];
  _data = memory_block._data;
}

template <typename Number, typename MemorySpaceType>
KOKKOS_FUNCTION MemoryBlockView<Number, MemorySpaceType> &
MemoryBlockView<Number, MemorySpaceType>::operator=(
    MemoryBlockView<Number, MemorySpaceType> const &memory_block_view)
{
  _size = memory_block_view._size;
  _dim_0 = memory_block_view._dim_0;
  _dim_1 = memory_block_view._dim_1;
  _dim_2 = memory_block_view._dim_2;
  _dim_3 = memory_block_view._dim_3;
  _dim_4 = memory_block_view._dim_4;
  _data = memory_block_view._data;

  return *this;
}

template <typename Number, typename MemorySpaceType>
KOKKOS_FUNCTION Number &
MemoryBlockView<Number, MemorySpaceType>::operator()(unsigned int i) const
{
  ASSERT(i < _dim_0, "Out-of-bound access.");

  return _data[i];
}

template <typename Number, typename MemorySpaceType>
KOKKOS_FUNCTION Number &
MemoryBlockView<Number, MemorySpaceType>::operator()(unsigned int i,
                                                     unsigned int j) const
{
  ASSERT(i < _dim_0, "Out-of-bound access.");
  ASSERT(j < _dim_1, "Out-of-bound access.");

  return _data[i * _dim_1 + j];
}

template <typename Number, typename MemorySpaceType>
KOKKOS_FUNCTION Number &MemoryBlockView<Number, MemorySpaceType>::operator()(
    unsigned int i, unsigned int j, unsigned int k) const
{
  ASSERT(i < _dim_0, "Out-of-bound access.");
  ASSERT(j < _dim_1, "Out-of-bound access.");
  ASSERT(k < _dim_2, "Out-of-bound access.");

  return _data[i * (_dim_1 * _dim_2) + j * _dim_2 + k];
}

template <typename Number, typename MemorySpaceType>
KOKKOS_FUNCTION Number &MemoryBlockView<Number, MemorySpaceType>::operator()(
    unsigned int i, unsigned int j, unsigned int k, unsigned int l) const
{
  ASSERT(i < _dim_0, "Out-of-bound access.");
  ASSERT(j < _dim_1, "Out-of-bound access.");
  ASSERT(k < _dim_2, "Out-of-bound access.");
  ASSERT(l < _dim_3, "Out-of-bound access.");

  return _data[i * (_dim_1 * _dim_2 * _dim_3) + j * (_dim_2 * _dim_3) +
               k * _dim_3 + l];
}

template <typename Number, typename MemorySpaceType>
KOKKOS_FUNCTION Number &MemoryBlockView<Number, MemorySpaceType>::operator()(
    unsigned int i, unsigned int j, unsigned int k, unsigned int l,
    unsigned int m) const
{
  ASSERT(i < _dim_0, "Out-of-bound access.");
  ASSERT(j < _dim_1, "Out-of-bound access.");
  ASSERT(k < _dim_2, "Out-of-bound access.");
  ASSERT(l < _dim_3, "Out-of-bound access.");
  ASSERT(m < _dim_4, "Out-of-bound access.");

  return _data[i * (_dim_1 * _dim_2 * _dim_3 * _dim_4) +
               j * (_dim_2 * _dim_3 * _dim_4) + k * (_dim_3 * _dim_4) +
               l * _dim_4 + m];
}

template <typename Number, typename MemorySpaceType>
KOKKOS_FUNCTION unsigned int
MemoryBlockView<Number, MemorySpaceType>::size() const
{
  return _size;
}

template <typename Number, typename MemorySpaceType>
KOKKOS_FUNCTION unsigned int
MemoryBlockView<Number, MemorySpaceType>::extent(unsigned int i) const
{
  switch (i)
  {
  case 0:
    return _dim_0;
  case 1:
    return _dim_1;
  case 2:
    return _dim_2;
  case 3:
    return _dim_3;
  case 4:
    return _dim_4;
  default:
    return 0;
  }
}

template <typename Number, typename MemorySpaceType>
KOKKOS_FUNCTION Number *MemoryBlockView<Number, MemorySpaceType>::data() const
{
  return _data;
}
} // namespace adamantine

#endif
