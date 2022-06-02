/* Copyright (c) 2021-2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MEMORY_BLOCK_HH
#define MEMORY_BLOCK_HH

#include <utils.hh>

#include <array>
#include <ostream>

namespace adamantine
{
/**
 * This class allocates a block of memory on the host or the device. To access
 * the underlying data use MemoryBlockView.
 */
template <typename Number, typename MemorySpaceType>
class MemoryBlock
{
public:
  using memory_space = MemorySpaceType;

  /**
   * Maximum number of dimensions the block of memory can be subdivided into.
   */
  static int constexpr n_dim = 5;

  /**
   * Default constructor. No memory is allocated.
   */
  MemoryBlock() = default;

  /**
   * Constructor.
   */
  MemoryBlock(unsigned int dim_0, unsigned int dim_1 = 0,
              unsigned int dim_2 = 0, unsigned int dim_3 = 0,
              unsigned int dim_4 = 0);

  /**
   * Copy constructor.
   */
  MemoryBlock(MemoryBlock<Number, MemorySpaceType> const &other);

  /**
   * Copy @p other and move the data to the correct memory space.
   */
  template <typename MemorySpaceType2>
  MemoryBlock(MemoryBlock<Number, MemorySpaceType2> const &other);

  /**
   * Copy @p input_data and move the data to the correct memory space.
   */
  MemoryBlock(std::vector<Number> const &input_data);

  /**
   * Free the memory and reallocate new memory block.
   */
  void reinit(unsigned int dim_0, unsigned int dim_1 = 0,
              unsigned int dim_2 = 0, unsigned int dim_3 = 0,
              unsigned int dim_4 = 0);

  /**
   * Free the memory, reallocate new memory block, and copy the data from @p
   * other to the correct memory space.
   */
  template <typename MemorySpaceType2>
  void reinit(MemoryBlock<Number, MemorySpaceType2> const &other);

  /**
   * Free the memory, reallocate new memory block, and copy the data from @p
   * input_data to the correct memory space.
   */
  void reinit(std::vector<Number> const &input_data);

  /**
   * Destructor.
   */
  ~MemoryBlock();

  /**
   * Free the memory.
   */
  void clear();

  /**
   * Number of allocated elements.
   */
  unsigned int size() const;

  /**
   * Return the @p i dimension.
   */
  unsigned int extent(unsigned i) const;

  /**
   * Initialize the memory block to zero.
   */
  void set_zero();

  /**
   * Return the pointer to the underlying data.
   */
  Number *data() const;

private:
  template <typename Number2, typename MemorySpaceType2>
  friend std::ostream &
  operator<<(std::ostream &out,
             MemoryBlock<Number2, MemorySpaceType2> const &memory_block);

  template <typename Number2, typename MemorySpaceType2>
  friend class MemoryBlock;

  template <typename Number2, typename MemorySpaceType2>
  friend class MemoryBlockView;

  unsigned int _size = 0;
  std::array<unsigned int, n_dim> _extent = {{0, 0, 0, 0, 0}};
  Number *_data = nullptr;
};

template <typename Number, typename MemorySpaceType>
MemoryBlock<Number, MemorySpaceType>::MemoryBlock(unsigned int dim_0,
                                                  unsigned int dim_1,
                                                  unsigned int dim_2,
                                                  unsigned int dim_3,
                                                  unsigned int dim_4)
{
  _extent[0] = dim_0;
  _extent[1] = dim_1;
  _extent[2] = dim_2;
  _extent[3] = dim_3;
  _extent[4] = dim_4;

  _size = _extent[0];
  for (unsigned int i = 1; i < n_dim; ++i)
  {
    if (_extent[i] != 0)
      _size *= _extent[i];
  }

  _data = Memory<Number, MemorySpaceType>::allocate_data(_size);
}

template <typename Number, typename MemorySpaceType>
MemoryBlock<Number, MemorySpaceType>::MemoryBlock(
    MemoryBlock<Number, MemorySpaceType> const &other)
{
  _size = other._size;
  _extent = other._extent;
  _data = Memory<Number, MemorySpaceType>::allocate_data(_size);
  deep_copy(_data, MemorySpaceType{}, other._data, MemorySpaceType{}, _size);
}

template <typename Number, typename MemorySpaceType>
template <typename MemorySpaceType2>
MemoryBlock<Number, MemorySpaceType>::MemoryBlock(
    MemoryBlock<Number, MemorySpaceType2> const &other)
{
  _size = other._size;
  _extent = other._extent;
  _data = Memory<Number, MemorySpaceType>::allocate_data(_size);
  deep_copy(_data, MemorySpaceType{}, other._data, MemorySpaceType2{}, _size);
}

template <typename Number, typename MemorySpaceType>
MemoryBlock<Number, MemorySpaceType>::MemoryBlock(
    std::vector<Number> const &input_data)
{
  _size = input_data.size();
  _extent[0] = _size;
  _data = Memory<Number, MemorySpaceType>::allocate_data(_size);
  deep_copy(_data, MemorySpaceType{}, input_data.data(),
            dealii::MemorySpace::Host{}, _size);
}

template <typename Number, typename MemorySpaceType>
void MemoryBlock<Number, MemorySpaceType>::reinit(unsigned int dim_0,
                                                  unsigned int dim_1,
                                                  unsigned int dim_2,
                                                  unsigned int dim_3,
                                                  unsigned int dim_4)
{
  clear();
  _extent[0] = dim_0;
  _extent[1] = dim_1;
  _extent[2] = dim_2;
  _extent[3] = dim_3;
  _extent[4] = dim_4;

  _size = _extent[0];
  for (unsigned int i = 1; i < n_dim; ++i)
  {
    if (_extent[i] != 0)
      _size *= _extent[i];
  }

  _data = Memory<Number, MemorySpaceType>::allocate_data(_size);
}

template <typename Number, typename MemorySpaceType>
template <typename MemorySpaceType2>
void MemoryBlock<Number, MemorySpaceType>::reinit(
    MemoryBlock<Number, MemorySpaceType2> const &other)
{
  clear();
  _size = other._size;
  _extent = other._extent;
  _data = Memory<Number, MemorySpaceType>::allocate_data(_size);
  deep_copy(*this, other);
}

template <typename Number, typename MemorySpaceType>
void MemoryBlock<Number, MemorySpaceType>::reinit(
    std::vector<Number> const &input_data)
{
  clear();
  _size = input_data.size();
  _extent[0] = _size;
  _extent[1] = 0;
  _extent[2] = 0;
  _extent[3] = 0;
  _extent[4] = 0;
  _data = Memory<Number, MemorySpaceType>::allocate_data(_size);
  deep_copy(_data, MemorySpaceType{}, input_data.data(),
            dealii::MemorySpace::Host{}, _size);
}

template <typename Number, typename MemorySpaceType>
MemoryBlock<Number, MemorySpaceType>::~MemoryBlock()
{
  clear();
}

template <typename Number, typename MemorySpaceType>
void MemoryBlock<Number, MemorySpaceType>::clear()
{
  if (_data != nullptr)
  {
    _size = 0;
    _extent = {{0, 0, 0, 0, 0}};
    Memory<Number, MemorySpaceType>::delete_data(_data);
    _data = nullptr;
  }
}

template <typename Number, typename MemorySpaceType>
unsigned int MemoryBlock<Number, MemorySpaceType>::size() const
{
  return _size;
}

template <typename Number, typename MemorySpaceType>
unsigned int MemoryBlock<Number, MemorySpaceType>::extent(unsigned int i) const
{
  ASSERT(i < n_dim, "i greater than the number of dimensions of MemoryBlock");

  return _extent[i];
}

template <typename Number, typename MemorySpaceType>
Number *MemoryBlock<Number, MemorySpaceType>::data() const
{
  return _data;
}

template <typename Number, typename MemorySpaceType>
void MemoryBlock<Number, MemorySpaceType>::set_zero()
{
  Memory<Number, MemorySpaceType>::set_zero(_data, _size);
}

template <typename Number, typename MemorySpaceType>
std::ostream &
operator<<(std::ostream &out,
           MemoryBlock<Number, MemorySpaceType> const &memory_block)
{
  MemoryBlock<Number, dealii::MemorySpace::Host> memory_block_host(
      memory_block);
  for (unsigned int i = 0; i < memory_block_host._size; ++i)
  {
    out << i << ": " << memory_block_host._data[i] << "\n";
  }

  return out;
}
} // namespace adamantine

#endif
