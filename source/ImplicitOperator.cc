/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <ImplicitOperator.hh>
#include <instantiation.hh>
#include <utils.hh>

namespace adamantine
{
template <typename MemorySpaceType>
ImplicitOperator<MemorySpaceType>::ImplicitOperator(
    std::shared_ptr<Operator<MemorySpaceType>> explicit_operator, bool jfnk)
    : _jfnk(jfnk), _explicit_operator(explicit_operator)
{
}

template <typename MemorySpaceType>
void ImplicitOperator<MemorySpaceType>::vmult(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  if (_jfnk == true)
  {
    dealii::LA::distributed::Vector<double, MemorySpaceType> tmp_dst(
        dst.get_partitioner());
    dealii::LA::distributed::Vector<double, MemorySpaceType> tmp_src(src);
    tmp_src *= (1. + 1e-10);
    _explicit_operator->vmult(dst, tmp_src);
    _explicit_operator->vmult(tmp_dst, src);
    dst -= tmp_dst;
    dst /= 1e-10;
  }
  else
    _explicit_operator->jacobian_vmult(dst, src);

  dst.scale(*_inverse_mass_matrix);
  dst *= -_tau;
  dst += src;
}

template <typename MemorySpaceType>
void ImplicitOperator<MemorySpaceType>::Tvmult(
    dealii::LA::distributed::Vector<double, MemorySpaceType> & /*dst*/,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const & /*src*/)
    const
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <typename MemorySpaceType>
void ImplicitOperator<MemorySpaceType>::vmult_add(
    dealii::LA::distributed::Vector<double, MemorySpaceType> & /*dst*/,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const & /*src*/)
    const
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <typename MemorySpaceType>
void ImplicitOperator<MemorySpaceType>::Tvmult_add(
    dealii::LA::distributed::Vector<double, MemorySpaceType> & /*dst*/,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const & /*src*/)
    const
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

// Instantiation
template class ImplicitOperator<dealii::MemorySpace::Host>;
template class ImplicitOperator<dealii::MemorySpace::Default>;
} // namespace adamantine
