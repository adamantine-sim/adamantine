/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef OPERATOR_HH
#define OPERATOR_HH

#include <types.hh>
#include <utils.hh>

#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace adamantine
{
/**
 * This class defines the interface that every operator needs to implement.
 */
template <typename MemorySpaceType>
class Operator : public dealii::Subscriptor
{
public:
  Operator() = default;

  virtual ~Operator() = default;

  /**
   * Return the dimension of the codomain (or range) space. To remember: the
   * matrix is of dimension m×n.
   */
  virtual dealii::types::global_dof_index m() const = 0;

  /**
   * Return the dimension of the domain space. To remember: the matrix is of
   * dimension m×n.
   */
  virtual dealii::types::global_dof_index n() const = 0;

  /**
   * Matrix-vector multiplication. This function applies the operator to the
   * vector src.
   * \param[in] src
   * \param[out] dst
   */
  virtual void
  vmult(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
        dealii::LA::distributed::Vector<double, MemorySpaceType> const &src)
      const = 0;

  /**
   * Matrix-vector multiplication with the transposed matrix. This function
   * applies the transpose of the operator to the vector src.
   * \param[in] src
   * \param[out] dst
   */
  virtual void
  Tvmult(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
         dealii::LA::distributed::Vector<double, MemorySpaceType> const &src)
      const = 0;

  /**
   * Matrix-vector multiplication and addition of the result to dst. This
   * function applies the operator to the vector src and add the result to the
   * vector dst.
   * \param[in] src
   * \param[inout] dst
   */
  virtual void
  vmult_add(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
            dealii::LA::distributed::Vector<double, MemorySpaceType> const &src)
      const = 0;

  /**
   * Matrix-vector multiplication with the transposed matrix and addition of
   * the result to dst. This function applies the transpose of the operator to
   * the vector src and add the result to the vector dst.
   * \param[in] src
   * \param[inout] dst
   */
  virtual void
  Tvmult_add(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
             dealii::LA::distributed::Vector<double, MemorySpaceType> const
                 &src) const = 0;

  /**
   * Matrix-vector multiplication with the Jacobian. This function applies the
   * Jacobian of the operator to the vector src.
   * \param[in] src
   * \param[inout] dst
   */
  virtual void jacobian_vmult(
      [[maybe_unused]] dealii::LA::distributed::Vector<double, MemorySpaceType>
          &dst,
      [[maybe_unused]] dealii::LA::distributed::Vector<
          double, MemorySpaceType> const &src) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
};
} // namespace adamantine

#endif
