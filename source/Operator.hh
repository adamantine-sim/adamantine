/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _OPERATOR_HH_
#define _OPERATOR_HH_

#include "types.hh"
#include "utils.hh"
#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace adamantine
{

template <typename NumberType>
class Operator : public dealii::Subscriptor
{
public:
  Operator() = default;

  virtual ~Operator() = default;

  virtual dealii::types::global_dof_index m() const = 0;

  virtual dealii::types::global_dof_index n() const = 0;

  virtual void
  vmult(dealii::LA::distributed::Vector<NumberType> &dst,
        dealii::LA::distributed::Vector<NumberType> const &src) const = 0;

  virtual void
  Tvmult(dealii::LA::distributed::Vector<NumberType> &dst,
         dealii::LA::distributed::Vector<NumberType> const &src) const = 0;

  virtual void
  vmult_add(dealii::LA::distributed::Vector<NumberType> &dst,
            dealii::LA::distributed::Vector<NumberType> const &src) const = 0;

  virtual void
  Tvmult_add(dealii::LA::distributed::Vector<NumberType> &dst,
             dealii::LA::distributed::Vector<NumberType> const &src) const = 0;

  virtual void
  jacobian_vmult(dealii::LA::distributed::Vector<NumberType> &dst,
                 dealii::LA::distributed::Vector<NumberType> const &src) const
  {
    (void)dst;
    (void)src;
    ASSERT_THROW(false, "Function not implemented.");
  }
};
}

#endif
