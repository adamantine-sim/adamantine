/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _IMPLICIT_OPERATOR_HH_
#define _IMPLICIT_OPERATOR_HH_

#include "Operator.hh"

namespace adamantine
{
/**
 * This class uses an operator F and creates an operator
 * \f$I-\tau M^{-1} \frac{F}{dy}\f$.
 * This operator is used when using an implicit time stepping scheme.
 */
template <typename NumberType>
class ImplicitOperator : public Operator<NumberType>
{
public:
  ImplicitOperator(std::shared_ptr<Operator<NumberType>> explicit_operator,
                   bool jfnk);

  dealii::types::global_dof_index m() const override;

  dealii::types::global_dof_index n() const override;

  void
  vmult(dealii::LA::distributed::Vector<NumberType> &dst,
        dealii::LA::distributed::Vector<NumberType> const &src) const override;

  void
  Tvmult(dealii::LA::distributed::Vector<NumberType> &dst,
         dealii::LA::distributed::Vector<NumberType> const &src) const override;

  void vmult_add(
      dealii::LA::distributed::Vector<NumberType> &dst,
      dealii::LA::distributed::Vector<NumberType> const &src) const override;

  void Tvmult_add(
      dealii::LA::distributed::Vector<NumberType> &dst,
      dealii::LA::distributed::Vector<NumberType> const &src) const override;

  void set_tau(double tau);

  void set_inverse_mass_matrix(
      std::shared_ptr<dealii::LA::distributed::Vector<NumberType>>
          inverse_mass_matrix);

private:
  bool _jfnk;
  double _tau;
  std::shared_ptr<dealii::LA::distributed::Vector<NumberType>>
      _inverse_mass_matrix;
  std::shared_ptr<Operator<NumberType>> _explicit_operator;
};

template <typename NumberType>
inline dealii::types::global_dof_index ImplicitOperator<NumberType>::m() const
{
  _explicit_operator->m();
}

template <typename NumberType>
inline dealii::types::global_dof_index ImplicitOperator<NumberType>::n() const
{
  _explicit_operator->n();
}

template <typename NumberType>
inline void ImplicitOperator<NumberType>::set_tau(double tau)
{
  _tau = tau;
}

template <typename NumberType>
inline void ImplicitOperator<NumberType>::set_inverse_mass_matrix(
    std::shared_ptr<dealii::LA::distributed::Vector<NumberType>>
        inverse_mass_matrix)
{
  _inverse_mass_matrix = inverse_mass_matrix;
}
}

#endif
