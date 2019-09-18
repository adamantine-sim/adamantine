/* Copyright (c) 2016 - 2017, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef IMPLICIT_OPERATOR_HH
#define IMPLICIT_OPERATOR_HH

#include "Operator.hh"

namespace adamantine
{
/**
 * This class uses an operator \f$F\f$ and creates an operator
 * \f$I-\tau M^{-1} \frac{F}{dy}\f$. This operator is then inverted when using
 * an implicit time stepping scheme.
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

  /**
   * Set the parameter \f$\tau\f$ defined by the Runge-Kutta method.
   */
  void set_tau(double tau);

  /**
   * Set the shared pointer of the inverse of the mass matrix.
   */
  void set_inverse_mass_matrix(
      std::shared_ptr<dealii::LA::distributed::Vector<NumberType>>
          inverse_mass_matrix);

private:
  /**
   * Flag to switch between Jacobian-Free Newton Krylov method and exact
   * Jacobian method.
   */
  bool _jfnk;
  /**
   * Parameter of the Runge-Kutta method used.
   */
  double _tau;
  /**
   * Shared pointer of the inverse of the mass matrix.
   */
  std::shared_ptr<dealii::LA::distributed::Vector<NumberType>>
      _inverse_mass_matrix;
  /**
   * Shared pointer of the operator \f$F\f$.
   */
  std::shared_ptr<Operator<NumberType>> _explicit_operator;
};

template <typename NumberType>
inline dealii::types::global_dof_index ImplicitOperator<NumberType>::m() const
{
  return _explicit_operator->m();
}

template <typename NumberType>
inline dealii::types::global_dof_index ImplicitOperator<NumberType>::n() const
{
  return _explicit_operator->n();
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
} // namespace adamantine

#endif
