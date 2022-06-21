/* Copyright (c) 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MECHANICAL_PHYSICS_HH
#define MECHANICAL_PHYSICS_HH

#include <Geometry.hh>
#include <MechanicalOperator.hh>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/hp/fe_collection.h>

namespace adamantine
{
// TODO Inherit from Physics or remove Physics entirely?
template <int dim>
class MechanicalPhysics
{
public:
  /**
   * Constructor.
   */
  MechanicalPhysics(MPI_Comm const &communicator,
                    boost::property_tree::ptree const &database,
                    Geometry<dim> &geometry);

  /**
   * Setup the DoFHandler, the AffineConstraints, and the MechanicalOperator.
   */
  void setup_dofs();

  /**
   * Solve the mechanical problem and return the solution.
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> solve();

  /**
   * Return the DoFHandler.
   */
  dealii::DoFHandler<dim> const &get_dof_handler() const;

private:
  /**
   * Associated Geometry.
   */
  Geometry<dim> &_geometry;
  /**
   * Associated FECollection.
   */
  dealii::hp::FECollection<dim> _fe_collection;
  /**
   * Associated DoFHandler.
   */
  dealii::DoFHandler<dim> _dof_handler;
  /**
   * Associated AffineConstraints.
   */
  dealii::AffineConstraints<double> _affine_constraints;
  /**
   * Associated QCollection.
   */
  dealii::hp::QCollection<dim> _q_collection;
  /**
   * Pointer to the MechanicalOperator
   */
  std::unique_ptr<MechanicalOperator<dim>> _mechanical_operator;
};

template <int dim>
inline dealii::DoFHandler<dim> const &
MechanicalPhysics<dim>::get_dof_handler() const
{
  return _dof_handler;
}

} // namespace adamantine

#endif
