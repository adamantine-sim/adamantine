/* Copyright (c) 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MECHANICAL_PHYSICS_DUMMY_HH
#define MECHANICAL_PHYSICS_DUMMY_HH

namespace adamantine
{
/*
 *  This is a dummy implementation of MechanicalPhysics for the sole purpose of
 * having valid declarations in the case that adamantine is not compiled with
 * the deal.II weak forms library. At no time should any of these methods be
 * called in a calculation.
 */
template <int dim, typename MemorySpaceType>
class MechanicalPhysics
{
public:
  MechanicalPhysics(MPI_Comm const &, unsigned int, Geometry<dim> &,
                    MaterialProperty<dim, MemorySpaceType> &,
                    std::vector<double>,
                    [[maybe_unused]] bool include_gravity = false)
  {
    throw_exception();
  };

  void setup_dofs() { throw_exception(); };

  void setup_dofs(
      dealii::DoFHandler<dim> const &,
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &,
      std::vector<bool> const &)
  {
    throw_exception();
  };

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> solve()
  {
    throw_exception();
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dummy;
    return dummy;
  };

  dealii::DoFHandler<dim> &get_dof_handler()
  {
    throw_exception();
    return _dof_handler;
  };

  dealii::AffineConstraints<double> &get_affine_constraints()
  {
    throw_exception();
    return _affine_constraints;
  };

private:
  void throw_exception()
  {
    ASSERT_THROW(false, "Error: Code execution should never reach methods in "
                        "this dummy class.");
  };

  dealii::DoFHandler<dim> _dof_handler;

  dealii::AffineConstraints<double> _affine_constraints;
};

} // namespace adamantine

#endif
