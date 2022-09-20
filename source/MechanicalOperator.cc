/* Copyright (c) 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <MechanicalOperator.hh>
#include <instantiation.hh>
#include <utils.hh>

#include <deal.II/base/index_set.h>
#include <deal.II/differentiation/ad.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <weak_forms/bilinear_forms.h>
#include <weak_forms/weak_forms.h>

namespace adamantine
{
namespace
{
// TODO Find a place for this
template <int dim>
class RightHandSide : public dealii::TensorFunction<1, dim, double>
{
public:
  dealii::Tensor<1, dim, double>
  value(const dealii::Point<dim> & /*p*/) const override
  {
    dealii::Tensor<1, dim, double> out;
    out[axis<dim>::z] = -10.;

    return out;
  }
};
} // namespace

template <int dim, typename MemorySpaceType>
MechanicalOperator<dim, MemorySpaceType>::MechanicalOperator(
    MPI_Comm const &communicator,
    MaterialProperty<dim, MemorySpaceType> &material_properties,
    double const initial_temperature)
    : _communicator(communicator), _initial_temperature(initial_temperature),
      _material_properties(material_properties)
{
}

template <int dim, typename MemorySpaceType>
void MechanicalOperator<dim, MemorySpaceType>::reinit(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<double> const &affine_constraints,
    dealii::hp::QCollection<dim> const &q_collection)
{
  _dof_handler = &dof_handler;
  _affine_constraints = &affine_constraints;
  _q_collection = &q_collection;
  assemble_system();
}

template <int dim, typename MemorySpaceType>
void MechanicalOperator<dim, MemorySpaceType>::vmult(
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> &dst,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
        &src) const
{
  _system_matrix.vmult(dst, src);
}

template <int dim, typename MemorySpaceType>
void MechanicalOperator<dim, MemorySpaceType>::Tvmult(
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> &dst,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
        &src) const
{
  _system_matrix.Tvmult(dst, src);
}

template <int dim, typename MemorySpaceType>
void MechanicalOperator<dim, MemorySpaceType>::vmult_add(
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> &dst,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
        &src) const
{
  _system_matrix.vmult_add(dst, src);
}

template <int dim, typename MemorySpaceType>
void MechanicalOperator<dim, MemorySpaceType>::Tvmult_add(
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> &dst,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
        &src) const
{
  _system_matrix.Tvmult_add(dst, src);
}

template <int dim, typename MemorySpaceType>
void MechanicalOperator<dim, MemorySpaceType>::update_temperature(
    dealii::DoFHandler<dim> const &thermal_dof_handler,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
        &temperature)
{
  _thermal_dof_handler = &thermal_dof_handler;
  _temperature = temperature;
}

template <int dim, typename MemorySpaceType>
void MechanicalOperator<dim, MemorySpaceType>::assemble_system()
{
  // Create the sparsity pattern. Since we use a Trilinos matrix we don't need
  // the sparsity pattern to outlive the sparse matrix.
  unsigned int const n_dofs = _dof_handler->n_dofs();
  dealii::DynamicSparsityPattern dsp(n_dofs, n_dofs);
  dealii::DoFTools::make_sparsity_pattern(*_dof_handler, dsp,
                                          *_affine_constraints, false);

  _system_matrix.reinit(dsp);

  // Create the test and the trial functions
  dealiiWeakForms::WeakForms::TestFunction<dim> const test;
  dealiiWeakForms::WeakForms::TrialSolution<dim> const trial;
  dealiiWeakForms::WeakForms::SubSpaceExtractors::Vector const
      subspace_extractor(0, "u", "\\mathbf{u}");

  auto const test_ss = test[subspace_extractor];
  auto const trial_ss = trial[subspace_extractor];

  // Get the gradient
  auto const test_grad = test_ss.gradient();
  auto const trial_grad = trial_ss.gradient();

  // Create a functor to evaluate to the stiffness tensor.
  // For now, we ignore the quadrature point.
  dealiiWeakForms::WeakForms::TensorFunctor<4, dim> const stiffness_coeff(
      "C", "\\mathcal{C}");
  auto stiffness_tensor = stiffness_coeff.template value<double, dim>(
      [this](dealii::FEValuesBase<dim> const &fe_values,
             unsigned int const /* q_point */) {
        auto const &cell = fe_values.get_cell();

        dealii::Tensor<4, dim, double> C;
        dealii::SymmetricTensor<2, dim> const I =
            dealii::unit_symmetric_tensor<dim>();
        double lambda = this->_material_properties.get_mechanical_property(
            cell, StateProperty::lame_first_parameter);
        double mu = this->_material_properties.get_mechanical_property(
            cell, StateProperty::lame_second_parameter);

        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
              for (unsigned int l = 0; l < dim; ++l)
                C[i][j][k][l] = lambda * I[i][j] * I[k][l] +
                                mu * (I[i][k] * I[j][l] + I[i][l] * I[j][k]);

        return C;
      });

  _system_rhs.reinit(_dof_handler->locally_owned_dofs(), _communicator);
  auto const test_val = test_ss.value();
  dealiiWeakForms::WeakForms::VectorFunctionFunctor<dim> const rhs_coeff(
      "f", "\\mathbf{f}");
  RightHandSide<dim> const rhs;

  // Assemble the bilinear from
  dealiiWeakForms::WeakForms::MatrixBasedAssembler<dim> assembler;
  // FIXME the formulation below is more widespread but it doesn't work in
  // dealii-weak_forms yet. Keeping the implementation to use it in the future.
  // assembler +=
  //     dealiiWeakForms::WeakForms::bilinear_form(test_grad, lambda + mu,
  //                                               trial_grad)
  //         .dV() +
  //     dealiiWeakForms::WeakForms::bilinear_form(test_grad, mu, trial_grad)
  //         .delta_IJ()
  //         .dV() -
  //     dealiiWeakForms::WeakForms::linear_form(test_val,
  //     rhs_coeff.value(rhs)).dV();
  assembler +=
      dealiiWeakForms::WeakForms::bilinear_form(test_grad, stiffness_tensor,
                                                trial_grad)
          .dV() -
      dealiiWeakForms::WeakForms::linear_form(test_val, rhs_coeff.value(rhs))
          .dV();

  std::unique_ptr<dealii::hp::FEValues<dim>> temperature_hp_fe_values;
  // If the initial temperature is positive, we solve the thermoelastic problem.
  if (_initial_temperature >= 0.)
  {
    // Create a functor to evaluate the thermal expansion
    temperature_hp_fe_values = std::make_unique<dealii::hp::FEValues<dim>>(
        _thermal_dof_handler->get_fe_collection(), *_q_collection,
        dealii::update_values);

    dealiiWeakForms::WeakForms::TensorFunctor<2, dim> const expansion_coeff(
        "B", "\\mathcal{B}");
    auto expansion_tensor = expansion_coeff.template value<double, dim>(
        [&](dealii::FEValuesBase<dim> const &fe_values,
            unsigned int const q_point) {
          // fe_values is associated with the mechanical DoFHandler. We use it
          // to get the cell and then we evaluate the temperature at the
          // quadrature point using the temperature DoFHandler.
          auto const &cell = fe_values.get_cell();
          // Since we use a Triangulation cell to reinitialize the hp::FEValues,
          // it will automatically choose the zero-th finite element.
          temperature_hp_fe_values->reinit(cell);
          auto &temperature_fe_values =
              temperature_hp_fe_values->get_present_fe_values();
          double delta_T = -_initial_temperature;
          for (unsigned int i = 0; i < temperature_fe_values.dofs_per_cell; ++i)
          {
            delta_T += temperature_fe_values.shape_value(i, q_point);
          }

          double alpha = this->_material_properties.get_mechanical_property(
              cell, StateProperty::thermal_expansion_coef);
          double lambda = this->_material_properties.get_mechanical_property(
              cell, StateProperty::lame_first_parameter);
          double mu = this->_material_properties.get_mechanical_property(
              cell, StateProperty::lame_second_parameter);

          // Get the identity 2nd-order tensor
          auto B = dealii::Physics::Elasticity::StandardTensors<dim>::I;
          B *= (3. * lambda + 2 * mu) * alpha * delta_T;

          return B;
        });
    // Need to think if it's really a linear form or a bilinear form and where
    // the grad is supposed to apply. We could use a bilinear form because the
    // temperature is also dependent on the position but then, we would
    // need to use petrov-galerkin formulation.
    assembler -=
        dealiiWeakForms::WeakForms::linear_form(test_grad, expansion_tensor)
            .dV();
  }

  // Now we pass in concrete objects to get data from and assemble into.
  assembler.assemble_system(_system_matrix, _system_rhs, *_affine_constraints,
                            *_dof_handler, *_q_collection);

  if (_bilinear_form_output)
  {
    if (dealii::Utilities::MPI::this_mpi_process(_communicator) == 0)
    {
      std::cout << "Solving the following elastostatic problem" << std::endl;
      dealiiWeakForms::WeakForms::SymbolicDecorations const decorator(
          dealiiWeakForms::WeakForms::SymbolicNamesAscii(),
          dealiiWeakForms::WeakForms::SymbolicNamesLaTeX(
              symbolic_names("\\mathbf{v}_")));
      std::cout << assembler.as_latex(decorator) << std::endl;
    }
    _bilinear_form_output = false;
  }
}
} // namespace adamantine

INSTANTIATE_DIM_HOST(MechanicalOperator)
#ifdef ADAMANTINE_HAVE_CUDA
INSTANTIATE_DIM_DEVICE(MechanicalOperator)
#endif
