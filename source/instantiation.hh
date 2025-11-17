/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <boost/preprocessor/seq/for_each_product.hpp>

// clang-format off
#define ADAMANTINE_DIM (2)(3)
#define ADAMANTINE_USE_TABLE (true)(false)
#define ADAMANTINE_N_MATERIALS (-1)(1)
#define ADAMANTINE_P_ORDER (0)(1)(2)(3)(4)
#define ADAMANTINE_FE_DEGREE (1)(2)(3)(4)(5)
#define ADAMANTINE_MATERIAL_STATE (adamantine::Solid)(adamantine::SolidLiquid)(adamantine::SolidLiquidPowder)
#define ADAMANTINE_QUADRATURE_TYPE (dealii::QGauss<1>)(dealii::QGaussLobatto<1>)
// clang-format on

// Instantiation of the class for:
//   - dim = 2 and 3
#define ADAMANTINE_D(z, SEQ)                                                   \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ)>;
#define INSTANTIATE_DIM(NAME)                                                  \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(ADAMANTINE_D, ((NAME))(ADAMANTINE_DIM))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - p_order = 0 to 4
//   - material_state = Solid, SolidLiquid, and SolidLiquidPowder
//   - memory_space = Host
#define ADAMANTINE_D_N_P_M_HOST(z, SEQ)                                        \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              dealii::MemorySpace::Host>;
#define INSTANTIATE_DIM_NMAT_PORDER_MATERIALSTATES_HOST(NAME)                  \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_P_M_HOST,                                                 \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_P_ORDER)(ADAMANTINE_MATERIAL_STATE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - p_order = 0 to 4
//   - material_state = Solid, SolidLiquid, and SolidLiquidPowder
//   - memory_space = Default
#define ADAMANTINE_D_N_P_M_DEV(z, SEQ)                                         \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              dealii::MemorySpace::Default>;
#define INSTANTIATE_DIM_NMAT_PORDER_MATERIALSTATES_DEVICE(NAME)                \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_P_M_DEV,                                                  \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_P_ORDER)(ADAMANTINE_MATERIAL_STATE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - use_table = true and false
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - material_state = Solid
//   - memory_space = Host
#define ADAMANTINE_D_N_U_P_F_S_HOST(z, SEQ)                                    \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              BOOST_PP_SEQ_ELEM(5, SEQ), adamantine::Solid,                    \
              dealii::MemorySpace::Host>;
#define INSTANTIATE_DIM_NMAT_USETABLE_PORDER_FEDEGREE_S_HOST(NAME)             \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_U_P_F_S_HOST,                                             \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_USE_TABLE)(ADAMANTINE_P_ORDER)(ADAMANTINE_FE_DEGREE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_material = -1 and 1
//   - use_table = true and false
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - material_state = SolidLiquid
//   - memory_space = Host
#define ADAMANTINE_D_N_U_P_F_SL_HOST(z, SEQ)                                   \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              BOOST_PP_SEQ_ELEM(5, SEQ), adamantine::SolidLiquid,              \
              dealii::MemorySpace::Host>;
#define INSTANTIATE_DIM_NMAT_USETABLE_PORDER_FEDEGREE_SL_HOST(NAME)            \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_U_P_F_SL_HOST,                                            \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_USE_TABLE)(ADAMANTINE_P_ORDER)(ADAMANTINE_FE_DEGREE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - use_table = true and false
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - material_state = SolidLiquidPowder
//   - memory_space = Host
#define ADAMANTINE_D_N_U_P_F_SLP_HOST(z, SEQ)                                  \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              BOOST_PP_SEQ_ELEM(5, SEQ), adamantine::SolidLiquidPowder,        \
              dealii::MemorySpace::Host>;
#define INSTANTIATE_DIM_NMAT_USETABLE_PORDER_FEDEGREE_SLP_HOST(NAME)           \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_U_P_F_SLP_HOST,                                           \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_USE_TABLE)(ADAMANTINE_P_ORDER)(ADAMANTINE_FE_DEGREE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - use_table = true and false
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - material_state = Solid
//   - memory_space = Default
#define ADAMANTINE_D_N_U_P_F_S_DEV(z, SEQ)                                     \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              BOOST_PP_SEQ_ELEM(5, SEQ), adamantine::Solid,                    \
              dealii::MemorySpace::Default>;
#define INSTANTIATE_DIM_NMAT_USETABLE_PORDER_FEDEGREE_S_DEVICE(NAME)           \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_U_P_F_S_DEV,                                              \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_USE_TABLE)(ADAMANTINE_P_ORDER)(ADAMANTINE_FE_DEGREE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - use_table = true and false
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - material_state = SolidLiquid
//   - memory_space = Default
#define ADAMANTINE_D_N_U_P_F_SL_DEV(z, SEQ)                                    \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              BOOST_PP_SEQ_ELEM(5, SEQ), adamantine::SolidLiquid,              \
              dealii::MemorySpace::Default>;
#define INSTANTIATE_DIM_NMAT_USETABLE_PORDER_FEDEGREE_SL_DEVICE(NAME)          \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_U_P_F_SL_DEV,                                             \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_USE_TABLE)(ADAMANTINE_P_ORDER)(ADAMANTINE_FE_DEGREE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - use_table = true and false
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - material_state = SolidLiquidPowder
//   - memory_space = Default
#define ADAMANTINE_D_N_U_P_F_SLP_DEV(z, SEQ)                                   \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              BOOST_PP_SEQ_ELEM(5, SEQ), adamantine::SolidLiquidPowder,        \
              dealii::MemorySpace::Default>;
#define INSTANTIATE_DIM_NMAT_USETABLE_PORDER_FEDEGREE_SLP_DEVICE(NAME)         \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_U_P_F_SLP_DEV,                                            \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_USE_TABLE)(ADAMANTINE_P_ORDER)(ADAMANTINE_FE_DEGREE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - material_state = Solid
//   - memory_space = Host
//   - QuadratureType = dealii::QGauss<1> and dealii::QGaussLobatto<1>
#define ADAMANTINE_D_N_P_F_S_Q_HOST(z, SEQ)                                    \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              adamantine::Solid, dealii::MemorySpace::Host,                    \
              BOOST_PP_SEQ_ELEM(5, SEQ)>;
#define INSTANTIATE_DIM_NMAT_PORDER_FEDEGREE_S_QUAD_HOST(NAME)                 \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_P_F_S_Q_HOST,                                             \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_P_ORDER)(ADAMANTINE_FE_DEGREE)(ADAMANTINE_QUADRATURE_TYPE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - material_state = SolidLiquid
//   - memory_space = Host
//   - QuadratureType = dealii::QGauss<1> and dealii::QGaussLobatto<1>
#define ADAMANTINE_D_N_P_F_SL_Q_HOST(z, SEQ)                                   \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              adamantine::SolidLiquid, dealii::MemorySpace::Host,              \
              BOOST_PP_SEQ_ELEM(5, SEQ)>;
#define INSTANTIATE_DIM_NMAT_PORDER_FEDEGREE_SL_QUAD_HOST(NAME)                \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_P_F_SL_Q_HOST,                                            \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_P_ORDER)(ADAMANTINE_FE_DEGREE)(ADAMANTINE_QUADRATURE_TYPE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - material_state = SolidLiquidPowder
//   - memory_space = Host
//   - QuadratureType = dealii::QGauss<1> and dealii::QGaussLobatto<1>
#define ADAMANTINE_D_N_P_F_SLP_Q_HOST(z, SEQ)                                  \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              adamantine::SolidLiquidPowder, dealii::MemorySpace::Host,        \
              BOOST_PP_SEQ_ELEM(5, SEQ)>;
#define INSTANTIATE_DIM_NMAT_PORDER_FEDEGREE_SLP_QUAD_HOST(NAME)               \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_P_F_SLP_Q_HOST,                                           \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_P_ORDER)(ADAMANTINE_FE_DEGREE)(ADAMANTINE_QUADRATURE_TYPE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - material_state = Solid
//   - memory_space = Default
//   - QuadratureType = dealii::QGauss<1> and dealii::QGaussLobatto<1>
#define ADAMANTINE_D_N_P_F_S_Q_DEV(z, SEQ)                                     \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              adamantine::Solid, dealii::MemorySpace::Default,                 \
              BOOST_PP_SEQ_ELEM(5, SEQ)>;
#define INSTANTIATE_DIM_NMAT_PORDER_FEDEGREE_S_QUAD_DEVICE(NAME)               \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_P_F_S_Q_DEV,                                              \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_P_ORDER)(ADAMANTINE_FE_DEGREE)(ADAMANTINE_QUADRATURE_TYPE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - material_state = SolidLiquid
//   - memory_space = Default
//   - QuadratureType = dealii::QGauss<1> and dealii::QGaussLobatto<1>
#define ADAMANTINE_D_N_P_F_SL_Q_DEV(z, SEQ)                                    \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              adamantine::SolidLiquid, dealii::MemorySpace::Default,           \
              BOOST_PP_SEQ_ELEM(5, SEQ)>;
#define INSTANTIATE_DIM_NMAT_PORDER_FEDEGREE_SL_QUAD_DEVICE(NAME)              \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_P_F_SL_Q_DEV,                                             \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_P_ORDER)(ADAMANTINE_FE_DEGREE)(ADAMANTINE_QUADRATURE_TYPE))

// Instantiation of the class for:
//   - dim = 2 and 3
//   - n_materials = -1 and 1
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - material_state = SolidLiquidPowder
//   - memory_space = Default
//   - QuadratureType = dealii::QGauss<1> and dealii::QGaussLobatto<1>
#define ADAMANTINE_D_N_P_F_SLP_Q_DEV(z, SEQ)                                   \
  template class adamantine::BOOST_PP_SEQ_ELEM(                                \
      0, SEQ)<BOOST_PP_SEQ_ELEM(1, SEQ), BOOST_PP_SEQ_ELEM(2, SEQ),            \
              BOOST_PP_SEQ_ELEM(3, SEQ), BOOST_PP_SEQ_ELEM(4, SEQ),            \
              adamantine::SolidLiquidPowder, dealii::MemorySpace::Default,     \
              BOOST_PP_SEQ_ELEM(5, SEQ)>;
#define INSTANTIATE_DIM_NMAT_PORDER_FEDEGREE_SLP_QUAD_DEVICE(NAME)             \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                               \
      ADAMANTINE_D_N_P_F_SLP_Q_DEV,                                            \
      ((NAME))(                                                                \
          ADAMANTINE_DIM)(ADAMANTINE_N_MATERIALS)(ADAMANTINE_P_ORDER)(ADAMANTINE_FE_DEGREE)(ADAMANTINE_QUADRATURE_TYPE))
