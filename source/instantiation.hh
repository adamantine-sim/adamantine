/* Copyright (c) 2016 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <boost/preprocessor/repeat_from_to.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/tuple/replace.hpp>

// clang-format off
// Instantiation of the class for:
//   - dim = 2 and 3
#define INST_DIM(z, dim, class_name) template class adamantine::class_name<dim>;
#define INSTANTIATE_DIM(class_name) BOOST_PP_REPEAT_FROM_TO(2, 4, INST_DIM, class_name)

#define INST_DIM_DEVICE(z, dim, class_name) template class adamantine::class_name<dim, dealii::MemorySpace::Default>;
#define INSTANTIATE_DIM_DEVICE(class_name) BOOST_PP_REPEAT_FROM_TO(2, 4, INST_DIM_DEVICE, class_name)

#define INST_DIM_HOST(z, dim, class_name) template class adamantine::class_name<dim, dealii::MemorySpace::Host>;
#define INSTANTIATE_DIM_HOST(class_name) BOOST_PP_REPEAT_FROM_TO(2, 4, INST_DIM_HOST, class_name)

#define TUPLE_N (0, 0, 0, 0, 0)
#define TUPLE(class_name) BOOST_PP_TUPLE_REPLACE(TUPLE_N, 0, class_name)

// Instantiation of the class for:
//   - dim = 2 and 3
//   - p_order = 0 to 4
#define M_P_ORDER_HOST_1(z, p_order, TUPLE_1) \
  template class adamantine::BOOST_PP_TUPLE_ELEM(0, TUPLE_1)<BOOST_PP_TUPLE_ELEM(1, TUPLE_1),\
  p_order, dealii::MemorySpace::Host>;
#define M_DIM_HOST_1(z, dim, TUPLE_0) \
  BOOST_PP_REPEAT_FROM_TO(0, 5, M_P_ORDER_HOST_1, BOOST_PP_TUPLE_REPLACE(TUPLE_0, 1, dim))
#define INSTANTIATE_DIM_PORDER_HOST(TUPLE_0) BOOST_PP_REPEAT_FROM_TO(2, 4, M_DIM_HOST_1, TUPLE_0)

#define M_P_ORDER_DEVICE(z, p_order, TUPLE_1) \
  template class adamantine::BOOST_PP_TUPLE_ELEM(0, TUPLE_1)<BOOST_PP_TUPLE_ELEM(1, TUPLE_1),\
  p_order, dealii::MemorySpace::Default>;
#define M_DIM_DEVICE_1(z, dim, TUPLE_0) \
  BOOST_PP_REPEAT_FROM_TO(0, 5, M_P_ORDER_DEVICE, BOOST_PP_TUPLE_REPLACE(TUPLE_0, 1, dim))
#define INSTANTIATE_DIM_PORDER_DEVICE(TUPLE_0) BOOST_PP_REPEAT_FROM_TO(2, 4, M_DIM_DEVICE_1, TUPLE_0)

// Instantiation of the class for:
//   - dim = 2 and 3
//   - use_table = true or false
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
#define USE_TABLE (true)(false)

#define M_FE_DEGREE_HOST_2(z, fe_degree, TUPLE_4) \
  template class adamantine::BOOST_PP_TUPLE_ELEM(0, TUPLE_4)<BOOST_PP_TUPLE_ELEM(1, TUPLE_4),\
  BOOST_PP_TUPLE_ELEM(2, TUPLE_4), BOOST_PP_TUPLE_ELEM(3, TUPLE_4), fe_degree,\
  dealii::MemorySpace::Host>;
#define M_P_ORDER_HOST_2(z, p_order, TUPLE_3) \
  BOOST_PP_REPEAT_FROM_TO(1, 6, M_FE_DEGREE_HOST_2, BOOST_PP_TUPLE_REPLACE(TUPLE_3, 3, p_order))
#define M_USE_TABLE_HOST_2(z, TUPLE_2, use_table) \
  BOOST_PP_REPEAT_FROM_TO(0, 5, M_P_ORDER_HOST_2, BOOST_PP_TUPLE_REPLACE(TUPLE_2, 2, use_table))
#define M_DIM_HOST_2(z, dim, TUPLE_1) \
  BOOST_PP_SEQ_FOR_EACH(M_USE_TABLE_HOST_2, BOOST_PP_TUPLE_REPLACE(TUPLE_1, 1, dim), USE_TABLE)
#define INSTANTIATE_DIM_USETABLE_PORDER_FEDEGREE_HOST(TUPLE_0) BOOST_PP_REPEAT_FROM_TO(2, 4, M_DIM_HOST_2, TUPLE_0)

#define M_FE_DEGREE_DEVICE_2(z, fe_degree, TUPLE_4) \
  template class adamantine::BOOST_PP_TUPLE_ELEM(0, TUPLE_4)<BOOST_PP_TUPLE_ELEM(1, TUPLE_4),\
  BOOST_PP_TUPLE_ELEM(2, TUPLE_4), BOOST_PP_TUPLE_ELEM(3, TUPLE_4), fe_degree,\
  dealii::MemorySpace::Default>;
#define M_P_ORDER_DEVICE_2(z, p_order, TUPLE_3) \
  BOOST_PP_REPEAT_FROM_TO(1, 6, M_FE_DEGREE_DEVICE_2, BOOST_PP_TUPLE_REPLACE(TUPLE_3, 3, p_order))
#define M_USE_TABLE_DEVICE_2(z, TUPLE_2, use_table) \
  BOOST_PP_REPEAT_FROM_TO(0, 5, M_P_ORDER_DEVICE_2, BOOST_PP_TUPLE_REPLACE(TUPLE_2, 2, use_table))
#define M_DIM_DEVICE_2(z, dim, TUPLE_1) \
  BOOST_PP_SEQ_FOR_EACH(M_USE_TABLE_DEVICE_2, BOOST_PP_TUPLE_REPLACE(TUPLE_1, 1, dim), USE_TABLE)
#define INSTANTIATE_DIM_USETABLE_PORDER_FEDEGREE_DEVICE(TUPLE_0) BOOST_PP_REPEAT_FROM_TO(2, 4, M_DIM_DEVICE_2, TUPLE_0)

// Instantiation of the class for:
//   - dim = 2 and 3
//   - p_order = 0 to 4
//   - fe_degree = 1 to 5
//   - QuadratureType = dealii::QGauss<1> and dealii::QGaussLobatto<1>
#define QUADRATURE_TYPE (dealii::QGauss<1>)(dealii::QGaussLobatto<1>)

#define M_QUADRATURE_TYPE_HOST_3(z, TUPLE_3, quadrature)\
  template class adamantine::BOOST_PP_TUPLE_ELEM(0, TUPLE_3)<BOOST_PP_TUPLE_ELEM(1, TUPLE_3),\
BOOST_PP_TUPLE_ELEM(2, TUPLE_3), BOOST_PP_TUPLE_ELEM(3, TUPLE_3), dealii::MemorySpace::Host,\
quadrature>;
#define M_FE_DEGREE_HOST_3(z, fe_degree, TUPLE_2) \
  BOOST_PP_SEQ_FOR_EACH(M_QUADRATURE_TYPE_HOST_3, \
      BOOST_PP_TUPLE_REPLACE(TUPLE_2, 3, fe_degree), QUADRATURE_TYPE)
#define M_P_ORDER_HOST_3(z, p_order, TUPLE_1) \
  BOOST_PP_REPEAT_FROM_TO(1, 6, M_FE_DEGREE_HOST_3, BOOST_PP_TUPLE_REPLACE(TUPLE_1, 2, p_order))
#define M_DIM_HOST_3(z, dim, TUPLE_0) \
  BOOST_PP_REPEAT_FROM_TO(0, 5, M_P_ORDER_HOST_3, BOOST_PP_TUPLE_REPLACE(TUPLE_0, 1, dim))
#define INSTANTIATE_DIM_PORDER_FEDEGREE_QUAD_HOST(TUPLE_0) BOOST_PP_REPEAT_FROM_TO(2, 4, M_DIM_HOST_3, TUPLE_0)

#define M_QUADRATURE_TYPE_DEVICE_3(z, TUPLE_3, quadrature)\
  template class adamantine::BOOST_PP_TUPLE_ELEM(0, TUPLE_3)<BOOST_PP_TUPLE_ELEM(1, TUPLE_3),\
BOOST_PP_TUPLE_ELEM(2, TUPLE_3), BOOST_PP_TUPLE_ELEM(3, TUPLE_3), dealii::MemorySpace::Default,\
quadrature>;
#define M_FE_DEGREE_DEVICE_3(z, fe_degree, TUPLE_2) \
  BOOST_PP_SEQ_FOR_EACH(M_QUADRATURE_TYPE_DEVICE_3, \
      BOOST_PP_TUPLE_REPLACE(TUPLE_2, 3, fe_degree), QUADRATURE_TYPE)
#define M_P_ORDER_DEVICE_3(z, p_order, TUPLE_1) \
  BOOST_PP_REPEAT_FROM_TO(1, 6, M_FE_DEGREE_DEVICE_3, BOOST_PP_TUPLE_REPLACE(TUPLE_1, 2, p_order))
#define M_DIM_DEVICE_3(z, dim, TUPLE_0) \
  BOOST_PP_REPEAT_FROM_TO(0, 5, M_P_ORDER_DEVICE_3, BOOST_PP_TUPLE_REPLACE(TUPLE_0, 1, dim))
#define INSTANTIATE_DIM_PORDER_FEDEGREE_QUAD_DEVICE(TUPLE_0) BOOST_PP_REPEAT_FROM_TO(2, 4, M_DIM_DEVICE_3, TUPLE_0)

// clang-format on
