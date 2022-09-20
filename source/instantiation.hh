/* Copyright (c) 2016 - 2021, the adamantine authors.
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

#define INST_DIM_DEVICE(z, dim, class_name) template class adamantine::class_name<dim, dealii::MemorySpace::CUDA>;
#define INSTANTIATE_DIM_DEVICE(class_name) BOOST_PP_REPEAT_FROM_TO(2, 4, INST_DIM_DEVICE, class_name)

#define INST_DIM_HOST(z, dim, class_name) template class adamantine::class_name<dim, dealii::MemorySpace::Host>;
#define INSTANTIATE_DIM_HOST(class_name) BOOST_PP_REPEAT_FROM_TO(2, 4, INST_DIM_HOST, class_name)

#define TUPLE_N (0, 0, 0)
#define TUPLE(class_name) BOOST_PP_TUPLE_REPLACE(TUPLE_N, 0, class_name)

// Instantiation of the class for:
//   - dim = 2 and 3
//   - fe_degree = 1 to 10
#define M_FE_DEGREE(z, fe_degree, TUPLE_1) \
  template class adamantine::BOOST_PP_TUPLE_ELEM(0, TUPLE_1)<BOOST_PP_TUPLE_ELEM(1, TUPLE_1),\
  fe_degree, dealii::MemorySpace::Host>;
#define M_DIM(z, dim, TUPLE_0) \
  BOOST_PP_REPEAT_FROM_TO(1, 11, M_FE_DEGREE, BOOST_PP_TUPLE_REPLACE(TUPLE_0, 1, dim))
#define INSTANTIATE_DIM_FEDEGREE_HOST(TUPLE_0) BOOST_PP_REPEAT_FROM_TO(2, 4, M_DIM, TUPLE_0)

// Instantiation of the class for:
//   - dim = 2 and 3
//   - fe_degree = 1 to 10
//   - QuadratureType = dealii::QGauss<1> and dealii::QGaussLobatto<1>
#define QUADRATURE_TYPE (dealii::QGauss<1>)(dealii::QGaussLobatto<1>)

#define MA_QUADRATURE_TYPE(z, TUPLE_2, quadrature)\
  template class adamantine::BOOST_PP_TUPLE_ELEM(0, TUPLE_2)<BOOST_PP_TUPLE_ELEM(1, TUPLE_2),\
BOOST_PP_TUPLE_ELEM(2, TUPLE_2), dealii::MemorySpace::Host, quadrature>;
#define MA_FE_DEGREE(z, fe_degree, TUPLE_1) \
  BOOST_PP_SEQ_FOR_EACH(MA_QUADRATURE_TYPE, \
      BOOST_PP_TUPLE_REPLACE(TUPLE_1, 2, fe_degree), QUADRATURE_TYPE)
#define MA_DIM(z, dim, TUPLE_0) \
  BOOST_PP_REPEAT_FROM_TO(1, 11, MA_FE_DEGREE, BOOST_PP_TUPLE_REPLACE(TUPLE_0, 1, dim))
#define INSTANTIATE_DIM_FEDEGREE_QUAD_HOST(TUPLE_0) BOOST_PP_REPEAT_FROM_TO(2, 4, MA_DIM, TUPLE_0)

// clang-format on
