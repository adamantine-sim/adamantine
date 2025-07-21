/* SPDX-FileCopyrightText: Copyright (c) 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef BOUNDARY_HH
#define BOUNDARY_HH

#include <deal.II/base/types.h>

#include <boost/property_tree/ptree.hpp>

#include <string>
#include <vector>

namespace adamantine
{
/**
 * Enum on the different types of boundary condition supported. Some of them can
 * be combined, for example radiative and convective.
 */
enum BoundaryType
{
  invalid = 0,
  adiabatic = 0x1,
  radiative = 0x2,
  convective = 0x4,
  clamped = 0x8
};

/**
 * Global operator which returns an object in which all bits are set which are
 * either set in the first or the second argument. This operator exists since if
 * it did not then the result of the bit-or operator | would be an integer which
 * would in turn trigger a compiler warning when we tried to assign it to an
 * object of type BoundaryType.
 */
inline BoundaryType operator|(const BoundaryType b1, const BoundaryType b2)
{
  return static_cast<BoundaryType>(static_cast<unsigned int>(b1) |
                                   static_cast<unsigned int>(b2));
}

/**
 * Global operator which sets the bits from the second argument also in the
 * first one.
 */
inline BoundaryType &operator|=(BoundaryType &b1, const BoundaryType b2)
{
  b1 = b1 | b2;
  return b1;
}

/**
 * Global operator which returns an object in which all bits are set which are
 * set in the first as well as the second argument. This operator exists since
 * if it did not then the result of the bit-and operator & would be an integer
 * which would in turn trigger a compiler warning when we tried to assign it to
 * an object of type BoundaryType.
 */
inline BoundaryType operator&(const BoundaryType b1, const BoundaryType b2)
{
  return static_cast<BoundaryType>(static_cast<unsigned int>(b1) &
                                   static_cast<unsigned int>(b2));
}

/**
 * Global operator which clears all the bits in the first argument if they are
 * not also set in the second argument.
 */
inline BoundaryType &operator&=(BoundaryType &b1, const BoundaryType b2)
{
  b1 = b1 & b2;
  return b1;
}

/**
 * This class parses the Boundary database and stores the BoundaryType
 * associated with each boundary id.
 */
class Boundary
{
public:
  /**
   * Constructor.
   */
  Boundary(boost::property_tree::ptree const &database,
           std::vector<dealii::types::boundary_id> const &boundary_ids,
           bool const mechanical_only);

  /**
   * Return the number of boundary ids associated with the Geometry.
   */
  unsigned int n_boundary_ids() const;

  /**
   * Return the BoundaryType associated with the given boundary id.
   */
  BoundaryType get_boundary_type(dealii::types::boundary_id boundary_id) const;

  /**
   * Return all the boundary ids associated with the given BoundaryType
   */
  std::vector<dealii::types::boundary_id>
  get_boundary_ids(BoundaryType type) const;

private:
  /**
   * Vector of all the BoundaryType present in the Geometry.
   */
  std::vector<BoundaryType> _types;

  /**
   * Return the BoundaryType associated with the given string.
   */
  BoundaryType parse_boundary_line(std::string boundary_type_str);
};

inline unsigned int Boundary::n_boundary_ids() const { return _types.size(); }

inline BoundaryType Boundary::get_boundary_type(unsigned int boundary_id) const
{
  if (boundary_id < _types.size() - 1)
    return _types[boundary_id];
  else
    return _types.back();
}
} // namespace adamantine

#endif
