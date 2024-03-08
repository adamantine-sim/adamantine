/* Copyright (c) 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MATERIAL_STATES_HH
#define MATERIAL_STATES_HH

namespace adamantine
{
/**
 * This struct describes a material that can be found in three states: solid,
 * liquid, and powder as it is the case in PBF.
 */
struct SolidLiquidPowder
{
  /**
   * Enum on the possible material states.
   */
  enum class State
  {
    solid,
    liquid,
    powder,
    SIZE
  };

  /**
   * Maximum different number of states a given material can be.
   */
  static unsigned int constexpr n_material_states =
      static_cast<unsigned int>(State::SIZE);
};

/**
 * This struct describes a material that can be found in two states: solid and
 * liquid as it is the case in DED.
 */
struct SolidLiquid
{
  /**
   * Enum on the possible material states.
   */
  enum class State
  {
    solid,
    liquid,
    SIZE
  };

  /**
   * Maximum different number of states a given material can be.
   */
  static unsigned int constexpr n_material_states =
      static_cast<unsigned int>(State::SIZE);
};

/**
 * This struct describes a material that can be found in one state: solid as it
 * is the case in FDM.
 */
struct Solid
{
  /**
   * Enum on the possible material states.
   */
  enum class State
  {
    solid,
    liquid,
    SIZE
  };

  /**
   * Maximum different number of states a given material can be.
   */
  static unsigned int constexpr n_material_states =
      static_cast<unsigned int>(State::SIZE);
};

} // namespace adamantine

#endif
