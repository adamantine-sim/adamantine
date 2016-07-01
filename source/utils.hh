/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <boost/assert.hpp>
#include <stdexcept>
#include <string>

namespace adamantine
{

void ASSERT(bool cond, std::string const &message)
{
  BOOST_ASSERT_MSG(cond, message.c_str());
}

void ASSERT_THROW(bool cond, std::string const &message)
{
  if (cond == false)
    throw std::runtime_error(message);
}

}
