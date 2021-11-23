/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef VALIDATE_INPUT_DATABASE_HH
#define VALIDATE_INPUT_DATABASE_HH

#include <boost/program_options.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
void validate_input_database(boost::property_tree::ptree &database);
} // namespace adamantine

#endif
