/* SPDX-FileCopyrightText: Copyright (c) 2021 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
