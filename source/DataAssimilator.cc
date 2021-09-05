/* Copyright (c) 2020 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <DataAssimilator.hh>
namespace adamantine
{
template <typename MemorySpaceType>
void DataAssimilator<MemorySpaceType>::updateEnsemble(
    std::vector<dealii::LA::distributed::Vector<double, MemorySpaceType>>
        &sim_data,
    dealii::LA::distributed::Vector<double, MemorySpaceType> &expt_data)
{
}
} // namespace adamantine
