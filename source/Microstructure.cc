/* SPDX-FileCopyrightText: Copyright (c) 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <Microstructure.hh>
#include <instantiation.hh>

#include <filesystem>
#include <iomanip>

namespace adamantine
{
template <int dim>
Microstructure<dim>::Microstructure(MPI_Comm communicator,
                                    std::string const &filename_prefix)
    : _communicator(communicator), _filename_prefix(filename_prefix)
{
  auto rank = dealii::Utilities::MPI::this_mpi_process(_communicator);
  _file.open(_filename_prefix + "_" + std::to_string(rank) + ".txt");
  // Set the precision to 9 digits
  _file << std::setprecision(9);
}

template <int dim>
Microstructure<dim>::~Microstructure()
{
  // Concatenate all the temporary files written by each processor.
  // This is good enough for now but it won't scale. We probably want to use
  // ADIOS in the future.
  _file.close();
  MPI_Barrier(_communicator);
  if (dealii::Utilities::MPI::this_mpi_process(_communicator) == 0)
  {
    std::string global_filename = _filename_prefix + ".txt";
    // Remove the global file if it already exist. If it doesn't exist, remove
    // returns an error code that we ignore.
    std::filesystem::remove(global_filename);
    unsigned int const n_procs =
        dealii::Utilities::MPI::n_mpi_processes(_communicator);
    for (unsigned int rank = 0; rank < n_procs; ++rank)
    {
      // This needs to be inside the loop. I don't understand why but if
      // global_file is defined outside the loop the file is not written
      // correctly.
      std::ofstream global_file(global_filename, std::ios::app);
      std::string local_filename =
          _filename_prefix + "_" + std::to_string(rank) + ".txt";
      std::ifstream local_file(local_filename);
      global_file << local_file.rdbuf();

      std::filesystem::remove(local_filename);
    }
  }
}

template <int dim>
void Microstructure<dim>::set_old_temperature(
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
        &old_temperature)
{
  _old_temperature = old_temperature;
}
} // namespace adamantine

//-------------------- Explicit Instantiations --------------------//
INSTANTIATE_DIM(Microstructure)
