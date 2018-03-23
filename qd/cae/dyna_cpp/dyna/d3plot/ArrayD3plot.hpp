
#ifndef ARRAYD3PLOT_HPP
#define ARRAYD3PLOT_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <dyna_cpp/math/Tensor.hpp>

#include <dyna_cpp/dyna/d3plot/AbstractBuffer.hpp>
#include <dyna_cpp/dyna/d3plot/D3plotHeader.hpp>

namespace qd {
 
class ArrayD3plot
{
public:
  D3plotHeader header;

  bool _has_nel10;               // dunno anymore
  bool _has_external_numbers_I8; // if 64bit integers written, not 32
  bool _has_internal_energy;
  bool _has_temperatures;
  bool _has_mass_scaling_info; // true if dyna_it > 10 (little more complicate)

  int32_t _n_deletion_vars; // size of deletion array in the file

  int32_t _word_position; // tracker of word position in file
  int32_t _words_to_read;
  int32_t _word_position_of_states; // remembers where states begin

  bool useFemzip; // femzip usage?
  int32_t femzip_state_offset;

  // buffer for data
  std::unique_ptr<AbstractBuffer> buffer;

  // Data
  std::map<std::string, Tensor<int32_t>> int_data;
  std::map<std::string, Tensor<float>> float_data;
  std::map<std::string, std::vector<std::string>> string_data;
};
}

#endif