
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
private:
  const std::string& _filepath;

  D3plotHeader header;

  bool _has_nel10;               // dunno anymore
  bool _has_external_numbers_I8; // if 64bit integers written, not 32
  bool _has_internal_energy;
  bool _has_temperatures;
  bool _has_mass_scaling_info; // true if dyna_it > 10 (little more complicate)

  int64_t _n_deletion_vars; // size of deletion array in the file

  int64_t _word_position; // tracker of word position in file
  int64_t _words_to_read;
  int64_t _word_position_of_states; // remembers where states begin

  bool _use_femzip; // femzip usage?
  int64_t _femzip_state_offset;

  // buffer for data
  std::unique_ptr<AbstractBuffer> _buffer;

  // Data
  std::map<std::string, Tensor<int32_t>> _int_data;
  std::map<std::string, Tensor<float>> _float_data;
  std::map<std::string, std::vector<std::string>> _string_data;

public:
  ArrayD3plot(const std::string& filepath);
  void read_header();
};
}

#endif