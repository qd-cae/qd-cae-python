
#include <dyna_cpp/dyna/ArrayD3plot.hpp>

namespace qd {

/**Constructor of an ArrayD3plot
 *
 *
 */
ArrayD3plot::ArrayD3plot(const std::string& filepath)
  : _filepath(filepath)
  , _has_nel10(false)
  , _has_external_numbers_I8(false)
  , _has_internal_energy(false)
  , _has_temperatures(false)
  , _has_mass_scaling_info(false)
  , _n_deletion_vars(0)
  , _word_position(0)
  , _words_to_read(0)
  , _word_position_of_states(0)
  , _use_femzip(false)
  , _femzip_state_offset(0)
  , _buffer([](const std::string& filepath) {

// WTF is this ?!?!?!
// This is a lambda for initialization of the buffer variable
// Since the buffer is a std::unique_ptr I need to do it in the
// initializer list. And since it is a little bit more complicated,
// I need to use a lambda function
#ifdef QD_USE_FEMZIP

    if (FemzipBuffer::is_femzip_compressed(filepath))
      return std::move(std::make_unique<FemzipBuffer>(filepath));
    else {
      const int32_t bytesPerWord = 4;
      return std::move(std::make_unique<D3plotBuffer>(_filename, bytesPerWord));
    }

#else
    const int32_t bytesPerWord = 4;
    return std::move(std::make_unique<D3plotBuffer>(_filename, bytesPerWord));
#endif

  }(filepath))
{}

} // namespace:qd