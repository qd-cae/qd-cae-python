
#ifndef FEM_UTILITY_HPP

#include <dyna_cpp/utility/MathUtility.hpp>
#include <stdexcept>
#include <vector>


namespace qd {

/** Compute a variable for states
 *
 * @param _data : data vector
 * @param _mode : computation mode
 *
 * modes are:
 *  > 1 = max
 *  > 2 = min
 *  > 3 = out
 *  > 4 = mid
 *  > 5 = in
 *  > 6 = mean
 */
template<typename T>
inline T
compute_state_var_from_mode(const std::vector<T>& _data, int32_t _mode)
{

#ifdef QD_DEBUG
  if (_data.size() < 1) {
    throw std::invalid_argument("Can not compute state var from empty vector.");
  }
#endif

  switch (_mode) {
    case 1: // max
      return MathUtility::max(_data);
      break;
    case 2: // min
      return MathUtility::min(_data);
      break;
    case 3: // out
      return _data.back();
      break;
    case 4: // mid
      return MathUtility::middle(_data);
      break;
    case 5: // in
      return _data[0];
      break;
    case 6: // mean
      return MathUtility::mean(_data);
      break;
  } // end:switch

  return -1;
}

/** Compute a variable for states
 *
 * @param _data : data vector
 * @param _mode : computation mode
 *
 * modes are:
 *  > 1 = max
 *  > 2 = min
 *  > 3 = out
 *  > 4 = mid
 *  > 5 = in
 *  > 6 = mean
 */
template<typename T>
std::vector<T>
compute_state_var_from_mode(const std::vector<std::vector<T>>& _data,
                            int32_t _mode)
{

#ifdef QD_DEBUG
  if (_data.size() < 1)
    throw std::invalid_argument(
      "Can not compute state var from empty matrix vector.");
#endif

  std::vector<T> ret(_data.size());

  for (size_t ii = 0; ii < _data.size(); ++ii) {
    ret[ii] = compute_state_var_from_mode(_data[ii], _mode);
  }

  return ret;
}

/** Compute a variable for states
 *
 * @param _data : data vector
 * @param _modes : computation mode for every row
 *
 * modes are:
 *  > 1 = max
 *  > 2 = min
 *  > 3 = out
 *  > 4 = mid
 *  > 5 = in
 *  > 6 = mean
 */
template<typename T>
std::vector<T>
compute_state_var_from_mode(const std::vector<std::vector<T>>& _data,
                            const std::vector<int32_t> _modes)
{

#ifdef QD_DEBUG
  if (_data.size() < 1)
    throw std::invalid_argument(
      "Can not compute state var from empty matrix vector.");
  if (_modes.size() != _data.size())
    throw std::invalid_argument(
      "Data and computation modes must ahve same shape.");
#endif

  std::vector<T> ret(_data.size());

  for (size_t ii = 0; ii < _data.size(); ++ii) {
    ret[ii] = compute_state_var_from_mode(_data[ii], _modes[ii]);
  }

  return ret;
}

} // namespace qd

#endif