
#ifndef MATHUTILITY_HPP
#define MATHUTILITY_HPP

#include <algorithm> // sort
#include <cmath>
#include <numeric>
#include <vector>

namespace qd {

class MathUtility
{

public:
  // vector
  template<typename T>
  static T v_median(std::vector<T> vec);
  template<typename T>
  static std::vector<T> v_add(std::vector<T> a, std::vector<T> b);
  template<typename T>
  static std::vector<T> v_subtr(std::vector<T> a, std::vector<T> b);
  template<typename T>
  static std::vector<T> v_dot(std::vector<T> a, std::vector<T> b);

  // matrix
  template<typename T>
  static std::vector<std::vector<T>> m_zeros(size_t n1, size_t n2);
  template<typename T>
  static std::vector<std::vector<T>> m_mult(std::vector<std::vector<T>> a,
                                            std::vector<std::vector<T>> b);

  // matrix vector
  template<typename T>
  static std::vector<T> mv_mult(std::vector<std::vector<T>> a,
                                std::vector<T> b);

  // functions
  template<typename T>
  static T mean(const std::vector<T>& _vec);
  template<typename T>
  static T middle(const std::vector<T>& _vec);
  template<typename T>
  static T max(const std::vector<T>& _vec);
  template<typename T>
  static T min(const std::vector<T>& _vec);
  template<typename T>
  static T mises_stress(const std::vector<T>& _stress_vector);
};

/*--------------------*/
/* Template functions */
/*--------------------*/

/*
 * Add two vectors.
 */
template<typename T>
std::vector<T>
MathUtility::v_add(std::vector<T> a, std::vector<T> b)
{

  if (a.size() != b.size())
    throw(
      std::invalid_argument("Can not subtract to vectors with unequal sizes."));

  for (size_t ii = 0; ii < a.size(); ++ii) {
    a[ii] += b[ii];
  }

  return a;
}

/*
 * Subtract two vectors.
 */
template<typename T>
std::vector<T>
MathUtility::v_subtr(std::vector<T> a, std::vector<T> b)
{

  if (a.size() != b.size())
    throw(
      std::invalid_argument("Can not subtract to vectors with unequal sizes."));

  for (size_t ii = 0; ii < a.size(); ++ii) {
    a[ii] -= b[ii];
  }

  return a;
}

/*
 * Median of a vector.
 */
template<typename T>
T
MathUtility::v_median(std::vector<T> vec)
{

  if (vec.empty())
    return 0.;

  sort(vec.begin(), vec.end());

  if (vec.size() % 2 == 0) {
    return (vec[vec.size() / 2 - 1] + vec[vec.size() / 2]) / 2;
  } else {
    return vec[vec.size() / 2];
  }
}

/*
 * Dot product of two vectors.
 */
template<typename T>
std::vector<T>
MathUtility::v_dot(std::vector<T> a, std::vector<T> b)
{

  if (a.size() != b.size())
    throw(std::invalid_argument(
      "Can not dot multiply vectors with unequal sizes."));

  for (size_t ii = 0; ii < a.size(); ++ii) {
    a[ii] *= b[ii];
  }

  return a;
}

/*
 * Initialize a matrix with zeros.
 */
template<typename T>
std::vector<std::vector<T>>
MathUtility::m_zeros(size_t n1, size_t n2)
{

  if (n1 == 0)
    throw(
      std::invalid_argument("matrix_zeros dimension 1 must be at least 1."));
  if (n2 == 0)
    throw(
      std::invalid_argument("matrix_zeros dimension 2 must be at least 1."));

  std::vector<std::vector<T>> matrix(n1);
  for (size_t ii = 0; ii < n1; ii++) {
    std::vector<T> matrix2(n2);
    matrix[ii] = matrix2;
    // matrix[ii].reserve(n2);
  }

  return matrix;
}

/*
 * Product of two matrices.
 */
template<typename T>
std::vector<std::vector<T>>
MathUtility::m_mult(std::vector<std::vector<T>> a,
                    std::vector<std::vector<T>> b)
{

  if ((a.size() < 1) | (b.size() < 1))
    throw(std::invalid_argument("Can not dot multiply empty containers."));

  std::vector<std::vector<T>> res = MathUtility::m_zeros(a.size(), b[0].size());

  for (size_t ii = 0; ii < res.size(); ++ii) {
    if (a[ii].size() != b.size())
      throw(std::invalid_argument("Matrix a and b do not meet dimensions."));
    for (size_t jj = 0; jj < res[ii].size(); ++jj) {
      for (size_t kk = 0; kk < b.size(); ++kk)
        res[ii][jj] += a[ii][kk] * b[kk][jj];
    }
  }

  return res;
}

/*
 * Product of a matrix with a vector.
 */
template<typename T>
std::vector<T>
MathUtility::mv_mult(std::vector<std::vector<T>> a, std::vector<T> b)
{

  if ((a.size() < 1) | (b.size() < 1))
    throw(std::invalid_argument("Can not dot multiply empty containers."));

  std::vector<T> res(b.size());

  for (size_t ii = 0; ii < b.size(); ++ii) {
    if (a[ii].size() != b.size())
      throw(
        std::invalid_argument("Matrix a and vector b do not meet dimensions."));
    for (size_t jj = 0; jj < b.size(); ++jj) {
      res[ii] += a[ii][jj] * b[jj];
    }
  }

  return res;
}

/** Compute the mean of a vector
 *
 * @param _vec : vector with data
 */
template<typename T>
inline T
MathUtility::mean(const std::vector<T>& _vec)
{

  // check
  static_assert(std::is_arithmetic<T>::value, "Type must be a number");

  T res = 0;
  for (auto entry : _vec) {
    res += entry;
  }
  return res / ((T)_vec.size());
}

/** Get the middle of a vector
 *
 * @param _vec : vector with data
 */
template<typename T>
inline T
MathUtility::middle(const std::vector<T>& _vec)
{

  // check
  static_assert(std::is_arithmetic<T>::value, "Type must be a number");

  auto mid = _vec.size() / 2;
  auto rest = _vec.size() % 2;
  if (rest)
    return _vec[mid];
  else
    return (_vec[mid] + _vec[mid + 1]) / (static_cast<T>(2));
}

/** Compute the max of a vector
 *
 * @param _vec : vector with data
 */
template<typename T>
inline T
MathUtility::max(const std::vector<T>& _vec)
{

  // check
  static_assert(std::is_arithmetic<T>::value, "Type must be a number");

  return *std::max_element(std::begin(_vec), std::end(_vec));
}

/** Compute the max of a vector
 *
 * @param _vec : vector with data
 */
template<typename T>
inline T
MathUtility::min(const std::vector<T>& _vec)
{

  // check
  static_assert(std::is_arithmetic<T>::value, "Type must be a number");

  return *std::min_element(std::begin(_vec), std::end(_vec));
}

/** Mises stress from a stress vector (xx,yy,zz,xy,yz,xz)
 *
 * @param _stress_vector : vector of stress components
 */
template<typename T>
inline T
MathUtility::mises_stress(const std::vector<T>& _stress_vector)
{

  return sqrt(_stress_vector[0] * _stress_vector[0] +
              _stress_vector[1] * _stress_vector[1] +
              _stress_vector[2] * _stress_vector[2] -
              _stress_vector[0] * _stress_vector[1] -
              _stress_vector[0] * _stress_vector[2] -
              _stress_vector[1] * _stress_vector[2] +
              3 * (_stress_vector[3] * _stress_vector[3] +
                   _stress_vector[4] * _stress_vector[4] +
                   _stress_vector[5] * _stress_vector[5]));
}

/** Converts negative indexes into positive ones python style
 *
 * @param _index index
 * @param _size max size of a container
 *
 * E.g. -1 -> last etc.
 */
template<typename T,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_signed<T>::value>::type* = nullptr>
inline T
index_treatment(T _index, size_t _size)
{

  if (_index < 0)
    _index = static_cast<T>(_size) - _index;

  return _index;
}

/** Converts negative indexes into positive ones python style
 *
 * @param _index index
 * @param _size max size of a container
 *
 * Unsigned types are always positive. This empty function ensures,
 * that no performance penalty is happening when we use unsigned
 * types.
 */
template<typename T,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_unsigned<T>::value>::type* = nullptr>
inline T
index_treatment(T _index, size_t _size)
{
  return _index;
}

/** Checks if argument is non negative and returns it back.
 *
 * @param _number
 *
 */
template<typename T,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_signed<T>::value>::type* = nullptr>
void
check_non_negative(T _number)
{
  if (_number < 0)
    throw(std::invalid_argument("Argument may not be negative."));
}

/** Checks if argument is non negative and returns it back.
 *
 * @param _number
 *
 * Does nothing for unsigned integrals, thus has no overhead of being used.
 */
template<typename T,
         typename std::enable_if<std::is_integral<T>::value &&
                                 std::is_unsigned<T>::value>::type* = nullptr>
void
check_non_negative(T _number)
{}

} // namespace qd

#endif
