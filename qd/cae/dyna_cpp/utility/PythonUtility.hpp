
#ifndef PYTHONUTILITY_HPP
#define PYTHONUTILITY_HPP

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// includes
extern "C"
{
#include <Python.h>
  //#include <numpy/arrayobject.h>
}

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <dyna_cpp/math/Tensor.hpp>

// namespaces
namespace qd {
namespace py {

/** Test if a PyObject is an integral (int or long)
 * @param PyObject* obj
 * @return bool isIntegral
 */
static inline bool
PyObject_isIntegral(PyObject* obj)
{
  if (PyLong_Check(obj)) {
    return true;

#if PY_MAJOR_VERSION < 3
  } else if (PyInt_Check(obj)) {
    return true;
#endif

  } else {
    return false;
  }
}

/** Check if a python object has a certain type
 * @param PyObject* obj : python object
 * @return bool is_type
 */
template<
  typename T,
  typename std::enable_if<std::is_same<std::string, T>::value>::type* = nullptr>
inline bool
is_type(PyObject* obj)
{
  return pybind11::isinstance<pybind11::str>(obj);
}

template<typename T,
         typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
inline bool
is_type(PyObject* obj)
{
  return pybind11::isinstance<pybind11::int_>(obj);
}

template<
  typename T,
  typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
inline bool
is_type(PyObject* obj)
{
  return pybind11::isinstance<pybind11::float_>(obj);
}

/** Convert a python list or tuple to a vector
 *
 * @param U _container : container to convert to a vector
 * @return vec
 */
template<typename T,
         typename U,
         typename std::enable_if<
           std::is_same<pybind11::list, U>::value ||
           std::is_same<pybind11::tuple, U>::value>::type* = nullptr>
std::vector<T>
container_to_vector(U _container, const std::string& error_message = "")
{
  std::vector<T> res;
  for (const auto& entry : _container) {

    // Check if casting is possible
    if (!is_type<T>(entry.ptr())) {
      if (!error_message.empty()) {
        throw(std::invalid_argument(error_message));
      } else {
        throw(std::invalid_argument(
          "An entry of the python list/tuple does not have a valid type!"));
      }
    }

    // cast and add it
    res.push_back(pybind11::cast<T>(entry));
  }

  return res;
}

/** Try to convert a string to a python number
 *
 * @param _str string to convert
 * @return obj python object
 *
 * If the conversion to integer or float fails,
 * a python string is returned.
 */
inline pybind11::object
try_number_conversion(const std::string& _str)
{

  switch (get_string_type(_str)) {
    case (StringType::INTEGER):
      return pybind11::int_(std::stoi(_str));
      break;
    case (StringType::FLOAT):
      return pybind11::float_(std::stof(_str));
      break;
    default:
      return pybind11::str(_str);
  }
}

/** Returns an empty numpy array
 *
 * @param size_t ndims
 * @returns pybind11::array_t<T> empty array
 */
template<typename T,
         typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
pybind11::array_t<T>
create_empty_array(size_t ndims)
{
  std::vector<size_t> strides(ndims, 0);
  std::vector<size_t> shape(ndims, 0);
  return std::move(pybind11::array(shape, strides, (T*)nullptr));
}

/** Convert a 1D c++ vector into a numpy array (pybind stylye)
 *
 * @param std::vector<T> data
 * @return pybind11::array_t<T>
 *
 * Makes a copy in memory!
 */
template<typename T,
         typename std::enable_if<std::is_floating_point<T>::value ||
                                 std::is_integral<T>::value>::type* = nullptr>
pybind11::array_t<T>
vector_to_nparray(const std::vector<T>& _data)
{
  // check for emptiness
  if (_data.size() == 0)
    return std::move(create_empty_array<T>(1));

  // allocate array
  pybind11::array_t<T> _array(_data.size());

  // copy data into array
  std::copy(_data.begin(), _data.end(), (T*)_array.mutable_data());

  return std::move(_array);
}

/** Convert a 2D c++ vector into a numpy array (pybind stylye)
 *
 * @param std::vector< std::vector<T> > data
 * @return pybind11::array_t<T>
 *
 * Makes a copy in memory!
 */
template<typename T,
         typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
pybind11::array_t<T>
vector_to_nparray(const std::vector<std::vector<T>>& _data)
{
  // check for zero rows and columns
  if ((_data.size() == 0) || (_data[0].size() == 0)) {
    return std::move(create_empty_array<T>(2));
  }

  // array allocation
  pybind11::array_t<T> _array({ _data.size(), _data[0].size() });

  size_t pos = 0;
  for (size_t iRow = 0; iRow < _data.size(); ++iRow) {
    // check array shape
    if (iRow > 0 && (_data[iRow].size() != _data[iRow - 1].size()))
      throw(std::invalid_argument("Can not convert vector<vector<T>> to "
                                  "np.ndarray for having improper shape."));

    // copy row
    std::copy(
      _data[iRow].begin(), _data[iRow].end(), (T*)_array.mutable_data() + pos);

    // remember offset for numpy data
    pos += _data[iRow].size();
  }

  return std::move(_array);
}

/** Convert an internal tensor to numpy
 *
 * @param tensor
 * @return ret : numpy array
 */
template<typename T>
pybind11::array_t<T>
tensor_to_nparray_copy(std::shared_ptr<qd::Tensor<T>> tensor)
{

  if (tensor.size() == 0) {
    return std::move(create_empty_array<T>(1));
  }

  auto& data = tensor.get_data();

  pybind11::array_t<T, pybind11::array::c_style> ret;
  ret.resize(tensor.get_shape());
  std::copy(data.begin(), data.end(), ret.mutable_data());

  // do the thing
  return ret;
}

/** Convert an internal tensor to numpy
 *
 * @param tensor
 * @return ret : numpy array
 */
template<typename T>
inline pybind11::object
tensor_to_nparray(std::shared_ptr<qd::Tensor<T>> tensor)
{
  auto tensor_py = pybind11::cast(tensor);
  pybind11::array_t<T> ret(tensor_py);

  return ret;
}

// template<typename T>
// inline pybind11::array
// tensor_to_nparray(std::shared_ptr<qd::Tensor<T>> tensor)
// {
//   auto tensor_py = pybind11::cast(tensor);
//   pybind11::array ret(pybind11::buffer_info(tensor_py));

//   return ret;
// }

} // end:namespace:py
} // end:namespace:qd

#endif