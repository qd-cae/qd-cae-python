
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

#include <dyna_cpp/utility/debug.hpp>

namespace qd {

template<typename T>
class Tensor : public traced<Tensor<T>>
{
private:
  std::vector<size_t> _shape;
  std::vector<T> _data;
  size_t get_offset(const std::vector<size_t> indexes);

public:
  Tensor();
  Tensor(std::initializer_list<size_t> list);
  Tensor(std::vector<size_t> list, const T* data);
  const std::vector<size_t>& get_shape() const;
  void set(const std::vector<size_t>& indexes, T value);
  void set(std::initializer_list<size_t> indexes, T value);
  void resize(const std::vector<size_t>& _new_shape);
  void reshape(const std::vector<size_t>& new_shape);
  void push_back(T _value);

  size_t size() const;
  std::vector<T>& get_data() &;
  std::vector<T>&& move_data() &&;
  void shrink_to_fit();
  void reserve(size_t n_elements);

  void print() const;
};

template<typename T>
using Tensor_ptr = std::shared_ptr<Tensor<T>>;

/** Create an empty tensor
 *
 */
template<typename T>
Tensor<T>::Tensor()
{}

/** Create a tensor from an initializer list.
 *
 * @param list : shape of the tensor given by initializer list
 */
template<typename T>
Tensor<T>::Tensor(std::initializer_list<size_t> list)
  : _shape(list)
  , _data(std::accumulate(begin(list),
                          end(list),
                          static_cast<size_t>(1),
                          std::multiplies<size_t>()))
{}

/** Create a tensor from a shape and data
 *
 * @param list : shape of the tensor given by initializer list
 */
template<typename T>
Tensor<T>::Tensor(std::vector<size_t> list, const T* data)
  : _shape(list)
  , _data(std::accumulate(std::begin(list),
                          std::end(list),
                          static_cast<size_t>(1),
                          std::multiplies<size_t>()))
{
  std::copy(data, data + _data.size(), _data.begin());
}

/** Compute the array offset from indexes
 *
 * @param indexes : indexes of the tensor
 * @return entry_index : index of entry in 1D data array
 */
template<typename T>
inline size_t
Tensor<T>::get_offset(const std::vector<size_t> indexes)
{
  if (_data.size() == 0)
    throw(std::invalid_argument("Tensor index too large."));

  if (indexes.size() != _shape.size())
    throw(std::invalid_argument(
      "Tensor index dimension different from tensor dimension."));

  size_t entry_index = 0;
  for (size_t ii = 0; ii < indexes.size(); ++ii) {
    size_t offset = std::accumulate(std::begin(_shape) + ii + 1,
                                    std::end(_shape),
                                    1,
                                    std::multiplies<size_t>());
    entry_index += indexes[ii] * offset;
  }

  if (entry_index > _data.size())
    throw(std::invalid_argument("Tensor index too large."));

  return entry_index;
}

/** set an entry in the tensor
 *
 * @param _indexes : indexes of the entry given by a vector
 * @param value : value to set
 *
 */
template<typename T>
inline void
Tensor<T>::set(const std::vector<size_t>& indexes, T value)
{
  _data[this->get_offset(indexes)] = value;
}

/** set an entry in the tensor
 *
 * @param _indexes : index list initializer style
 * @param value : value to set
 *
 * Example: tensor.set({1,2,3},4)
 */
template<typename T>
inline void
Tensor<T>::set(std::initializer_list<size_t> _indexes, T value)
{
  std::vector<size_t> indexes(_indexes);
  this->set(indexes, value);
}

/** Get the size of the data vector
 *
 * @return size : number of elements in the data vector
 */
template<typename T>
inline size_t
Tensor<T>::size() const
{
  return _data.size();
}

/** Resize a tensor
 *
 * @param _new_shape : new shape
 */
template<typename T>
void
Tensor<T>::resize(const std::vector<size_t>& _new_shape)
{
  size_t _new_data_len = std::accumulate(begin(_new_shape),
                                         end(_new_shape),
                                         static_cast<size_t>(1),
                                         std::multiplies<size_t>());
  _data.resize(_new_data_len);
  _shape = _new_shape;
}

/** Reshaoe the tensor
 *
 * @param new_shape
 *
 * Checks for correct shape.
 */
template<typename T>
void
Tensor<T>::reshape(const std::vector<size_t>& new_shape)
{
  size_t new_data_len = std::accumulate(begin(new_shape),
                                        end(new_shape),
                                        static_cast<size_t>(1),
                                        std::multiplies<size_t>());

  if (_data.size() != new_data_len)
    throw(
      std::invalid_argument("Tensor reshape does not match container size."));

  _shape = new_shape;
}

/** push a new value into the back of the data buffer
 *
 * @param _value : new value to add
 */
template<typename T>
inline void
Tensor<T>::push_back(T value)
{
  _data.push_back(value);
}

/** Get the underlying buffer of the tensor
 *
 * @return data : 1-dimensional data buffer
 */
template<typename T>
inline std::vector<T>&
Tensor<T>::get_data() &
{
  return _data;
};

/** Move the underlying data buffer out of the tensor
 *
 * @return data : 1-dimensional data buffer
 */
template<typename T>
inline std::vector<T>&&
Tensor<T>::move_data() &&
{
  _shape.clear();
  return std::move(_data);
}

/** Get the shape of a tensor
 *
 * @return shape : shape of the tensor
 */
template<typename T>
inline const std::vector<size_t>&
Tensor<T>::get_shape() const
{
  return _shape;
}

/** Shrink the data buffer to fit the data
 *
 */
template<typename T>
inline void
Tensor<T>::shrink_to_fit()
{
  _data.shrink_to_fit();
}

/** Reserve enough space for n_elements
 */
template<typename T>
inline void
Tensor<T>::reserve(size_t n_elements)
{
  _data.reserve(n_elements);
}

/** Print the linear memory of the tensor
 */
template<typename T>
void
Tensor<T>::print() const
{
  for (const auto entry : _data)
    std::cout << entry << " ";
  std::cout << std::endl;
}

} // namespace qd

#endif
