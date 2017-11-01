
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

namespace qd {

template<typename T>
class Tensor
{
private:
  std::vector<size_t> shape;
  std::vector<T> data;
  size_t get_offset(const std::vector<size_t> indexes);

public:
  Tensor();
  Tensor(std::initializer_list<size_t> list);
  void set(const std::vector<size_t>& indexes, T value);
  void set(std::initializer_list<size_t> indexes, T value);
  void resize(const std::vector<size_t>& _new_shape);
  void push_back(T _value);
  size_t size() const;
  std::vector<T>& get_data();
  std::vector<size_t> get_shape() const;
  void print() const;
};

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
  : shape(list)
  , data(std::accumulate(begin(list),
                         end(list),
                         static_cast<size_t>(1),
                         std::multiplies<>()))
{}

/** Compute the array offset from indexes
 *
 * @param indexes : indexes of the tensor
 * @return entry_index : index of entry in 1D data array
 */
template<typename T>
inline size_t
Tensor<T>::get_offset(const std::vector<size_t> indexes)
{
  if (data.size() == 0)
    throw(std::invalid_argument("Tensor index too large."));

  if (indexes.size() != shape.size())
    throw(std::invalid_argument(
      "Tensor index dimension different from tensor dimension."));

  size_t entry_index = 0;
  for (size_t ii = 0; ii < indexes.size(); ++ii) {
    size_t offset = std::accumulate(
      begin(shape) + ii + 1, end(shape), 1, std::multiplies<>());
    entry_index += indexes[ii] * offset;
  }

  if (entry_index > data.size())
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
  data[this->get_offset(indexes)] = value;
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
  return this->data.size();
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
                                         std::multiplies<>());
  this->data.resize(_new_data_len);
  this->shape = _new_shape;
}

/** push a new value into the back of the data buffer
 *
 * @param _value : new value to add
 */
template<typename T>
inline void
Tensor<T>::push_back(T _value)
{
  this->data.push_back(_value);
}

/** Get the underlying buffer of the tensor
 *
 * @return data : 1-dimensional data buffer
 */
template<typename T>
inline std::vector<T>&
Tensor<T>::get_data()
{
  return this->data;
};

/** Get the shape of a tensor
 *
 * @return shape : shape of the tensor
 */
template<typename T>
inline std::vector<size_t>
Tensor<T>::get_shape() const
{
  return this->shape;
}

/** Print the linear memory of the tensor
 */
template<typename T>
void
Tensor<T>::print() const
{
  for (const auto entry : data)
    std::cout << entry << " ";
  std::cout << std::endl;
}

} // namespace qd

#endif