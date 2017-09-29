
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
  std::vector<long> shape;
  std::vector<T> data;

public:
  Tensor(std::initializer_list<size_t> list);
  void set(std::initializer_list<size_t> indexes, T value);
  void print() const;
};

template<typename T>
Tensor<T>::Tensor(std::initializer_list<size_t> list)
  : shape(list.size())
  , data(std::accumulate(begin(list), end(list), 1, std::multiplies<>()))
{
}

template<typename T>
inline void
Tensor<T>::set(std::initializer_list<size_t> indexes, T value)
{
  if (indexes.size() != shape.size())
    throw(std::invalid_argument(
      "Tensor index dimension different from tensor dimension."));
  // data[ii + shape[2] * (jj + shape[1] * kk)] = value; // row major C
  // data[ii + shape[0] * (jj + shape[1] * kk)] = T; // col major F
}

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