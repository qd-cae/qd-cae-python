
#ifndef CONTAINER_HPP
#define CONTAINER_HPP

#include <algorithm>
#include <vector>


namespace qd {

/** Remove multiple vector indexes fast
 *
 * @param vector
 * @param to_remove
 * @param sort : sort to_remove (if not done already)
 *
 * The idnexes to remove must be sorted!
 */
template<typename T, typename U>
void
vector_remove_indexes(std::vector<T>& vector,
                      std::vector<U> to_remove,
                      bool sort = true)
{
  static_assert(std::is_integral<U>::value, "indexes must be integral.");

  // sort and make unique
  if (sort) {
    std::sort(to_remove.begin(), to_remove.end());
    to_remove.erase(std::unique(to_remove.begin(), to_remove.end()),
                    to_remove.end());
  }

  auto vector_base = vector.begin();
  size_t down_by = 0;

  for (auto iter = to_remove.cbegin(); iter < to_remove.cend();
       iter++, down_by++) {
    size_t next = (iter + 1 == to_remove.cend() ? vector.size() : *(iter + 1));

    std::move(vector_base + *iter + 1,
              vector_base + next,
              vector_base + *iter - down_by);
  }
  vector.resize(vector.size() - to_remove.size());
}

} // namespace qd

#endif
