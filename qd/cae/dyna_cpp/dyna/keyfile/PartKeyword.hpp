
#ifndef PARTKEYWORD_HPP
#define PARTKEYWORD_HPP

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <dyna_cpp/db/DB_Parts.hpp>
#include <dyna_cpp/dyna/keyfile/Keyword.hpp>

namespace qd {

class PartKeyword : public Keyword
{
private:
  DB_Parts* db_parts;
  std::vector<int32_t> part_ids;
  std::vector<std::string> comments_between_card0_and_card1;
  std::vector<std::string> unparsed_data;
  std::vector<std::string> trailing_lines;

public:
  explicit PartKeyword(DB_Parts* _db_parts,
                       const std::vector<std::string>& _lines,
                       int64_t _iLine);
  void load();
  template<typename T>
  std::shared_ptr<Part> add_part(T _part_id, const std::string& _name = "");
  template<typename T>
  std::shared_ptr<Part> add_part(
    T _part_id,
    const std::string& _name = "",
    const std::vector<std::string>& _additional_lines =
      std::vector<std::string>());

  template<typename T>
  inline std::shared_ptr<Part> get_partByIndex(T _part_index);
  inline std::vector<std::shared_ptr<Part>> get_parts();
  inline size_t get_nParts() const;
  std::string str() override;
};

/** Add a part to the keyword
 *
 * @param _part_id
 * @param _name
 * @return part
 */
template<typename T>
std::shared_ptr<Part>
PartKeyword::add_part(T _part_id, const std::string& _name)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  auto part_id_i32 = static_cast<int32_t>(_part_id);
  auto part = db_parts->add_partByID(part_id_i32, _name);
  part_ids.push_back(part_id_i32);
  comments_between_card0_and_card1.push_back("");
  unparsed_data.push_back("");
  return part;
}

/** Add a part to the keyword
 *
 * @param _part_id
 * @param _name
 * @param _additional_card_data : additional missing card information as string
 * @return part
 */
template<typename T>
std::shared_ptr<Part>
PartKeyword::add_part(T _part_id,
                      const std::string& _name,
                      const std::vector<std::string>& _additional_lines)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  auto part_id_i32 = static_cast<int32_t>(_part_id);
  auto part = db_parts->add_partByID(part_id_i32, _name);
  part_ids.push_back(part_id_i32);

  if (_additional_lines.empty())
    lines.push_back(std::string(field_size, ' '));
  else {
    lines.push_back(std::string(field_size, ' ') + _additional_lines[0]);
    for (size_t iAdditionalLine = 1; iAdditionalLine < _additional_lines.size();
         ++iAdditionalLine)
      lines.push_back(_additional_lines[iAdditionalLine]);
  }

  return part;
}

/** Get a part by index
 *
 * @param _part_index
 * @return part
 */
template<typename T>
std::shared_ptr<Part>
PartKeyword::get_partByIndex(T _part_index)
{
  _part_index = index_treatment(_part_index, part_ids.size());
  if (static_cast<size_t>(_part_index) > part_ids.size())
    throw(std::invalid_argument("Part index out of bounds for part keyword."));
  return this->db_parts->get_partByID(part_ids[_part_index]);
}

/** Get all the parts in the keyword
 *
 * @return parts
 */
std::vector<std::shared_ptr<Part>>
PartKeyword::get_parts()
{
  return db_parts->get_partByID(part_ids);
}

/** Get the number of parts in the keyword
 *
 * @return nParts
 */
size_t
PartKeyword::get_nParts() const
{
  return part_ids.size();
}

} // namespace:qd

#endif