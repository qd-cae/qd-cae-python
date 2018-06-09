
#ifndef DYNAKEYWORD_HPP
#define DYNAKEYWORD_HPP

#include <map>
#include <string>
#include <vector>

namespace qd {

class DynaKeyword
{

private:
  std::string keyword_name;
  std::string title;

  size_t nEmptyLines; // this count is remembered as some limit

  std::map<size_t, std::string> rows; // ...

public:
  DynaKeyword(const std::string& _keyword_name);
  void set_title(const std::string& _title);
  void parse_keyfile_row(std::string _line, const size_t iCardRow);
  std::string get_card_row(const size_t iRow);
  const size_t get_nCardRows();
  const size_t get_nCardCols();
};

} // namespace qd

#endif
