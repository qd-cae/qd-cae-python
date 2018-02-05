
#ifndef INCLUDEKEYWORD_HPP
#define INCLUDEKEYWORD_HPP

#include <string>
#include <vector>

#include <dyna_cpp/dyna/KeyFile.hpp>
#include <dyna_cpp/dyna/Keyword.hpp>

namespace qd {

class IncludeKeyword : public Keyword
{
private:
  KeyFile* parent_db;
  std::vector<std::shared_ptr<KeyFile>> includes;

public:
  IncludeKeyword(KeyFile* parent_db,
                 const std::vector<std::string> lines,
                 int64_t _iLine);
  void resolve();

  // getters
  inline std::vector<std::shared_ptr<KeyFile>> get_includes();
};

} // namespace:qd

#endif