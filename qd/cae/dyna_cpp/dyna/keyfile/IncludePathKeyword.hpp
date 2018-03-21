
#ifndef INCLUDEPATHKEYWORD_HPP
#define INCLUDEPATHKEYWORD_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <dyna_cpp/dyna/keyfile/Keyword.hpp>

namespace qd {

class IncludePathKeyword : public Keyword
{
public:
  IncludePathKeyword(const std::vector<std::string> _lines, int64_t _iLine);

  bool is_relative() const;
  std::vector<std::string> get_include_dirs();
};

} // namespace:qd

#endif