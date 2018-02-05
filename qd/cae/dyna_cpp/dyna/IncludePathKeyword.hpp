
#ifndef INCLUDEPATHKEYWORD_HPP
#define INCLUDEPATHKEYWORD_HPP

#include <dyna_cpp/dyna/KeyFile.hpp>
#include <dyna_cpp/dyna/Keyword.hpp>

namespace qd {

class IncludePathKeyword : public Keyword
{
public:
  IncludePathKeyword(const std::vector<std::string> _lines, int64_t _iLine);
  std::vector<std::string> get_include_dirs();
};

} // namespace:qd

#endif