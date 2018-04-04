
#ifndef INCLUDEKEYWORD_HPP
#define INCLUDEKEYWORD_HPP

#include <string>
#include <vector>

#include <dyna_cpp/dyna/keyfile/Keyword.hpp>

namespace qd {

// forward declaration
class KeyFile;

class IncludeKeyword : public Keyword
{
private:
  KeyFile* parent_kf;
  std::vector<std::shared_ptr<KeyFile>> includes;

  std::vector<std::string> trailing_lines;
  std::vector<std::string> unresolved_filepaths; // for writing l8ter

public:
  IncludeKeyword(KeyFile* _parent_kf,
                 const std::vector<std::string> _lines,
                 int64_t _iLine);
  void load();
  void load(bool _load_mesh);

  // getters
  inline std::vector<std::shared_ptr<KeyFile>>& get_includes();

  std::string str() override;
};

/** Get all includes in the include keyword
 *
 * @return includes : KeyFile objects of the include
 */
std::vector<std::shared_ptr<KeyFile>>&
IncludeKeyword::get_includes()
{
  return includes;
}

} // namespace:qd

#endif