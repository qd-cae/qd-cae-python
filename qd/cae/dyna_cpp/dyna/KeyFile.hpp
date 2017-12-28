
#ifndef KEYFILE_HPP
#define KEYFILE_HPP

// includes
#include <dyna_cpp/db/FEMFile.hpp>
#include <dyna_cpp/dyna/Keyword.hpp>

#include <map>
#include <stdexcept>
#include <string>

// forward declarations
class D3plot;

namespace qd {

/**
 * This is a class for reading LS-Dyna input files.The mesh will be parsed with
 * it's properties, currently only in a limited way.
 */
class KeyFile : public FEMFile
{
public:
  enum class KeywordType
  {
    NONE,
    NODE,
    ELEMENT_BEAM,
    ELEMENT_SHELL,
    ELEMENT_SOLID,
    PART,
    INCLUDE,
    COMMENT,
    GENERIC
  };

private:
  bool load_includes;
  double encryption_detection_threshold;
  std::vector<std::shared_ptr<KeyFile>> includes;
  std::map<std::string, std::vector<std::shared_ptr<Keyword>>> keywords;

  void read_mesh(const std::string& _filepath);
  void parse_file(const std::string& _filepath);

public:
  KeyFile();
  KeyFile(const std::string& _filepath,
          bool _load_includes = true,
          double _encryption_detection = 0.7);
};

} // namespace qd

#endif
