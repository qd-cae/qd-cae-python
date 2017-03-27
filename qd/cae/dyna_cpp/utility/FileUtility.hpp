
#ifndef FILEUTILITY_HPP
#define FILEUTILITY_HPP

#include <string>
#include <vector>

/** Utility class for file related things.
 */
class FileUtility {

public:
  static bool check_ExistanceAndAccess(std::string);
  static std::vector<std::string> globVector(std::string);
  static std::vector<std::string> read_textFile(std::string _filepath);
  static std::vector<std::string> findDynaResultFiles(std::string _base_file);

};

#endif
