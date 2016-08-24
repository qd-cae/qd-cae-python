
#ifndef FILEUTILITY_HPP
#define FILEUTILITY_HPP

#include <string>
#include <vector>

using namespace std;

/** Utility class for file related things.
 */
class FileUtility {

public:
  static bool check_ExistanceAndAccess(string);
  static vector<string> globVector(string);
  static vector<string> read_textFile(string filepath);
  static vector<string> findDynaResultFiles(string _base_file);

};

#endif
