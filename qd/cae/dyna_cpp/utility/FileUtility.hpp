
#ifndef FILEUTILITY_HPP
#define FILEUTILITY_HPP

#include <string>
#include <vector>

namespace qd {

bool check_ExistanceAndAccess(std::string);

std::vector<std::string> globVector(std::string);

std::vector<std::string>
read_textFile(std::string _filepath);

std::vector<std::string>
findDynaResultFiles(std::string _base_file);

void
delete_file(const std::string& _path);

} // namespace qd

#endif
