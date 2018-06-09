
#ifndef FILEUTILITY_HPP
#define FILEUTILITY_HPP

#include <cstdio>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace qd {

bool
check_ExistanceAndAccess(const std::string&);

std::string
join_path(const std::string& _path1, const std::string& _path2);

std::vector<std::string>
glob_vector(const std::string&);

std::vector<std::string>
read_text_file(const std::string& _filepath);

std::vector<char>
read_binary_file(const std::string& _filepath);

std::vector<std::string>
find_dyna_result_files(const std::string& _base_file);

double
get_entropy(const std::vector<char>& _buffer);

void
delete_file(const std::string& _path);

void
save_file(const std::string& _filepath, const std::string& _data);

void
disable_stdout();

void
enable_stdout();

} // namespace qd

#endif
