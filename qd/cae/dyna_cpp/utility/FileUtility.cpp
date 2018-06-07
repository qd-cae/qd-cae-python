
#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

extern "C"
{
#include <stdlib.h>
}

// WINDOWS
#ifdef _WIN32
extern "C"
{
#include <io.h>}
#include <stdio.h>
#include <tchar.h>
#include <windows.h>
}
#define NULL_DEVICE "NUL:"

#else // LINUX
#include "glob.h"
#include <unistd.h>

#define NULL_DEVICE "/dev/null"

#endif

namespace qd {

/** Join filepaths correctly
 *
 * @param _path1 : path to first file
 * @param _path2 : path to secon file
 * @return combined_path : joined path
 */
std::string
join_path(const std::string& _path1, const std::string& _path2)
{
  if (_path1.empty())
    return _path2;

  auto last_char = _path1[_path1.size() - 1];
  if (last_char == '/' || last_char == '\\')
    return _path1 + _path2;
  else
    return _path1 + '/' + _path2;
}

/** Read the lines of a text file into a vector
 * @param filepath : path of the text file
 *
 * throws an expection in case of an IO-Error.
 */
std::vector<std::string>
read_text_file(const std::string& filepath)
{

  // vars
  std::string linebuffer;
  std::vector<std::string> filebuffer;

  // open stream
  std::ifstream filestream(filepath.c_str());
  if (!filestream.is_open())
    throw(std::invalid_argument("Error while opening file " + filepath));

  // read data
  while (getline(filestream, linebuffer)) {
    filebuffer.push_back(linebuffer);
  }

  // Check for Error
  if (filestream.bad()) {
    filestream.close();
    throw(std::invalid_argument("Error during reading file " + filepath));
  }

  // Close file
  filestream.close();

  // return
  return filebuffer;
}

/** Read a binary file into a memory buffer
 *
 * @param filepath : path of the text file
 * @return data : memory buffer
 */
std::vector<char>
read_binary_file(const std::string& filepath)
{

  // load file
  std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
  if (!ifs.is_open())
    throw(std::invalid_argument("Error while opening file " + filepath));

  auto pos = ifs.tellg();
  size_t length = pos;

  std::vector<char> data(length);
  ifs.seekg(0, std::ios::beg);
  ifs.read(&data[0], length);

  // Check for Error
  if (ifs.bad()) {
    ifs.close();
    throw(std::invalid_argument("Error during reading file " + filepath));
  } else {
    ifs.close();
  }

  return data;
}

/** Delete a file
 *
 * @param _path : path to file to delete
 */
void
delete_file(const std::string& _path)
{

  if (remove(_path.c_str()) != 0) {
    throw(std::runtime_error("Deletion of file " + _path + " failed."));
  }
}

/** Save data to a file
 * @param _filepath
 * @param _data
 */
void
save_file(const std::string& _filepath, const std::string& _data)
{

  std::ofstream fs;
  fs.open(_filepath, std::ofstream::binary);
  fs << _data;
  fs.close();
}

/** Compute the entropy of a file buffer
 *
 * @param _buffer : buffer which contains the lines of a text file
 * @return entropy : entropy of the text file
 *
 * The entropy should be between 0 (ordered) and 8 (random).
 */
double
get_entropy(const std::vector<char>& _buffer)
{

  // count bytes
  std::map<char, long long> frequencies;
  // for (const auto& line : _buffer) {
  for (char c : _buffer) {
    frequencies[c]++;
    //++char_count;
  }
  //}

  // calculate entropy
  double nChars = static_cast<double>(_buffer.size());
  double entropy = 0;
  for (std::pair<char, long long> p : frequencies) {
    double freq = static_cast<double>(p.second) / nChars;
    if (freq > 0.)
      entropy += freq * log2(freq);
  }

  return std::abs(entropy);
}

/* === WINDOWS === */
#ifdef _WIN32

static FILE* redirect_stdout = nullptr;
static std::mutex redirect_stdout_lock;

void
enable_stdout()
{
  std::lock_guard<std::mutex> lock(redirect_stdout_lock);

#ifdef QD_DEBUG
  if (redirect_stdout == nullptr)
    throw(std::runtime_error("Error, trying to disable stdout twice!"));
#else
  if (redirect_stdout == nullptr)
    return;
#endif

  if (fclose(redirect_stdout) != 0) {
    redirect_stdout = nullptr;
    throw(std::runtime_error(
      "Error redirecting output from NULL to stdout again."));
  }

  redirect_stdout = nullptr;
  if (!freopen("CON", "w", stdout)) {
    throw(std::runtime_error("Could not re-enable console output."));
  }
}

void
disable_stdout()
{
  std::lock_guard<std::mutex> lock(redirect_stdout_lock);

#ifdef QD_DEBUG
  if (redirect_stdout != nullptr)
    throw(std::runtime_error("Error, trying to disable stdout twice!"));
#else
  if (redirect_stdout != nullptr)
    return;
#endif

  redirect_stdout = freopen(NULL_DEVICE, "w", stdout);
  if (!redirect_stdout) {
    redirect_stdout = nullptr;
    throw(std::runtime_error("Can not redirect IO to NULL."));
  }
}

auto s2ws = [](const std::string& s) {
  int len;
  int slength = (int)s.length() + 1;
  len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
  wchar_t* buf = new wchar_t[len];
  MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
  std::wstring r(buf);
  delete[] buf;
  return r;
};

auto ws2s = [](const std::wstring& text) {
  std::locale const loc("");
  wchar_t const* from = text.c_str();
  std::size_t const len = text.size();
  std::vector<char> buffer(len + 1);
  std::use_facet<std::ctype<wchar_t>>(loc).narrow(
    from, from + len, '_', &buffer[0]);
  return std::string(&buffer[0], &buffer[len]);
};

bool
check_ExistanceAndAccess(const std::string& filepath)
{

  WIN32_FIND_DATA FindFileData;
  HANDLE handle = FindFirstFile(filepath.c_str(), &FindFileData);

  int found = handle != INVALID_HANDLE_VALUE;
  if (found) {
    FindClose(handle);
  }

  if (found != 0)
    return true;
  return false;
}

std::vector<std::string>
glob_vector(const std::string& pattern)
{

  // get file directory
  std::string directory = "";
  size_t pos = pattern.find_last_of("/\\");
  if (pos != std::string::npos)
    directory = pattern.substr(0, pos) + "/";

  // get files
  std::vector<std::string> files;
  WIN32_FIND_DATA FindFileData;
  HANDLE hFind = FindFirstFile(pattern.c_str(), &FindFileData);
  do {
    std::string fname(FindFileData.cFileName);

    files.push_back(directory + fname);

  } while (FindNextFile(hFind, &FindFileData) != 0);
  FindClose(hFind);

  // Sort files
  sort(files.begin(), files.end());

  return files;
}

/** Find dyna result files from the given base filename.
 * @param std::string _base_filepath
 */
std::vector<std::string>
find_dyna_result_files(const std::string& _base_filepath)
{

  // get file directory
  std::string directory = "";
  std::string base_filename = _base_filepath;
  size_t pos = _base_filepath.find_last_of("/\\");
  if (pos != std::string::npos) {
    base_filename = _base_filepath.substr(pos + 1, std::string::npos);
    directory = _base_filepath.substr(0, pos) + "/";
  }

  // get files
  std::vector<std::string> files;
  std::string win32_pattern = std::string(_base_filepath + "*");
  WIN32_FIND_DATA FindFileData;
  HANDLE hFind = FindFirstFileEx(
    win32_pattern.c_str(),
    FindExInfoStandard,
    &FindFileData,
    FindExSearchNameMatch,
    NULL,
    FIND_FIRST_EX_CASE_SENSITIVE); // Searches are case-sensitive.
  do {
    std::string fname(FindFileData.cFileName);
    if ((fname.substr(0, base_filename.size()) ==
         base_filename) // case sensitivity check
        && string_has_only_numbers(fname,
                                   base_filename.size())) // number ending only
      files.push_back(directory + fname);

  } while (FindNextFile(hFind, &FindFileData) != 0);
  FindClose(hFind);

  // Sort files
  sort(files.begin(), files.end());

  return files;
}

} // namespace qd

/* === LINUX === */
#else

static FILE* redirect_stdout = nullptr;
static std::mutex redirect_stdout_lock;

void
SwapIOB(FILE* A, FILE* B)
{

  FILE temp;

  // make a copy of IOB A (usually this is "stdout")
  std::memcpy(&temp, A, sizeof(FILE));

  // copy IOB B to A's location, now any output
  // sent to A is redirected thru B's IOB.
  std::memcpy(A, B, sizeof(FILE));

  // copy A into B, the swap is complete
  std::memcpy(B, &temp, sizeof(FILE));

} // end SwapIOB;

/** Disable stdout
 *
 * This function should be thread safe.
 */
void
disable_stdout()
{

// already redirected, is usually fine for non debugging
#ifdef QD_DEBUG
  if (redirect_stdout != nullptr)
    throw(std::runtime_error("Error, trying to disable stdout twice!"));
#else
  if (redirect_stdout != nullptr)
    return;
#endif

  std::lock_guard<std::mutex> lock(redirect_stdout_lock);

  redirect_stdout = fopen(NULL_DEVICE, "w");
  if (!redirect_stdout) {
    redirect_stdout = nullptr;
    throw(std::runtime_error("Can not redirect IO to NULL."));
  }

  SwapIOB(redirect_stdout, stdout);
}

/** Enables stdout
 *
 * This function should be thread safe.
 */
void
enable_stdout()
{
// already enabled, usually fine but not for debugging
#ifdef QD_DEBUG
  if (redirect_stdout == nullptr)
    throw(std::runtime_error("Error, trying to disable stdout twice!"));
#else
  if (redirect_stdout == nullptr)
    return;
#endif

  std::lock_guard<std::mutex> lock(redirect_stdout_lock);

  SwapIOB(redirect_stdout, stdout);
  if (fclose(redirect_stdout) != 0) {
    redirect_stdout = nullptr;
    throw(std::runtime_error(
      "Error redirecting output from NULL to stdout again."));
  }

  redirect_stdout = nullptr;
}

bool
check_ExistanceAndAccess(const std::string& filepath)
{
  std::ifstream ifile(filepath.c_str());
  return ifile.good();
}

std::vector<std::string>
glob_vector(std::string pattern)
{

  glob_t glob_result;
  glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  std::vector<std::string> files;
  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    files.push_back(std::string(glob_result.gl_pathv[i]));
  }
  globfree(&glob_result);

  sort(files.begin(), files.end());

  return files;
}

std::vector<std::string>
find_dyna_result_files(const std::string& _base_filepath)
{

  std::string pattern = std::string(_base_filepath + "*");
  glob_t glob_result;
  glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  std::vector<std::string> files;
  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    std::string fname(glob_result.gl_pathv[i]);
    if ((fname.substr(0, _base_filepath.size()) == _base_filepath) &&
        string_has_only_numbers(fname, _base_filepath.size()))
      files.push_back(fname);
  }
  globfree(&glob_result);

  sort(files.begin(), files.end());

  return files;
}

} // namespace qd

#endif
