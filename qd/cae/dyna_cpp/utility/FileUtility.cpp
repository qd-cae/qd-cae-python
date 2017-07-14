
#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

#include <boost/algorithm/string/trim.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <string>

#ifdef _WIN32
#include <stdio.h>
#include <tchar.h>
#include <windows.h>

#else
#include "glob.h"
#endif

namespace qd {

/** Read the lines of a text file into a vector
 * @param string filepath : path of the text file
 *
 * throws an expection in case of an IO-Error.
 */
std::vector<std::string>
FileUtility::read_textFile(std::string filepath)
{

  // vars
  std::string linebuffer;
  std::vector<std::string> filebuffer;

  // open stream
  ifstream filestream(filepath.c_str());
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

/* === WINDOWS === */
#ifdef _WIN32

bool
FileUtility::check_ExistanceAndAccess(std::string filepath)
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
FileUtility::globVector(std::string pattern)
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
FileUtility::findDynaResultFiles(std::string _base_filepath)
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
        &&
        string_has_only_numbers(fname,
                                base_filename.size())) // number ending only
      files.push_back(directory + fname);

  } while (FindNextFile(hFind, &FindFileData) != 0);
  FindClose(hFind);

  // Sort files
  sort(files.begin(), files.end());

  return files;
}

/* === LINUX === */
#else

bool
FileUtility::check_ExistanceAndAccess(std::string filepath)
{
  ifstream ifile(filepath.c_str());
  return ifile.good();
}

std::vector<std::string>
FileUtility::globVector(std::string pattern)
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
FileUtility::findDynaResultFiles(std::string _base_filepath)
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
