
#include "FileUtility.hpp"
#include "TextUtility.hpp"
#include <iostream>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <boost/algorithm/string/trim.hpp>
#ifdef _WIN32
	#include <windows.h>
	#include <tchar.h>
	#include <stdio.h>
#else
	#include "glob.h"
#endif


/** Read the lines of a text file into a vector
 * @param string filepath : path of the text file
 *
 * throws an expection in case of an IO-Error.
 */
vector<string> FileUtility::read_textFile(string filepath){

   // vars
   string linebuffer;
   vector<string> filebuffer;

   // open stream
   ifstream filestream(filepath.c_str());
   if (! filestream.is_open())
      throw(string("Error while opening file ")+filepath);

   // read data
   while(getline(filestream, linebuffer)) {
      filebuffer.push_back(linebuffer);
   }

   // Check for Error
   if (filestream.bad()){
      filestream.close();
      throw(string("Error during reading file ")+filepath);
   }

   // Close file
   filestream.close();

   // return
   return filebuffer;

}

/** Read a KeyFile's lines into a buffer. Includes will be catched
 * @param string filepath : path of the text file
 *
 * throws an expection in case of an IO-Error.
 */
vector<string> FileUtility::read_keyFile(string filepath){

   // vars
   string linebuffer;
   vector<string> filebuffer;

   // open stream
   ifstream filestream(filepath.c_str());
   if (! filestream.is_open())
      throw(string("Error while opening file "+filepath));

   // read data
   string edited_line;
   bool include_detected = false;
   bool load_include = false ;
   string include_filepath;
   while(getline(filestream, linebuffer)) {

      edited_line = boost::algorithm::trim_copy(preprocess_string_dyna(edited_line));

      // check for include statement
      if(edited_line.substr(0,8) == "*INCLUDE"){
         include_detected = true;
         continue;
      }

      // parse filepath
      if(include_detected){
         include_filepath = edited_line;
         load_include = true;
         include_detected = false;
         #ifdef QD_DEBUG
         cout << "Loading include file:" << include_filepath << endl;
         #endif
         continue;
      }

      // load include
      if(load_include & !include_filepath.empty() ){
         vector<string> include_lines = FileUtility::read_keyFile(include_filepath);
         filebuffer.insert(filebuffer.end(),include_lines.begin(),include_lines.end());
         load_include = false;
         #ifdef QD_DEBUG
         cout << "Include loaded." << endl;
         #endif
         continue;
      }

      // If nothing special happened, just save the line.
      filebuffer.push_back(linebuffer);
   }

   // Check for Error
   if (filestream.bad()){
      filestream.close();
      throw(string("Error during reading file ")+filepath);
   }

   // Close file
   filestream.close();

   // return
   return filebuffer;

}



#ifdef _WIN32

bool FileUtility::check_ExistanceAndAccess(string filepath){


	WIN32_FIND_DATA FindFileData;
	HANDLE handle = FindFirstFile(filepath.c_str(), &FindFileData) ;

	int found = handle != INVALID_HANDLE_VALUE;
	if(found)
	{
		FindClose(handle);
	}

	if(found != 0)
		return true;
	return false;
}


vector<string> FileUtility::globVector(string pattern){

	// get file directory
  string directory = "";
  size_t pos = pattern.find_last_of("/\\");
  if (pos != string::npos)
	  directory = pattern.substr(0,pos) + "/";

	// get files
	vector<string> files;
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = FindFirstFile(pattern.c_str(), &FindFileData);
	do
	{
		string fname(FindFileData.cFileName);

		files.push_back(directory+fname);

	} while(FindNextFile(hFind, &FindFileData) != 0);
	FindClose(hFind);

	// Sort files
	sort(files.begin(), files.end());

	return files;

}

/** Find dyna result files from the given base filename.
 * @param string _base_filepath
 */
vector<string> FileUtility::findDynaResultFiles(string _base_filepath){

   // get file directory
  string directory = "";
  size_t pos = _base_filepath.find_last_of("/\\");
  if (pos != string::npos)
	  directory = _base_filepath.substr(0,pos) + "/";

	// get files
   vector<string> files;
   string win32_pattern = string(_base_filepath+"*");
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = FindFirstFileEx(win32_pattern.c_str(),
                                  FindExInfoStandard,
                                  &FindFileData,
                                  FindExSearchNameMatch,
                                  NULL,
                                  FIND_FIRST_EX_CASE_SENSITIVE); // Searches are case-sensitive.
   do
	{
		string fname(FindFileData.cFileName);
      if(fname.substr(0,_base_filepath.size()) == _base_filepath) // case sensitivity check
         files.push_back(directory+fname);

	} while(FindNextFile(hFind, &FindFileData) != 0);
	FindClose(hFind);

	// Sort files
	sort(files.begin(), files.end());

   return files;

}

// linux
#else

bool FileUtility::check_ExistanceAndAccess(string filepath){
  ifstream ifile(filepath.c_str());
  return ifile.good();
}

vector<string> FileUtility::globVector(string pattern){

  glob_t glob_result;
  glob(pattern.c_str(),GLOB_TILDE,NULL,&glob_result);
  vector<string> files;
  for(unsigned int i=0;i<glob_result.gl_pathc;++i){
      files.push_back(string(glob_result.gl_pathv[i]));
  }
  globfree(&glob_result);

  sort(files.begin(), files.end());

  return files;
}


vector<string> FileUtility::findDynaResultFiles(string _base_filepath){

   string pattern = string(_base_filepath+"*")
   glob_t glob_result;
   glob(pattern.c_str(),GLOB_TILDE,NULL,&glob_result);
   vector<string> files;
   for(unsigned int i=0;i<glob_result.gl_pathc;++i){
      if(string(glob_result.gl_pathv[i]).substr(0,_base_filepath.size()) == _base_filepath )
         files.push_back(string(glob_result.gl_pathv[i]));
   }
   globfree(&glob_result);

   sort(files.begin(), files.end());

   return files;
}

#endif
