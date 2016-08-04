
#include "FileUtility.h"
#include <iostream>
#include <stdlib.h>
#include <string>
#include <algorithm>
#ifdef _WIN32
	#include <windows.h>
	#include <tchar.h>
	#include <stdio.h>
#else
	#include "glob.h"
#endif


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

#endif
