
#ifndef FILEUTILITY
#define FILEUTILITY

#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

class FileUtility {

  public:
  static bool check_ExistanceAndAccess(string);
  static vector<string> globVector(string);

};

#endif
