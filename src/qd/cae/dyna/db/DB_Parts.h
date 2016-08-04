
#ifndef DB_PARTS
#define DB_PARTS

#include <string>
#include <vector>
#include <map>

class Part;

using namespace std;

class DB_Parts {

  private:
  map<int,Part*> parts;
  map<int,Part*> partsByIndex;

  public:
  DB_Parts();
  ~DB_Parts();

  unsigned int size();
  void print_parts();
  vector<Part*> get_parts();
  Part* get_part_byName(string);
  Part* get_part_byID(int);
  Part* get_part_byIndex(int);
  Part* add_part(int,int);

};

#endif
