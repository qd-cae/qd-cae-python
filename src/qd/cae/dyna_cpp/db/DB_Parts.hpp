
#ifndef DB_PARTS_HPP
#define DB_PARTS_HPP

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

  size_t size();
  void print_parts();
  vector<Part*>& get_parts();
  Part* get_part_byName(string);
  Part* get_part_byID(int _id);
  Part* get_part_byIndex(int _index);
  Part* add_part(int _partIndex, int _partID);
  Part* add_part(int _partID);

};

#endif
