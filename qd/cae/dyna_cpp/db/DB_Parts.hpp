
#ifndef DB_PARTS_HPP
#define DB_PARTS_HPP

#include <string>
#include <vector>
#include <map>

class Part;
class FEMFile;

using namespace std;

class DB_Parts {

private:
  map<int,Part*> parts;
  map<int,Part*> partsByIndex;
  FEMFile* femfile;

public:
  DB_Parts(FEMFile* _femfile);
  ~DB_Parts();

  size_t size();
  void print_parts();
  vector<Part*> get_parts();
  Part* get_part_byName(string);
  template<typename T>
  Part* get_part_byID(T _id);
  template<typename T>
  Part* get_part_byIndex(T _index);
  Part* add_part(int _partIndex, int _partID);
  Part* add_part_byID(int _partID);

};


template<typename T>
Part* DB_Parts::get_part_byID(T _id){

  map<int,Part*>::iterator it = this->parts.find(_id);

  // Part existing
  if(it != parts.end()){
    return it->second;
  } else {
    // :(
    return NULL;
  }

}


template<typename T>
Part* DB_Parts::get_part_byIndex(T _index){

  map<int,Part*>::iterator it = this->partsByIndex.find(_index);

  // Part existing
  if(it != this->partsByIndex.end()){
    return it->second;
  } else {
    // :(
    return NULL;
  }

}
#endif
