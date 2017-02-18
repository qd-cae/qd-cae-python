
#include "DB_Parts.hpp"
#include "Part.hpp"

/**
 * Constructor
 */
DB_Parts::DB_Parts(){

}


/**
 * Destructor
 */
DB_Parts::~DB_Parts(){

  // Delete Parts
  for (map<int,Part*>::iterator it=parts.begin(); it!=parts.end(); ++it){
    delete it->second;
  }

}


/**
 * Create a part with it's index and id.
 */
Part* DB_Parts::add_part(int _partIndex, int _partID){

  Part* part = new Part(_partID,"");
  this->parts.insert(pair<int,Part*>(_partID,part));
  this->partsByIndex.insert(pair<int,Part*>(_partIndex,part));
  return part;

}

/** Create a part with it's id. The index is just size + 1.
 */
Part* DB_Parts::add_part_byID(int _partID){

  Part* part = new Part(_partID,"");
  int partIndex = this->parts.size()+1;
  this->parts.insert(pair<int,Part*>(_partID,part));
  this->partsByIndex.insert(pair<int,Part*>(partIndex,part));
  return part;

}

/**
 * Get the parts in the db in a vector.
 */
vector<Part*> DB_Parts::get_parts(){

	vector<Part*> ret(this->parts.size());
	for (map<int,Part*>::iterator it=this->parts.begin(); it!=this->parts.end(); ++it) {
		ret.push_back(it->second);
      cout << it->second << endl; // DEBUG
	}
	return ret;

}


/**
 * Get a part by it's name.
 */
Part* DB_Parts::get_part_byName(string _name){

  for (map<int,Part*>::iterator it=this->parts.begin(); it!=this->parts.end(); ++it) {

    if(it->second->get_name().compare(_name) == 0){
      return it->second;
    }
  }

  return NULL;

}


/**
 * Get the number of parts in the database.
 */
size_t DB_Parts::size(){
  if(parts.size() != partsByIndex.size())
    throw("Part Maps: Id-Map and Index-Map have unequal sizes.");
  return parts.size();
}


/**
 * Print the parts in the db.
 */
void DB_Parts::print_parts(){

  for (map<int,Part*>::iterator it=this->parts.begin(); it!=this->parts.end(); ++it) {
	cout << "partID:" << it->second->get_partID()
	     << " name:" << it->second->get_name()
		 << " nElems:" << it->second->get_elements().size()
		 << endl;
  }

}
