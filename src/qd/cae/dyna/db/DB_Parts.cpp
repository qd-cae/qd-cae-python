
#include "DB_Parts.h"
#include "Part.h"

/**
 * Constructor
 */
DB_Parts::DB_Parts(){

  //this->parts = map<int,Part*>();

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
Part* DB_Parts::add_part(int partIndex, int partID){

  Part* part = new Part(partID,"");
  this->parts.insert(pair<int,Part*>(partID,part));
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
 * Get a part by it's partID.
 */
Part* DB_Parts::get_part_byID(int partID){

  map<int,Part*>::iterator it = this->parts.find(partID);

  // Part existing
  if(it != parts.end()){
    return it->second;
  }

  // Part creation
  return NULL;

}


/**
 * Get a part by it's part index.
 */
Part* DB_Parts::get_part_byIndex(int partIndex){

  map<int,Part*>::iterator it = this->partsByIndex.find(partIndex);
  
  // Part existing
  if(it != this->partsByIndex.end()){
    return it->second;
  }

  cout << "DB_Parts::get_part_byIndex 2" << endl;
  
  // Part not found
  return NULL;

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
unsigned int DB_Parts::size(){
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
