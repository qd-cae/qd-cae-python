
#include "DB_Parts.hpp"
#include "FEMFile.hpp"
#include "Part.hpp"

using namespace std;

/**
 * Constructor
 */
DB_Parts::DB_Parts(FEMFile* _femfile)
  : femfile( _femfile ){

}


/**
 * Destructor
 */
DB_Parts::~DB_Parts(){

}


/** Create a part with it's id. The index is just size + 1.
 */
Part* DB_Parts::add_part_byID(int _partID, const std::string& name){

  #ifdef QD_DEBUG
  const auto& it = id2index.find(_partID);
  if( it != id2index.end() )
    throw(string("Trying to insert a part with same ID twice into the part-db!"));
  #endif

  unique_ptr<Part> part(new Part(_partID, name, this->femfile));
  this->parts.push_back( std::move(part) );
  this->id2index.insert( pair<int,size_t>(_partID, this->parts.size()-1) );
  return this->parts.back().get();

}

/**
 * Get the parts in the db in a vector.
 */
vector<Part*> DB_Parts::get_parts(){

	vector<Part*> ret(this->parts.size());
  for(const auto& part_ptr : parts){
    ret.push_back( part_ptr.get() );
  }
	
	return std::move(ret);

}


/**
 * Get a part by it's name.
 */
Part* DB_Parts::get_part_byName(const string& _name){

  for(auto& part_ptr : parts){
    if(part_ptr->get_name().compare(_name) == 0){
      return part_ptr.get();
    }
  }

  return nullptr;

}


/**
 * Get the number of parts in the database.
 */
size_t DB_Parts::size() const {
  
  #ifdef QD_DEBUG
  if(parts.size() != id2index.size())
    throw(string("Part Map and Index-Vector have unequal sizes."));
  #endif

  return parts.size();
}


/**
 * Print the parts in the db.
 */
void DB_Parts::print_parts() const {

  for(const auto& part_ptr : parts){
	  cout << "partID:" << part_ptr->get_partID()
	       << " name:" << part_ptr->get_name()
		     << " nElems:" << part_ptr->get_elements().size()
		     << endl;
  }

}
