
#ifndef DB_ELEMENTS_HPP
#define DB_ELEMENTS_HPP

// forward declarations
//class Element;
class DB_Nodes;
class DB_Parts;
class FEMFile;

// includes
#include <vector>
#include <map>
#include "Element.hpp"
#include "../utility/TextUtility.hpp"

using namespace std;

class DB_Elements {

private:
  FEMFile* femfile;
  DB_Nodes* db_nodes;
  DB_Parts* db_parts;

  /*
  vector<int> index2id_elements2;
  vector<int> index2id_elements4;
  vector<int> index2id_elements8;
  */
  map<int,size_t> id2index_elements2;
  map<int,size_t> id2index_elements4;
  map<int,size_t> id2index_elements8;
  vector<Element*> elements2;
  vector<Element*> elements4;
  vector<Element*> elements8;
  /*
  map<int,Element*> elements2;
  map<int,Element*> elements4;
  map<int,Element*> elements8;
  */

public:
  DB_Elements(FEMFile* _femfile);
  ~DB_Elements();
  FEMFile* get_femfile();
  DB_Nodes* get_db_nodes();
  Element* add_element_byD3plot(ElementType _eType, int _id, vector<int> _elem_data);
  //Element* add_element_byID(ElementType _eType,int _id, int _partid, vector<int> _node_ids)
  Element* add_element_byKeyFile(ElementType _eType, int _id, int _partid, vector<int> _node_ids);

  size_t size(ElementType _type = NONE);
  void reserve(const ElementType _type, const size_t _size);
  template <typename T>
  Element* get_elementByID(ElementType _eType, T _id);
  template <typename T>
  Element* get_elementByIndex(ElementType _eType, T _index);

};


/** Get the element by it's internal id and it's type.
 *
 * @param int _elementType : see description
 * @param int _elementID : id of the element (like in the solver)
 * @return Element* _element
 *
 * Type may be: Element.ElementType
 * NONE = 0 -> error
 * BEAM = 1
 * SHELL = 2
 * SOLID = 3
 */
template <typename T>
Element* DB_Elements::get_elementByID(ElementType _elementType,T _elementID){

  if(_elementType == BEAM){
    map<int,size_t>::iterator it = this->id2index_elements2.find(_elementID);
    if(it == id2index_elements2.end())
      return NULL;
    return elements2[it->second];

  } else if(_elementType == SHELL){
    map<int,size_t>::iterator it = this->id2index_elements4.find(_elementID);
    if(it == id2index_elements4.end())
      return NULL;
    return elements4[it->second];

  } else if(_elementType == SOLID){
    map<int,size_t>::iterator it = this->id2index_elements8.find(_elementID);
    if(it == id2index_elements8.end())
      return NULL;
    return elements8[it->second];

  }

  throw(string("Can not get element with elementType:")+to_string(_elementID));

}

/** Get the element by it's internal index and it's type.
 *
 * @param int _elementType : see description
 * @param int _elementIndex : index of the element (not id!)
 * @return Element* _element
 *
 * Type may be: Element.ElementType
 * NONE = 0 -> error
 * BEAM = 1
 * SHELL = 2
 * SOLID = 3
 */
template <typename T>
Element* DB_Elements::get_elementByIndex(ElementType _elementType,T _elementIndex){

   if(_elementType == BEAM){
     if(_elementIndex < elements2.size())
       return elements2[_elementIndex];
     return NULL;

   } else if(_elementType == SHELL){
     if(_elementIndex < elements4.size())
       return elements4[_elementIndex];
     return NULL;

   } else if(_elementType == SOLID){
     if(_elementIndex < elements8.size())
       return elements8[_elementIndex];
     return NULL;

   }

  throw("Can not get element with elementType:"+to_string(_elementIndex));

}



#endif
