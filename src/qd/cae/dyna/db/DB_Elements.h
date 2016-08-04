
#ifndef DB_ELEMENTS
#define DB_ELEMENTS

// forward declarations
//class Element;
class DB_Nodes;
class DB_Parts;
class D3plot;

// includes
#include <vector>
#include <map>
#include "Element.h"

using namespace std;

class DB_Elements {

  private:
  D3plot* d3plot;
  DB_Nodes* db_nodes;
  DB_Parts* db_parts;
  map<int,Element*> elements2;
  map<int,Element*> elements4;
  map<int,Element*> elements8;
  map<int,Element*> elements2ByIndex;
  map<int,Element*> elements4ByIndex;
  map<int,Element*> elements8ByIndex;

  public:
  DB_Elements(D3plot*, DB_Nodes*,DB_Parts*);
  ~DB_Elements();
  D3plot* get_d3plot();
  DB_Nodes* get_db_nodes();
  Element* add_element(ElementType,int,vector<int>);

  unsigned int size();
  Element* get_elementByID(int,int);
  Element* get_elementByIndex(int,int);

};

#endif
