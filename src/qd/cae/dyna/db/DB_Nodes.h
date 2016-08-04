
#ifndef DB_NODES
#define DB_NODES

// forward declarations
class Node;
class D3plot;
class DB_Elements;

// includes
#include <map>
#include <vector>

using namespace std;

class DB_Nodes {

  private:
  D3plot* d3plot;
  map<int,Node*> nodesByIndex; // starts at 1
  map<int,Node*> nodesByID;
  DB_Elements* db_elements;

  public:
  DB_Nodes(D3plot* d3plot);
  ~DB_Nodes();
  unsigned int size();
  D3plot* get_d3plot();
  DB_Elements* get_db_elements();
  void set_db_elements(DB_Elements*);
  Node* add_node(int,vector<float>);
  Node* get_nodeByID(int);
  Node* get_nodeByIndex(int);

};

#endif
