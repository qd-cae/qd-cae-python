
#ifndef DB_NODES
#define DB_NODES

// forward declarations
class Node;
class FEMFile;
class DB_Elements;

// includes
#include <map>
#include <vector>

// namespaces
using namespace std;

class DB_Nodes {

private:
  FEMFile* femfile;
  map<int,Node*> nodesByIndex; // starts at 1
  map<int,Node*> nodesByID;
  DB_Elements* db_elements;

public:
  DB_Nodes(FEMFile* _femfile);
  ~DB_Nodes();
  unsigned int size();
  FEMFile* get_femfile();
  DB_Elements* get_db_elements();
  void set_db_elements(DB_Elements*);
  Node* add_node(int,vector<float>);
  Node* get_nodeByID(int);
  Node* get_nodeByIndex(int);

};

#endif
