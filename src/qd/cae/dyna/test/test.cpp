
#include "../dyna/d3plot.h"
#include "../db/DB_Nodes.h"
#include "../db/DB_Elements.h"
#include "../db/Node.h"
#include "../db/Element.h"
#include <iostream>
#include <vector>
#include <set>

using namespace std;

int main(){

  //string filename = "./../test/crashbox/crashbox";
  string filename = "./../test/drop_tower.fz";
  //string filename = "./../test/crshbox.fz"; // error?

  // Reading
  try{

    D3plot* d3plot = new D3plot(filename,false);

    d3plot->read_states("vel");

    // Tests
    // Element
    cout << endl << "> TESTS" << endl;
    Element* element = d3plot->get_db_elements()->get_elementByID(SHELL,9770);
    cout << "Elem-ID:" << element->get_elementID() << endl;

    set<Node*> nodes = element->get_nodes();
    for(auto node : nodes) {
      cout << "Node-ID:" << node->get_nodeID() << endl;
      cout << "X:" << node->get_coords()[0] << endl;
      cout << "Y:" << node->get_coords()[1] << endl;
      cout << "Z:" << node->get_coords()[2] << endl;
    }
    /*
    cout << "elem-function 0:";
    for(unsigned int ii=0; ii < element->get_strain().size(); ii++)
      cout << element->get_strain()[ii][0] << " ";
    cout << endl << endl;
    */

    // Node
    Node* node = d3plot->get_db_nodes()->get_nodeByID(9950);
    cout << "Node-ID:" << node->get_nodeID() << endl;
    set<Element*> elements = node->get_elements();
    for(auto _element : elements){
      cout << "Element-ID:" << _element->get_elementID() << endl;
    }
    cout << "Vel:";
    for(unsigned int ii=0; ii < node->get_vel().size(); ii++)
      cout << node->get_vel()[ii][2] << " ";
    cout << endl << endl;


    cout << "Testrun successful." << endl << endl;
    getchar();
  } catch (const char* e){
    cout << e << endl;
  } catch (string e){
    cout << e << endl;
  }

  return 0;

}
