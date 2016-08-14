
#ifndef KEYFILE_HPP
#define KEYFILE_HPP

// includes
#include <string>
#include "../db/FEMFile.hpp"

// forward declarations
class D3plot;

// namespaces
using namespace std;

/**
 * This is a class for reading LS-Dyna input files.The mesh will be parsed with
 * it's properties, currently only in a limited way.
 */
class KeyFile : public FEMFile {

private:
   void read_mesh(string _filepath);

public:
   KeyFile();
   KeyFile(string _filepath);
   bool is_d3plot(){return false;};
   bool is_keyFile(){return true;};
   D3plot* get_d3plot(){throw(string("A KeyFile can not be cast to a D3plot."));};
   KeyFile* get_keyFile(){return this;};

};

#endif
