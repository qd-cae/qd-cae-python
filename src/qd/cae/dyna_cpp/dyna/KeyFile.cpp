
#include <iostream>
#include <fstream>
#include <stdexcept>      // std::invalid_argument
#include "../utility/BoostException.hpp"
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/trim.hpp>
#include "../utility/FileUtility.hpp"
#include "../utility/TextUtility.hpp"
#include "../db/DB_Nodes.hpp"
#include "../db/DB_Elements.hpp"
#include "../db/DB_Parts.hpp"
#include "../db/Element.hpp"
#include "../db/Part.hpp"
#include "KeyFile.hpp"

using namespace boost::algorithm;

/**
 * Constructor for a LS-Dyna input file.
 */
KeyFile::KeyFile() {

}



/** Constructor for reading a LS-Dyna input file.
 *
 * @param string filepath : filepath of a key file to read
 */
KeyFile::KeyFile(string _filepath)
   : FEMFile(_filepath) {

   // Read the mesh
   this->read_mesh(this->get_filepath());

}


/** Read the mesh from the file given in the filepath
 *
 */
void KeyFile::read_mesh(string _filepath){

   // Read the lines
   #ifdef QD_DEBUG
   cout << "Reading teyt file into buffer ... " << flush;
   #endif
   vector<string> lines = FileUtility::read_textFile(_filepath);
   #ifdef QD_DEBUG
   cout << "done." << endl;
   #endif

   // Get databases
   DB_Parts* db_parts = this->get_db_parts();
   DB_Nodes* db_nodes = this->get_db_nodes();
   DB_Elements* db_elements = this->get_db_elements();

   // Time to do the thing
   bool nodesection = false;
   bool elemsection = false;
   bool elemsection_solid = false;
   bool elemsection_beam = false;
   //bool elemthicksection = false;
   bool partsection = false;
   //bool propsection = false;
   //bool propsection_title = false;

   string line;
   vector<float> coords(3);
   vector<int> elemNodes_beam(2);
   vector<int> elemNodes_shell(4);
   vector<int> elemNodes_solid(8);
   int id;
   int partID;
   string title;
   size_t iCardLine = 0;


   #ifdef QD_DEBUG
   cout << "Parsing Text File ... " << endl;
   #endif
   for(vector<string>::size_type iLine = 0; iLine != lines.size(); iLine++) {

      // Remove comments, etc
      line = preprocess_string_dyna(lines[iLine]);

      // Skip empty lines
      if( trim_copy(line).empty() )
         line = string("");

      // TODO Line preprocessing
      // Remove comment stuff behind "$"

      /* NODES */
      if(trim_copy(line) == "*NODE"){
         nodesection = true;
         #ifdef QD_DEBUG
         cout << "Starting *NODE in line: " << (iLine+1) << endl;
         #endif
      } else if(nodesection & (line.find('*') == string::npos) & (!line.empty()) ){

         try {
            coords[0] = boost::lexical_cast<float>(trim_copy(line.substr(8,16)));
            coords[1] = boost::lexical_cast<float>(trim_copy(line.substr(24,16)));
            coords[2] = boost::lexical_cast<float>(trim_copy(line.substr(40,16)));
            db_nodes->add_node(boost::lexical_cast<int>(trim_copy(line.substr(0,8))),coords);
         } catch (const std::exception& ex){
            cerr << "Error reading node in line " << (iLine+1) << ":" << ex.what() << endl;
            nodesection = false;
         } catch (const string& ex) {
            cerr << "Error reading node in line " << (iLine+1) << ":" << ex << endl;
            nodesection = false;
         } catch (...) {
            cerr << "Error reading node in line " << (iLine+1) << ": Unknown error." << endl;
            nodesection = false;
         }
      } else if( nodesection & ((line.find('*') != string::npos) | line.empty()) ){
         nodesection = false;
         #ifdef QD_DEBUG
         cout << "*NODE finished in line: " << (iLine+1) << endl;
         #endif
      }


      /* ELEMENTS */
      if(trim_copy(line) == "*ELEMENT_SHELL"){
         elemsection = true;
         #ifdef QD_DEBUG
         cout << "Starting *ELEMENT_SHELL in line: " << (iLine+1) << endl;
         #endif
      } else if(elemsection & (line.find('*') == string::npos) & (!line.empty()) ){

         try {
            id = boost::lexical_cast<int>(trim_copy(line.substr(0,8)));
            partID = boost::lexical_cast<int>(trim_copy(line.substr(8,8)));
            elemNodes_shell[0] = boost::lexical_cast<int>(trim_copy(line.substr(16,8)));
            elemNodes_shell[1] = boost::lexical_cast<int>(trim_copy(line.substr(24,8)));
            elemNodes_shell[2] = boost::lexical_cast<int>(trim_copy(line.substr(32,8)));
            elemNodes_shell[3] = boost::lexical_cast<int>(trim_copy(line.substr(40,8)));
            db_elements->add_element_byKeyFile(SHELL, id, partID, elemNodes_shell);
         } catch (const std::exception& ex){
            cerr << "Error reading element in line " << (iLine+1) << ":" << ex.what() << endl;
            elemsection = false;
         } catch (const string& ex) {
            cerr << "Error reading element in line " << (iLine+1) << ":" << ex << endl;
            elemsection = false;
         } catch (...) {
            cerr << "Error reading element in line " << (iLine+1) << ": Unknown error." << endl;
            elemsection = false;
         }
      } else if( elemsection &  ((line.find('*') != string::npos) | line.empty()) ){
         elemsection = false;
         #ifdef QD_DEBUG
         cout << "*ELEMENT_SHELL finished in line: " << (iLine+1) << endl;
         #endif
      }


      /* ELEMENTS */
      if(trim_copy(line) == "*ELEMENT_SOLID"){
         elemsection_solid = true;
         iCardLine = 0;
         #ifdef QD_DEBUG
         cout << "Starting *ELEMENT_SOLID in line: " << (iLine+1) << endl;
         #endif
      } else if(elemsection_solid & (line.find('*') == string::npos) & (!line.empty()) ){

         try {

            if(iCardLine == 0){

               id = boost::lexical_cast<int>(trim_copy(line.substr(0,8)));
               partID = boost::lexical_cast<int>(trim_copy(line.substr(8,8)));
               ++iCardLine;

            } else if(iCardLine == 1){

               elemNodes_solid[0] = boost::lexical_cast<int>(trim_copy(line.substr(0,8)));
               elemNodes_solid[1] = boost::lexical_cast<int>(trim_copy(line.substr(8,8)));
               elemNodes_solid[2] = boost::lexical_cast<int>(trim_copy(line.substr(16,8)));
               elemNodes_solid[3] = boost::lexical_cast<int>(trim_copy(line.substr(24,8)));
               elemNodes_solid[4] = boost::lexical_cast<int>(trim_copy(line.substr(32,8)));
               elemNodes_solid[5] = boost::lexical_cast<int>(trim_copy(line.substr(40,8)));
               elemNodes_solid[6] = boost::lexical_cast<int>(trim_copy(line.substr(48,8)));
               elemNodes_solid[7] = boost::lexical_cast<int>(trim_copy(line.substr(56,8)));
               db_elements->add_element_byKeyFile(SOLID, id, partID, elemNodes_solid);
               iCardLine = 0;

            }

         } catch (const std::exception& ex){
            cerr << "Error reading element in line " << (iLine+1) << ":" << ex.what() << endl;
            elemsection_solid = false;
         } catch (const string& ex) {
            cerr << "Error reading element in line " << (iLine+1) << ":" << ex << endl;
            elemsection_solid = false;
         } catch (...) {
            cerr << "Error reading element in line " << (iLine+1) << ": Unknown error." << endl;
            elemsection_solid = false;
         }


      } else if( elemsection_solid &  ((line.find('*') != string::npos) | line.empty()) ){
         elemsection_solid = false;
         #ifdef QD_DEBUG
         cout << "*ELEMENT_SOLID finished in line: " << (iLine+1) << endl;
         #endif
      }


      // BEAMS
      if(trim_copy(line).substr(0,string("*ELEMENT_BEAM").size()) == "*ELEMENT_BEAM"){
         elemsection_beam = true;
         iCardLine = 0;
         #ifdef QD_DEBUG
         cout << "Starting *ELEMENT_BEAM in line: " << (iLine+1) << endl;
         #endif
      } else if(elemsection_beam & (line.find('*') == string::npos) & (!line.empty()) ){

         try {

            if(iCardLine == 0){

               id = boost::lexical_cast<int>(trim_copy(line.substr(0,8)));
               partID = boost::lexical_cast<int>(trim_copy(line.substr(8,8)));
               elemNodes_beam[0] = boost::lexical_cast<int>(trim_copy(line.substr(16,8)));
               elemNodes_beam[1] = boost::lexical_cast<int>(trim_copy(line.substr(24,8)));
               db_elements->add_element_byKeyFile(BEAM, id, partID, elemNodes_beam);
               ++iCardLine;

            } else if(iCardLine == 1){
               iCardLine = 0;
            }

         } catch (const std::exception& ex){
            cerr << "Error reading element in line " << (iLine+1) << ":" << ex.what() << endl;
            elemsection_beam = false;
         } catch (const string& ex) {
            cerr << "Error reading element in line " << (iLine+1) << ":" << ex << endl;
            elemsection_beam = false;
         } catch (...) {
            cerr << "Error reading element in line " << (iLine+1) << ": Unknown error." << endl;
            elemsection_beam = false;
         }


      } else if( elemsection_beam &  ((line.find('*') != string::npos) | line.empty()) ){
         elemsection_beam = false;
         #ifdef QD_DEBUG
         cout << "*ELEMENT_BEAM finished in line: " << (iLine+1) << endl;
         #endif
      }


      /* PART */
      if(trim_copy(line).substr(0,5) == "*PART"){

         partsection = true;
         #ifdef QD_DEBUG
         cout << "Starting *PART in line: " << (iLine+1) << endl;
         #endif
         iCardLine = 0;

      } else if(partsection & (line.find('*') == string::npos) & (!line.empty()) ){

         if( iCardLine == 0 ){
            title = trim_copy(line);
            ++iCardLine;
         } else if( iCardLine == 1 ) {

            try {

               id = boost::lexical_cast<int>(trim_copy(line.substr(0,10)));
               Part* part = db_parts->get_part_byID(id);
               if(part == NULL){
                  part = db_parts->add_part_byID(id);
               }
               part->set_name(title);
               ++iCardLine;

            } catch (const std::exception& ex){
               cerr << "Error reading part in line " << (iLine+1) << ":" << ex.what() << endl;
               partsection = false;
            } catch (const string& ex) {
               cerr << "Error reading part in line " << (iLine+1) << ":" << ex << endl;
               partsection = false;
            } catch (...) {
               cerr << "Error reading part in line " << (iLine+1) << ": Unknown error." << endl;
               partsection = false;
            }
         }

      } else if( partsection &  ((line.find('*') != string::npos) | (iCardLine > 1)) ){
         partsection = false;
         #ifdef QD_DEBUG
         cout << "*PART finished in line: " << (iLine+1) << endl;
         #endif
      }



   } // for lines
   #ifdef QD_DEBUG
   cout << "parsing of text-file done." << endl;
   #endif


}
