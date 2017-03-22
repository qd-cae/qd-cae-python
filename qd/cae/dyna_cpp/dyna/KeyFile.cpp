
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

using namespace std;
using namespace boost::algorithm;

// Enumeration
namespace Keyword{
   enum Keyword {NONE,
                 NODE,
                 ELEMENT_BEAM,
                 ELEMENT_SHELL,
                 ELEMENT_SOLID,
                 PART,
                 INCLUDE};
}



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

   #ifdef QD_DEBUG
   cout << " === Parsing File: " << _filepath << endl;
   #endif

   // File directory for Includes
   string directory = "";
   size_t pos = _filepath.find_last_of("/\\");
   if (pos != string::npos)
	   directory = _filepath.substr(0,pos) + "/";
   #ifdef QD_DEBUG
   cout << "Basic directory for *INCLUDE: " << directory << endl;
   #endif

   // Read the lines
   #ifdef QD_DEBUG
   cout << "Filling IO-Buffer ... " << flush;
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
   Keyword::Keyword keyword = Keyword::NONE;
   string line;
   string line_trimmed;
   vector<float> coords(3);
   vector<int> elemNodes_beam(2);
   vector<int> elemNodes_shell(4);
   vector<int> elemNodes_solid(8);
   int id;
   int partID = -1;
   string title;
   size_t iCardLine = 0;
   bool line_has_keyword = false;

   for(vector<string>::size_type iLine = 0; iLine != lines.size(); iLine++) {

      // Remove comments, etc
      //line = preprocess_string_dyna(lines[iLine]);
      line = lines[iLine];

      // Skip empty lines
      if( line[0] == '$' )
         continue;

      line_trimmed = trim_copy(line);
      line_has_keyword = (line_trimmed.find('*') != string::npos);

      /* INCLUDE */
      if(line_trimmed == "*INCLUDE"){
         keyword = Keyword::INCLUDE;
         #ifdef QD_DEBUG
         cout << "*INCLUDE in line: " << (iLine+1) << endl;
         #endif
      } else if ( keyword == Keyword::INCLUDE ){
         this->read_mesh(directory+line_trimmed); // basic directory is this file's
         keyword = Keyword::NONE;
      }


      /* NODES */
      if(line_trimmed == "*NODE"){

         keyword = Keyword::NODE;
         #ifdef QD_DEBUG
         cout << "Starting *NODE in line: " << (iLine+1) << endl;
         #endif

      } else if( (keyword==Keyword::NODE)
               && !line_has_keyword
               && (!line_trimmed.empty()) ){

         try {
            coords[0] = boost::lexical_cast<float>(trim_copy(line.substr(8,16)));
            coords[1] = boost::lexical_cast<float>(trim_copy(line.substr(24,16)));
            coords[2] = boost::lexical_cast<float>(trim_copy(line.substr(40,16)));
            db_nodes->add_node(boost::lexical_cast<int>(trim_copy(line.substr(0,8))),coords);
         } catch (const std::exception& ex){
            cerr << "Error reading node in line " << (iLine+1) << ":" << ex.what() << endl;
            keyword = Keyword::NODE;
         } catch (const string& ex) {
            cerr << "Error reading node in line " << (iLine+1) << ":" << ex << endl;
            keyword = Keyword::NODE;
         } catch (...) {
            cerr << "Error reading node in line " << (iLine+1) << ": Unknown error." << endl;
            keyword = Keyword::NODE;
         }
      } else if( (keyword==Keyword::NODE)
               && (line_has_keyword | line.empty()) ){

         keyword = Keyword::NONE;
         #ifdef QD_DEBUG
         cout << "*NODE finished in line: " << (iLine+1) << endl;
         #endif
      }


      /* ELEMENTS */
      if(line_trimmed == "*ELEMENT_SHELL"){
         keyword = Keyword::ELEMENT_SHELL;
         #ifdef QD_DEBUG
         cout << "Starting *ELEMENT_SHELL in line: " << (iLine+1) << endl;
         #endif
      } else if( (keyword == Keyword::ELEMENT_SHELL )
               && !line_has_keyword
               && (!line.empty()) ){

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
            keyword = Keyword::NONE;
         } catch (const string& ex) {
            cerr << "Error reading element in line " << (iLine+1) << ":" << ex << endl;
            keyword = Keyword::NONE;
         } catch (...) {
            cerr << "Error reading element in line " << (iLine+1) << ": Unknown error." << endl;
            keyword = Keyword::NONE;
         }
      } else if( (keyword == Keyword::ELEMENT_SHELL)
              &&  (line_has_keyword | line.empty()) ){

         keyword = Keyword::NONE;
         #ifdef QD_DEBUG
         cout << "*ELEMENT_SHELL finished in line: " << (iLine+1) << endl;
         #endif
      }


      /* ELEMENTS */
      if(line_trimmed == "*ELEMENT_SOLID"){
         keyword = Keyword::ELEMENT_SOLID;
         iCardLine = 0;
         #ifdef QD_DEBUG
         cout << "Starting *ELEMENT_SOLID in line: " << (iLine+1) << endl;
         #endif
      } else if( (keyword == Keyword::ELEMENT_SOLID)
               && !line_has_keyword
               && !line.empty() ){

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
            keyword = Keyword::NONE;
         } catch (const string& ex) {
            cerr << "Error reading element in line " << (iLine+1) << ":" << ex << endl;
            keyword = Keyword::NONE;
         } catch (...) {
            cerr << "Error reading element in line " << (iLine+1) << ": Unknown error." << endl;
            keyword = Keyword::NONE;
         }


      } else if( (keyword == Keyword::ELEMENT_SOLID)
              &&  (line_has_keyword | line.empty()) ){
         keyword = Keyword::NONE;
         #ifdef QD_DEBUG
         cout << "*ELEMENT_SOLID finished in line: " << (iLine+1) << endl;
         #endif
      }


      // BEAMS
      if(line_trimmed.substr(0,string("*ELEMENT_BEAM").size()) == "*ELEMENT_BEAM"){
         keyword = Keyword::ELEMENT_BEAM;
         iCardLine = 0;
         #ifdef QD_DEBUG
         cout << "Starting *ELEMENT_BEAM in line: " << (iLine+1) << endl;
         #endif
      } else if( (keyword == Keyword::ELEMENT_BEAM)
               && !line_has_keyword
               && (!line.empty()) ){

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
            keyword = Keyword::ELEMENT_BEAM;
         } catch (const string& ex) {
            cerr << "Error reading element in line " << (iLine+1) << ":" << ex << endl;
            keyword = Keyword::ELEMENT_BEAM;
         } catch (...) {
            cerr << "Error reading element in line " << (iLine+1) << ": Unknown error." << endl;
            keyword = Keyword::ELEMENT_BEAM;
         }


      } else if( (keyword == Keyword::ELEMENT_BEAM)
              &&  (line_has_keyword | line.empty()) ){
         keyword = Keyword::ELEMENT_BEAM;
         #ifdef QD_DEBUG
         cout << "*ELEMENT_BEAM finished in line: " << (iLine+1) << endl;
         #endif
      }


      /* PART */
      if(line_trimmed.substr(0,5) == "*PART"){

         keyword = Keyword::PART;
         #ifdef QD_DEBUG
         cout << "Starting *PART in line: " << (iLine+1) << endl;
         #endif
         iCardLine = 0;

      } else if( (keyword == Keyword::PART)
               && !line_has_keyword
               && (!line.empty()) ){

         if( iCardLine == 0 ){
            title = line_trimmed;
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
               keyword = Keyword::NONE;
            } catch (const string& ex) {
               cerr << "Error reading part in line " << (iLine+1) << ":" << ex << endl;
               keyword = Keyword::NONE;
            } catch (...) {
               cerr << "Error reading part in line " << (iLine+1) << ": Unknown error." << endl;
               keyword = Keyword::NONE;
            }

         }

      } else if( (keyword == Keyword::PART)
               && ( line_has_keyword | (iCardLine > 1)) ){

         keyword = Keyword::NONE;
         #ifdef QD_DEBUG
         cout << "*PART finished in line: " << (iLine+1) << endl;
         #endif
      }



   } // for lines
   #ifdef QD_DEBUG
   cout << "parsing of text-file done." << endl;
   #endif


}
