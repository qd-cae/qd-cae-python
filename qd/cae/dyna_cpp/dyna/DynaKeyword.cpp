
#include "DynaKeyword.hpp"
#include "../utility/BoostException.hpp"
#include "../utility/TextUtility.hpp"
#include <boost/algorithm/string/trim.hpp>

using namespace std;
using namespace boost::algorithm;

/** Constructor for a dyna keyword
 * @param string name : keyword e.g. "*SECTION_SHELL"
 */
DynaKeyword::DynaKeyword(const string& _keyword_name){

   this->init();
   this->keyword_name =_keyword_name;
}


/** Initialize the variables of the class with default vars.
 */
void DynaKeyword::init(){
   this->nEmptyLines = 0;
}


/** Set the title of the keyword
 * @param string name : title of the keyword
 */
void DynaKeyword::set_title(const string& _title){
   this->title = _title;
}


/** Parse a card row from a keyFile
 * @param string text : text string of the row
 * @param size_t iCardRow : text string of the row.
 *
 * Parse a keyword line belonging to this DynaKeyword. The constructor shall
 * be invoked with the keywords name itself e.g. *SECTION_SHELL. The next
 * following lines may be given to this routine.
 */
void DynaKeyword::parse_keyfile_row(string _line, const size_t iCardRow){


   string line_trimmed = boost::algorithm::trim_copy(_line);

   // values present
   if( !line_trimmed.empty() ){

      // save the card entries
      this->rows.insert( pair<size_t, string >(iCardRow,line_trimmed) );

   // no values
   } else {
      ++this->nEmptyLines;
      return;
   }

}


/** Get a card value
 * @param size_t iRow : row index
 * @param size_t iCol : column index
 * @return string& value : card value as a string
 *
 * Throws if iRow or iCol was too high. We just can't know how many empty lines
 * to read without specifying it for every keyword. So by default we assume
 * each empty line may mean something, which is ok when using a preprocessor
 * which cancels out empty spaces.
 */
string DynaKeyword::get_card_row(const size_t iRow){

   map<size_t, string >::iterator it = this->rows.find(iRow);

   // there is a saved line with iRow
   if( it != this->rows.end() ){

      return it->second;

   // line was empty BUT within empty line count
   // maybe there was a value, but it was empty/defaulted
   } else if ( iRow < this->nEmptyLines ) {

      return string(); // yay return empty

   // There was no card with iRow index
   } else {
      throw(string("iRow=")+to_string(iRow)+string(" was out of scope of empty line counter after keyword, so there definitely was no default value present."));
   }

}

/** Get hte number of accessible card rows
 *  @return size_t nRows
 */
const size_t DynaKeyword::get_nCardRows(){
   return this->nEmptyLines;
}
