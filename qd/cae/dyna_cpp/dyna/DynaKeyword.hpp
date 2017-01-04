
#ifndef DYNAKEYWORD_HPP
#define DYNAKEYWORD_HPP

#include <string>
#include <vector>
#include <map>

// forward declarations
// None

using namespace std;

class DynaKeyword {

private:
   string keyword_name;
   string title;

   size_t nEmptyLines; // this count is remembered as some limit

   map< size_t, string > rows; // ...

   void init();

public:
   DynaKeyword(const string& _keyword_name);
   void set_title(const string& _title);
   void parse_keyfile_row(string _line, const size_t iCardRow);
   string get_card_row(const size_t iRow);
   const size_t get_nCardRows();
   const size_t get_nCardCols();

};

#endif
