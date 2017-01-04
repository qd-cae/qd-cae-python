
#ifndef TEXTUTILITY_HPP
#define TEXTUTILITY_HPP

// includes
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>


/** Convert some type into a string.
 * @param T value : value to convert to string
 * @return string result
 */
template <typename T>
std::string to_string(T const & value)
{
	std::ostringstream os ;
	os << value ;
	return os.str() ;
}

/** Convert string into some type
* @param T value : value to convert to string
* @return string result
*/
template <typename T>
T string_to_type(std::string const & str){
   T value;
   std::stringstream ss(str);
   ss >> value;
   return value;
}


/** Extract numbers from a string
 * @param std::string _text
 */
template <typename T>
std::vector<T> extract_integers(const std::string& text){

   std::vector<T> numbers;

   std::stringstream ss;
   ss << text;
   T found;
   std::string temp;

   while(getline(ss, temp,' ')) {
      if(std::stringstream(temp)>>found)
      {
         numbers.push_back(found);
      }
   }

   return numbers;

}


/** Check if a string has only numbers
 * @param std::string _text : strig to check
 * @param size_t _pos = 0 : starting position
 *
 * Also returns true if the string is empty.
 */
 static bool string_has_only_numbers(const std::string& _text, size_t start_pos){

    // Check
    if(_text.size() == start_pos){
       return true;
    } else if(start_pos > _text.size()){
       throw(std::string("Starting pos is greater than the string length."));
    }
    if(_text.empty())
      return true;

    // Hardcoded stuff rulzZz
    const char number_chars[] = {'0','1','2','3','4','5','6','7','8','9'};

    // iterate
    bool char_is_number = false;
    for(size_t ii = start_pos; ii < _text.size(); ++ii){
       char_is_number = false;
       for(size_t jj = 0; jj < sizeof(number_chars)/sizeof(number_chars[0]); ++jj){
          if(_text[ii] == number_chars[jj] ){
             char_is_number = true;
             break;
          }
       }

       // return if one char was not a number
       if(!char_is_number)
          return false;
    }

    // Return true in case loop went through
    return true;

 }


/** Preprocess a string for dyna.
 * @param string s : string to preprocess
 * @return string& s : string with removed comments
 */
inline std::string preprocess_string_dyna(std::string _text){

   size_t pos = _text.find('$');
   if(pos != std::string::npos)
      return _text.substr(0,pos);
   return _text;

}


#endif
