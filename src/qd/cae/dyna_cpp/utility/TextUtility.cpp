
#include "TextUtility.hpp"
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>

/*
 * Extract all integer numbers from a string
 */
std::vector<unsigned int> TextUtility::extract_integers(std::string text){

   std::vector<unsigned int> numbers;

   std::stringstream ss;
   ss << text;
   unsigned int found;
   std::string temp;

   while(getline(ss, temp,' ')) {
      if(std::stringstream(temp)>>found)
      {
         numbers.push_back(found);
      }
   }

   return numbers;

}

/** perform a trim
 * @param string &s : string to trim
 * @return string s : trimmed string
 */
 std::string &TextUtility::rtrim(std::string &s) {
     s.erase(std::find_if(s.rbegin(), s.rend(),
             std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
     return s;
 }

/** perform a left trim
 * @param string &s : string to trim
 * @return string s : trimmed string
 */
std::string &TextUtility::ltrim(std::string &s) {
     s.erase(s.begin(), std::find_if(s.begin(), s.end(),
             std::not1(std::ptr_fun<int, int>(std::isspace))));
     return s;
 }


/** perform a right trim
 * @param string &s : string to trim
 * @return string s : trimmed string
 */
std::string &TextUtility::trim(std::string &s) {
     return ltrim(rtrim(s));
 }
