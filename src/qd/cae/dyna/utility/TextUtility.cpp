
#include "TextUtility.h"

/*
 * Extract all integer numbers from a string
 */
std::vector<unsigned int> TextUtility::extract_integers(std::string text){
   
   std::vector<unsigned int> numbers;
   
   std::stringstream ss;
   ss << text;
   unsigned int found;
   std::string temp;
   
   while(std::getline(ss, temp,' ')) {
      if(std::stringstream(temp)>>found)
      {
         numbers.push_back(found);
         //std::cout<<found<<std::endl;
      }
   }
   
   return numbers;
  
}