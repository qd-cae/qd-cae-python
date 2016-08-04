
#ifndef TEXTUTILITY
#define TEXTUTILITY

#include <vector>
#include <string>
#include <sstream>

/*
 * Convert some type into a string.
 */
template <typename T>
std::string to_string(T value)
{
	std::ostringstream os ;
	os << value ;
	return os.str() ;
}

/*
 * Text utility class
 */
class TextUtility {
   
   public:
   static std::vector<unsigned int> extract_integers(std::string text);
   
};


#endif