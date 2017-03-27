
#ifndef BOOSTUTILITY_HPP
#define BOOSTUTILITY_HPP

// includes
#include <vector>
#include <string>
#include <typeinfo>
#include <boost/python.hpp>
#include "dyna_cpp/utility/TextUtility.hpp"

// namespaces
namespace py = boost::python;

/** Convert a python object to a cpp type.
 *
 * @param boost::python::object _obj : object to translate
 * @return T ret
 */
 /*
template<typename T>
T pyType_to_cppType(py::object _obj){

   py::extract<T> get_cppType(_obj);
   if(get_cppType.check()){
      return get_cppType();
   } else {
      throw(string("Can not convert type:")+py::extract<string>(_obj.attr("__name__"))()+string(" to ")+typeid(T).name())
   }
}
*/

/** Convert a boost::python::list to a std::vector.
 *
 * @param boost::python::list _list : python list
 * @return vector<T> vec : converted python list
 */
template<typename T>
std::vector<T> list_to_vector(py::list& _list){

   std::vector<T> vec;
   for(size_t ii=0; ii < len(_list); ++ii){
         vec.push_back(py::extract<T>(_list[ii]));
   }
   return vec;
}

#endif
