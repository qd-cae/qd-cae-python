

// includes
#include "../dyna/D3plot.hpp"
#include "../db/Node.hpp"
#include "../db/Element.hpp"
#include "../db/Part.hpp"
#include "../utility/BoostException.hpp"
#include <vector>
#include <string>
#include <boost/python.hpp>

// namespaces
using namespace std;
namespace py = boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(D3plot_overloads, D3plot, 1, 3)

// wrapper
BOOST_PYTHON_MODULE(dyna_cpp)
{
   py::class_<D3plot>("D3plot",py::init<string,py::list,bool>());
   //py::class_<D3plot>("D3plot",py::init<string,py::list,bool>())
//   py::class_<D3plot>("D3plot",py::init<string,py::list,bool>())
//       .def(py::init<string, py::optional<py::list, bool> >());
   //py::class_<D3plot>("D3plot",py::no_init)
       .def(py::init<string, py::optional<py::list,bool> >());
   /*
    def("name", function_ptr);
    def("name", function_ptr, call_policies);
    def("name", function_ptr, "documentation string");
    def("name", function_ptr, call_policies, "documentation string");
    */
}
