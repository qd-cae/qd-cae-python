
#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/db/FEMFile.hpp>
#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/db/Part.hpp>
//#include <dyna_cpp/dyna/Binout.hpp>
#include <dyna_cpp/dyna/D3plot.hpp>
#include <dyna_cpp/dyna/KeyFile.hpp>
#include <dyna_cpp/dyna/RawD3plot.hpp>
#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/PythonUtility.hpp>

extern "C" {
#include <pybind11/numpy.h>
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <numpy/arrayobject.h>

#include <memory>
#include <string>
#include <vector>

using namespace pybind11::literals;

// get the docstrings
#include <dyna_cpp/python_api/docstrings.cpp>

/* This hack ensures translation from vector of objects to
 * python list of objects.
 *
 * Should also overwrite internal reference handling, so that
 * every instance itself references its' d3plot. This is
 * neccesary because if the d3plot is deallocated, some
 * functions will not work anymore.
 * (node -> node.get_elements() -> elem.get_nodes())
 */
// DEBUG: Somehow not working correctly ...
using ListCasterNodes =
  pybind11::detail::list_caster<std::vector<std::shared_ptr<qd::Node>>,
                                std::shared_ptr<qd::Node>>;
using ListCasterElements =
  pybind11::detail::list_caster<std::vector<std::shared_ptr<qd::Element>>,
                                std::shared_ptr<qd::Element>>;
using ListCasterParts =
  pybind11::detail::list_caster<std::vector<std::shared_ptr<qd::Part>>,
                                std::shared_ptr<qd::Part>>;
namespace pybind11 {
namespace detail {

// vector of nodes -> list of python nodes
template<>
struct type_caster<std::vector<std::shared_ptr<qd::Node>>> : ListCasterNodes
{
  static handle cast(const std::vector<std::shared_ptr<qd::Node>>& src,
                     return_value_policy,
                     handle parent)
  {
    return ListCasterNodes::cast(
      src, return_value_policy::reference_internal, parent);
  }
  static handle cast(const std::vector<std::shared_ptr<qd::Node>>* src,
                     return_value_policy pol,
                     handle parent)
  {
    return cast(*src, pol, parent);
  }
};

// vector of elements -> list of python elements
template<>
struct type_caster<std::vector<std::shared_ptr<qd::Element>>>
  : ListCasterElements
{
  static handle cast(const std::vector<std::shared_ptr<qd::Element>>& src,
                     return_value_policy,
                     handle parent)
  {
    return ListCasterElements::cast(
      src, return_value_policy::reference_internal, parent);
  }
  static handle cast(const std::vector<std::shared_ptr<qd::Element>>* src,
                     return_value_policy pol,
                     handle parent)
  {
    return cast(*src, pol, parent);
  }
};

// vector of parts -> list of python parts
template<>
struct type_caster<std::vector<std::shared_ptr<qd::Part>>> : ListCasterParts
{
  static handle cast(const std::vector<std::shared_ptr<qd::Part>>& src,
                     return_value_policy,
                     handle parent)
  {
    return ListCasterParts::cast(
      src, return_value_policy::reference_internal, parent);
  }
  static handle cast(const std::vector<std::shared_ptr<qd::Part>>* src,
                     return_value_policy pol,
                     handle parent)
  {
    return cast(*src, pol, parent);
  }
};
} // namespace detail
} // namespace pybind11

namespace qd {

/*========= PLUGIN: dyna_cpp =========*/
PYBIND11_MODULE(dyna_cpp, m)
{
  m.doc() = "c++ python wrapper for ls-dyna module";
  // pybind11::module m("dyna_cpp", "c++ python wrapper for ls-dyna module");

  // load numpy
  if (_import_array() < 0) {
    PyErr_SetString(PyExc_ImportError,
                    "numpy.core.multiarray failed to import");
    return; // nullptr;
  };

  // disable sigantures for documentation
  pybind11::options options;
  options.disable_function_signatures();

  // Node
  pybind11::class_<Node, std::shared_ptr<Node>> node_py(
    m, "Node", qd_node_class_docs);
  node_py
    .def("get_id",
         &Node::get_nodeID,
         pybind11::return_value_policy::take_ownership,
         node_get_id_docs)
    .def("__str__",
         &Node::str,
         pybind11::return_value_policy::take_ownership,
         node_str_docs)
    .def("get_coords",
         [](std::shared_ptr<Node> _node) {
           return qd::py::vector_to_nparray(_node->get_coords());
         },
         pybind11::return_value_policy::take_ownership,
         node_get_coords_docs)
    .def("get_disp",
         [](std::shared_ptr<Node> _node) {
           return qd::py::vector_to_nparray(_node->get_disp());
         },
         pybind11::return_value_policy::take_ownership,
         node_get_disp_docs)
    .def("get_vel",
         [](std::shared_ptr<Node> _node) {
           return qd::py::vector_to_nparray(_node->get_vel());
         },
         pybind11::return_value_policy::take_ownership,
         node_get_vel_docs)
    .def("get_accel",
         [](std::shared_ptr<Node> _node) {
           return qd::py::vector_to_nparray(_node->get_accel());
         },
         pybind11::return_value_policy::take_ownership,
         node_get_accel_docs)
    .def("get_elements",
         &Node::get_elements,
         pybind11::return_value_policy::reference_internal,
         node_get_elements_docs);

  // Element
  pybind11::class_<Element, std::shared_ptr<Element>> element_py(
    m, "Element", element_description);

  pybind11::enum_<Element::ElementType>(element_py, "type", element_type_docs)
    .value("none", Element::ElementType::NONE)
    .value("beam", Element::ElementType::BEAM)
    .value("shell", Element::ElementType::SHELL)
    .value("solid", Element::ElementType::SOLID)
    .value("tshell", Element::ElementType::TSHELL)
    .export_values();

  element_py
    .def("get_id",
         &Element::get_elementID,
         pybind11::return_value_policy::take_ownership,
         element_get_id_docs)
    .def("__str__",
         &Element::str,
         pybind11::return_value_policy::take_ownership,
         element_str_docs)
    .def("get_coords",
         [](std::shared_ptr<Element> _elem) {
           return qd::py::vector_to_nparray(_elem->get_coords());
         },
         pybind11::return_value_policy::take_ownership,
         element_get_coords_docs)
    .def("get_energy",
         [](std::shared_ptr<Element> _elem) {
           return qd::py::vector_to_nparray(_elem->get_energy());
         },
         pybind11::return_value_policy::take_ownership,
         element_get_energy_docs)
    .def("get_stress_mises",
         [](std::shared_ptr<Element> _elem) {
           return qd::py::vector_to_nparray(_elem->get_stress_mises());
         },
         pybind11::return_value_policy::take_ownership,
         element_get_stress_mises_docs)
    .def("get_plastic_strain",
         [](std::shared_ptr<Element> _elem) {
           return qd::py::vector_to_nparray(_elem->get_plastic_strain());
         },
         pybind11::return_value_policy::take_ownership,
         element_get_plastic_strain_docs)
    .def("get_strain",
         [](std::shared_ptr<Element> _elem) {
           return qd::py::vector_to_nparray(_elem->get_strain());
         },
         pybind11::return_value_policy::take_ownership,
         element_get_strain_docs)
    .def("get_stress",
         [](std::shared_ptr<Element> _elem) {
           return qd::py::vector_to_nparray(_elem->get_stress());
         },
         pybind11::return_value_policy::take_ownership,
         element_get_stress_docs)
    .def("get_history_variables",
         [](std::shared_ptr<Element> _elem) {
           return qd::py::vector_to_nparray(_elem->get_history_vars());
         },
         pybind11::return_value_policy::take_ownership,
         element_get_history_docs)
    .def("is_rigid",
         &Element::get_is_rigid,
         pybind11::return_value_policy::take_ownership,
         element_get_is_rigid_docs)
    .def("get_estimated_size",
         &Element::get_estimated_element_size,
         pybind11::return_value_policy::take_ownership,
         element_get_estimated_size_docs)
    .def("get_type",
         &Element::get_elementType,
         pybind11::return_value_policy::take_ownership,
         element_get_type_docs)
    .def("get_nodes",
         &Element::get_nodes,
         pybind11::return_value_policy::reference_internal,
         element_get_nodes_docs);

  // Part
  pybind11::class_<Part, std::shared_ptr<Part>> part_py(m, "QD_Part");
  part_py
    .def("get_name",
         &Part::get_name,
         pybind11::return_value_policy::take_ownership,
         part_get_name_docs)
    .def("get_id",
         &Part::get_partID,
         pybind11::return_value_policy::take_ownership,
         part_get_id_docs)
    .def("get_nodes",
         &Part::get_nodes,
         pybind11::return_value_policy::reference_internal,
         part_get_nodes_docs)
    .def("get_elements",
         &Part::get_elements,
         "element_filter"_a = Element::ElementType::NONE,
         pybind11::return_value_policy::reference_internal,
         part_get_elements_docs);

  // DB_Nodes
  pybind11::class_<DB_Nodes, std::shared_ptr<DB_Nodes>> db_nodes_py(
    m, "DB_Nodes", dbnodes_description);
  db_nodes_py
    .def("get_nNodes",
         &DB_Nodes::get_nNodes,
         pybind11::return_value_policy::take_ownership,
         dbnodes_get_nNodes_docs)
    .def("get_nodes",
         &DB_Nodes::get_nodes,
         pybind11::return_value_policy::take_ownership,
         dbnodes_get_nodes_docs)
    .def("get_nodeByID",
         (std::shared_ptr<Node>(DB_Nodes::*)(long)) &
           DB_Nodes::get_nodeByID<long>,
         "id"_a,
         pybind11::return_value_policy::reference_internal,
         dbnodes_get_nodeByID_docs)
    .def("get_nodeByID",
         //(std::vector<std::shared_ptr<Node>>(DB_Nodes::*)(pybind11::list)) &
         // DB_Nodes::get_nodeByID,
         [](std::shared_ptr<DB_Nodes> _db_nodes, pybind11::list _ids) {
           return _db_nodes->get_nodeByID(qd::py::container_to_vector<int32_t>(
             _ids, "An entry of the list was not a fully fledged integer."));
         },
         "id"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_nodeByID",
         //(std::vector<std::shared_ptr<Node>>(DB_Nodes::*)(pybind11::tuple))
         //& DB_Nodes::get_nodeByID,
         [](std::shared_ptr<DB_Nodes> _db_nodes, pybind11::tuple _ids) {
           return _db_nodes->get_nodeByID(qd::py::container_to_vector<int32_t>(
             _ids, "An entry of the list was not a fully fledged integer."));
         },
         "id"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_nodeByIndex",
         (std::shared_ptr<Node>(DB_Nodes::*)(long)) &
           DB_Nodes::get_nodeByIndex<long>,
         "index"_a,
         pybind11::return_value_policy::reference_internal,
         dbnodes_get_nodeByIndex_docs)
    .def(
      "get_nodeByIndex",
      //(std::vector<std::shared_ptr<Node>>(DB_Nodes::*)(pybind11::list)) &
      //  DB_Nodes::get_nodeByIndex,
      [](std::shared_ptr<DB_Nodes> _db_nodes, pybind11::list _indexes) {
        return _db_nodes->get_nodeByIndex(qd::py::container_to_vector<int32_t>(
          _indexes, "An entry of the list was not a fully fledged integer."));
      },
      "index"_a,
      pybind11::return_value_policy::reference_internal)
    .def(
      "get_nodeByIndex",
      //(std::vector<std::shared_ptr<Node>>(DB_Nodes::*)(pybind11::tuple)) &
      //  DB_Nodes::get_nodeByIndex,
      [](std::shared_ptr<DB_Nodes> _db_nodes, pybind11::tuple _indexes) {
        return _db_nodes->get_nodeByIndex(qd::py::container_to_vector<int32_t>(
          _indexes, "An entry of the list was not a fully fledged integer."));
      },
      "index"_a,
      pybind11::return_value_policy::reference_internal);

  // DB_Elements
  pybind11::class_<DB_Elements, std::shared_ptr<DB_Elements>> db_elements_py(
    m, "DB_Elements", dbelems_description);
  db_elements_py
    .def("get_nElements",
         &DB_Elements::get_nElements,
         "element_type"_a = Element::NONE,
         pybind11::return_value_policy::take_ownership,
         dbelems_get_nElements_docs)
    .def("get_elements",
         &DB_Elements::get_elements,
         "element_type"_a = Element::NONE,
         pybind11::return_value_policy::take_ownership,
         get_elements_docs)
    .def(
      "get_elementByID",
      (std::shared_ptr<Element>(DB_Elements::*)(Element::ElementType, long)) &
        DB_Elements::get_elementByID<long>,
      "element_type"_a,
      "id"_a,
      pybind11::return_value_policy::reference_internal)
    .def("get_elementByID",
         // (std::vector<std::shared_ptr<Element>>(DB_Elements::*)(
         // Element::ElementType, pybind11::list))
         // &DB_Elements::get_elementByID<long>,
         [](std::shared_ptr<DB_Elements> _db_elems,
            Element::ElementType _eType,
            pybind11::list _list) {
           return _db_elems->get_elementByID(
             _eType,
             qd::py::container_to_vector<long>(
               _list, "An entry of the id list was not an integer."));
         },
         "element_type"_a,
         "id"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_elementByID",
         //(std::vector<std::shared_ptr<Element>>(DB_Elements::*)(
         //  Element::ElementType, pybind11::tuple)) &
         //  DB_Elements::get_elementByID<long>,
         [](std::shared_ptr<DB_Elements> _db_elems,
            Element::ElementType _eType,
            pybind11::tuple _list) {
           return _db_elems->get_elementByID(
             _eType,
             qd::py::container_to_vector<long>(
               _list, "An entry of the id list was not an integer."));
         },
         "element_type"_a,
         "id"_a,
         pybind11::return_value_policy::reference_internal)
    .def(
      "get_elementByIndex",
      (std::shared_ptr<Element>(DB_Elements::*)(Element::ElementType, long)) &
        DB_Elements::get_elementByIndex<long>,
      "element_type"_a,
      "index"_a,
      pybind11::return_value_policy::reference_internal,
      dbelems_get_elementByIndex_docs)
    .def("get_elementByIndex",
         //(std::vector<std::shared_ptr<Element>>(DB_Elements::*)(
         //  Element::ElementType, pybind11::list)) &
         //  DB_Elements::get_elementByIndex<long>,
         [](std::shared_ptr<DB_Elements> _db_elems,
            Element::ElementType _eType,
            pybind11::tuple _list) {
           return _db_elems->get_elementByIndex(
             _eType,
             qd::py::container_to_vector<long>(
               _list, "An entry of the index list was not an integer."));
         },
         "element_type"_a,
         "index"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_elementByIndex",
         //(std::vector<std::shared_ptr<Element>>(DB_Elements::*)(
         //  Element::ElementType, pybind11::tuple)) &
         //  DB_Elements::get_elementByIndex<long>,
         [](std::shared_ptr<DB_Elements> _db_elems,
            Element::ElementType _eType,
            pybind11::list _list) {
           return _db_elems->get_elementByIndex(
             _eType,
             qd::py::container_to_vector<long>(
               _list, "An entry of the index list was not an integer."));
         },
         "element_type"_a,
         "index"_a,
         pybind11::return_value_policy::reference_internal);

  // DB_Parts
  pybind11::class_<DB_Parts, std::shared_ptr<DB_Parts>> db_parts_py(
    m, "DB_Parts", dbparts_description);
  db_parts_py
    .def("get_nParts",
         &DB_Parts::get_nParts,
         pybind11::return_value_policy::take_ownership,
         dbparts_get_nParts_docs)
    .def("get_parts",
         &DB_Parts::get_parts,
         pybind11::return_value_policy::reference_internal,
         dbparts_get_parts_docs)
    .def("get_partByID",
         (std::shared_ptr<Part>(DB_Parts::*)(long)) &
           DB_Parts::get_partByID<long>,
         "id"_a,
         pybind11::return_value_policy::reference_internal,
         dbparts_get_partByID_docs)
    .def("get_partByID",
         //(std::vector<std::shared_ptr<Part>>(DB_Parts::*)(pybind11::list)) &
         //  DB_Parts::get_partByID,
         [](std::shared_ptr<DB_Parts> _db_parts, pybind11::list _list) {
           return _db_parts->get_partByID(
             qd::py::container_to_vector<int32_t>(_list));
         },
         "id"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_partByID",
         //(std::vector<std::shared_ptr<Part>>(DB_Parts::*)(pybind11::tuple)) &
         //  DB_Parts::get_partByID,
         [](std::shared_ptr<DB_Parts> _db_parts, pybind11::tuple _list) {
           return _db_parts->get_partByID(
             qd::py::container_to_vector<int32_t>(_list));
         },
         "id"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_partByIndex",
         (std::shared_ptr<Part>(DB_Parts::*)(long)) &
           DB_Parts::get_partByIndex<long>,
         "index"_a,
         pybind11::return_value_policy::reference_internal,
         dbparts_get_partByIndex_docs)
    .def("get_partByIndex",
         //(std::vector<std::shared_ptr<Part>>(DB_Parts::*)(pybind11::list)) &
         //  DB_Parts::get_partByIndex,
         [](std::shared_ptr<DB_Parts> _db_parts, pybind11::list _list) {
           return _db_parts->get_partByIndex(
             qd::py::container_to_vector<int32_t>(_list));
         },
         "index"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_partByIndex",
         //(std::vector<std::shared_ptr<Part>>(DB_Parts::*)(pybind11::tuple)) &
         //  DB_Parts::get_partByIndex,
         [](std::shared_ptr<DB_Parts> _db_parts, pybind11::tuple _list) {
           return _db_parts->get_partByIndex(
             qd::py::container_to_vector<int32_t>(_list));
         },
         "index"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_partByName",
         &DB_Parts::get_partByName,
         "name"_a,
         pybind11::return_value_policy::reference_internal,
         dbparts_get_partByName_docs);

  // FEMFile
  pybind11::
    class_<FEMFile, DB_Parts, DB_Nodes, DB_Elements, std::shared_ptr<FEMFile>>
      femfile_py(m, "FEMFile", pybind11::multiple_inheritance());
  femfile_py.def("get_filepath",
                 &FEMFile::get_filepath,
                 pybind11::return_value_policy::take_ownership,
                 femfile_get_filepath_docs);

  // D3plot
  pybind11::class_<D3plot, FEMFile, std::shared_ptr<D3plot>> d3plot_py(
    m, "QD_D3plot", d3plot_description);
  d3plot_py
    .def(pybind11::init<std::string, std::string, bool>(),
         "filepath"_a,
         "read_states"_a = std::string(),
         "use_femzip"_a = false,
         d3plot_constructor)
    .def( // pybind11::init<std::string, pybind11::list, bool>(),
      pybind11::init(
        [](std::string _filepath, pybind11::list _variables, bool _use_femzip) {
          return std::make_shared<D3plot>(
            _filepath,
            qd::py::container_to_vector<std::string>(
              _variables, "An entry of read_states was not of type str"),
            _use_femzip);
        }),
      "filepath"_a,
      "read_states"_a = pybind11::list(),
      "use_femzip"_a = false)
    .def( // pybind11::init<std::string, pybind11::tuple, bool>(),
      pybind11::init([](std::string _filepath,
                        pybind11::tuple _variables,
                        bool _use_femzip) {
        return std::make_shared<D3plot>(
          _filepath,
          qd::py::container_to_vector<std::string>(
            _variables, "An entry of read_states was not of type str"),
          _use_femzip);
      }),
      "filepath"_a,
      "read_states"_a = pybind11::tuple(),
      "use_femzip"_a = false)
    .def("info", &D3plot::info, d3plot_info_docs)
    .def("read_states",
         (void (D3plot::*)(const std::string&)) & D3plot::read_states,
         d3plot_read_states_docs)
    .def("read_states",
         //(void (D3plot::*)(pybind11::list)) & D3plot::read_states)
         [](std::shared_ptr<D3plot> _d3plot, pybind11::list _list) {
           _d3plot->read_states(qd::py::container_to_vector<std::string>(
             _list, "An entry of read_states was not of type str"));
         })
    .def("read_states",
         //(void (D3plot::*)(pybind11::tuple)) & D3plot::read_states)
         [](std::shared_ptr<D3plot> _d3plot, pybind11::tuple _list) {
           _d3plot->read_states(qd::py::container_to_vector<std::string>(
             _list, "An entry of read_states was not of type str"));
         })
    .def("clear",
         //(void (D3plot::*)(pybind11::list)) & D3plot::clear,
         [](std::shared_ptr<D3plot> _d3plot, pybind11::list _list) {
           _d3plot->clear(qd::py::container_to_vector<std::string>(
             _list, "An entry of list was not of type str"));
         },
         "variables"_a = pybind11::list(),
         d3plot_clear_docs)
    .def("clear",
         //(void (D3plot::*)(pybind11::tuple)) & D3plot::clear,
         [](std::shared_ptr<D3plot> _d3plot, pybind11::tuple _list) {
           _d3plot->clear(qd::py::container_to_vector<std::string>(
             _list, "An entry of list was not of type str"));
         },
         "variables"_a = pybind11::tuple())
    .def("clear",
         (void (D3plot::*)(const std::string&)) & D3plot::clear,
         "variables"_a = pybind11::str())
    .def("get_timesteps",
         //&D3plot::get_timesteps_py,
         [](std::shared_ptr<D3plot> _d3plot) {
           return qd::py::vector_to_nparray(_d3plot->get_timesteps());
         },
         pybind11::return_value_policy::take_ownership,
         d3plot_get_timesteps_docs)
    .def("get_nTimesteps",
         &D3plot::get_nTimesteps,
         pybind11::return_value_policy::take_ownership,
         d3plot_get_nTimesteps_docs)
    .def("get_title",
         &D3plot::get_title,
         pybind11::return_value_policy::take_ownership,
         d3plot_get_title_docs);
  /*
.def("save_hdf5",
  &D3plot::save_hdf5,
  "filepath"_a,
  "overwrite_run"_a = true,
  "run_name"_a = "",
  "docs missing");
  */

  // RawD3plot
  pybind11::class_<RawD3plot, std::shared_ptr<RawD3plot>> raw_d3plot_py(
    m, "QD_RawD3plot");
  raw_d3plot_py
    .def(pybind11::init<std::string, bool>(),
         "filepath"_a,
         "use_femzip"_a = false,
         rawd3plot_constructor_description)
    .def("_get_string_names",
         &RawD3plot::get_string_names,
         pybind11::return_value_policy::take_ownership,
         rawd3plot_get_string_names_docs)
    .def("_get_string_data",
         &RawD3plot::get_string_data,
         "name"_a,
         pybind11::return_value_policy::take_ownership,
         rawd3plot_get_string_data_docs)
    .def("_get_int_names",
         &RawD3plot::get_int_names,
         pybind11::return_value_policy::take_ownership,
         rawd3plot_get_int_names_docs)
    .def("_get_int_data",
         [](std::shared_ptr<RawD3plot> _d3plot, std::string _entry_name) {
           return qd::py::tensor_to_nparray(_d3plot->get_int_data(_entry_name));
         },
         "name"_a,
         pybind11::return_value_policy::take_ownership,
         rawd3plot_get_int_data_docs)
    .def("_get_float_names",
         &RawD3plot::get_float_names,
         pybind11::return_value_policy::take_ownership,
         rawd3plot_get_float_names_docs)
    .def("_get_float_data",
         [](std::shared_ptr<RawD3plot> _d3plot, std::string _entry_name) {
           return qd::py::tensor_to_nparray(
             _d3plot->get_float_data(_entry_name));
         },
         "name"_a,
         pybind11::return_value_policy::take_ownership,
         rawd3plot_get_float_data_docs)
    .def("_set_float_data",
         [](std::shared_ptr<RawD3plot> _d3plot,
            std::string _entry_name,
            pybind11::array_t<float> _data) {

           auto shape_sizet = std::vector<size_t>(_data.ndim());
           for (ssize_t ii = 0; ii < _data.ndim(); ++ii)
             shape_sizet[ii] = _data.shape()[ii];

           _d3plot->set_float_data(_entry_name, shape_sizet, _data.data());

         },
         "name"_a,
         "array"_a,
         pybind11::return_value_policy::take_ownership)
    .def("_set_int_data",
         [](std::shared_ptr<RawD3plot> _d3plot,
            std::string _entry_name,
            pybind11::array_t<int> _data) {

           auto shape_sizet = std::vector<size_t>(_data.ndim());
           for (ssize_t ii = 0; ii < _data.ndim(); ++ii)
             shape_sizet[ii] = _data.shape()[ii];

           _d3plot->set_int_data(_entry_name, shape_sizet, _data.data());

         },
         "name"_a,
         "array"_a,
         pybind11::return_value_policy::take_ownership)
    .def("info", &RawD3plot::info);

  // KeyFile
  pybind11::class_<KeyFile, FEMFile, std::shared_ptr<KeyFile>> keyfile_py(
    m, "QD_KeyFile", keyfile_description);
  keyfile_py.def(pybind11::init<const std::string&, bool, double>(),
                 "filepath"_a,
                 "load_includes"_a = true,
                 "encryption_detection"_a = 0.7,
                 keyfile_constructor);

  // Binout
  /*
  const char* empty_description = { "\0" };
  pybind11::class_<Binout, std::shared_ptr<Binout>> binout_py(
    m, "QD_Binout", empty_description);
  binout_py.def(pybind11::init<std::string>(), "filepath"_a, empty_description);
  */

  // module functions
  m.def("get_file_entropy",
        [](std::string _filepath) {
          std::vector<char> buffer = read_binary_file(_filepath);
          return get_entropy(buffer);
        },
        pybind11::return_value_policy::take_ownership,
        module_get_file_entropy_description);

  // return m.ptr();
}

} //  namespace qd