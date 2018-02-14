
#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/db/FEMFile.hpp>
#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/db/Part.hpp>
//#include <dyna_cpp/dyna/Binout.hpp>
#include <dyna_cpp/dyna/D3plot.hpp>
#include <dyna_cpp/dyna/ElementKeyword.hpp>
#include <dyna_cpp/dyna/KeyFile.hpp>
#include <dyna_cpp/dyna/Keyword.hpp>
#include <dyna_cpp/dyna/NodeKeyword.hpp>
#include <dyna_cpp/dyna/PartKeyword.hpp>
#include <dyna_cpp/dyna/RawD3plot.hpp>
#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/PythonUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

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
         element_get_nodes_docs)
    .def("get_part_id",
         &Element::get_part_id,
         pybind11::return_value_policy::reference_internal);

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
    .def(pybind11::init<>())
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
         "data"_a)
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
         "data"_a)
    .def("_set_string_data",
         [](std::shared_ptr<RawD3plot> _d3plot,
            std::string _entry_name,
            pybind11::list _data) {

           _d3plot->set_string_data(
             _entry_name, qd::py::container_to_vector<std::string>(_data));
         },
         "name"_a,
         "data"_a)
    .def("_set_string_data",
         [](std::shared_ptr<RawD3plot> _d3plot,
            std::string _entry_name,
            pybind11::tuple _data) {

           _d3plot->set_string_data(
             _entry_name, qd::py::container_to_vector<std::string>(_data));
         },
         "name"_a,
         "data"_a)
    .def("info", &RawD3plot::info, rawd3plot_info_docs);

  // Keyword (and subclasses)
  pybind11::class_<Keyword, std::shared_ptr<Keyword>> keyword_py(m, "Keyword");
  pybind11::class_<NodeKeyword, Keyword, std::shared_ptr<NodeKeyword>>
    node_keyword_py(m, "NodeKeyword");
  pybind11::class_<ElementKeyword, Keyword, std::shared_ptr<ElementKeyword>>
    element_keyword_py(m, "ElementKeyword");
  pybind11::class_<PartKeyword, Keyword, std::shared_ptr<PartKeyword>>
    part_keyword_py(m, "PartKeyword");
  pybind11::
    class_<IncludePathKeyword, Keyword, std::shared_ptr<IncludePathKeyword>>
      include_path_keyword_py(m, "IncludePathKeyword");
  pybind11::class_<IncludeKeyword, Keyword, std::shared_ptr<IncludeKeyword>>
    include_keyword_py(m, "IncludeKeyword");

  pybind11::enum_<Keyword::Align>(keyword_py, "align", keyword_enum_align_docs)
    .value("left", Keyword::Align::LEFT)
    .value("middle", Keyword::Align::MIDDLE)
    .value("right", Keyword::Align::RIGHT)
    .export_values();

  keyword_py
    .def(pybind11::init<std::string, int64_t>(),
         "lines"_a,
         "line_index"_a = 0,
         keyword_constructor_docs)
    .def(pybind11::init<std::vector<std::string>, int64_t>(),
         "lines"_a,
         "line_index"_a = 0)
    .def("__str__",
         &Keyword::str,
         pybind11::return_value_policy::take_ownership,
         keyword_str_docs)
    .def("__repr__",
         [](std::shared_ptr<Keyword> self) {
           return std::string("<Keyword: " + self->get_keyword_name() + ">");
         },
         pybind11::return_value_policy::take_ownership,
         keyword_repr_docs)
    .def("__iter__",
         [](std::shared_ptr<Keyword> kw) {
           auto& buffer = kw->get_lines();
           return pybind11::make_iterator(buffer.begin(), buffer.end());
         },
         pybind11::keep_alive<0, 1>(),
         keyword_iter_docs)
    .def("__getitem__",
         [](std::shared_ptr<Keyword> self, int64_t iCard) {
           return self->get_card(iCard);
         },
         pybind11::return_value_policy::take_ownership,
         keyword_getitem_docs)
    .def("__getitem__",
         [](std::shared_ptr<Keyword> kw, const std::string& _field_name) {
           return py::try_number_conversion(kw->get_card_value(_field_name));
         },
         pybind11::return_value_policy::take_ownership)
    .def("__getitem__",
         [](std::shared_ptr<Keyword> self, std::tuple<int64_t, int64_t> arg) {
           return py::try_number_conversion(
             self->get_card_value(std::get<0>(arg), std::get<1>(arg)));
         },
         pybind11::return_value_policy::take_ownership)
    .def("__getitem__",
         [](std::shared_ptr<Keyword> self,
            std::tuple<int64_t, int64_t, size_t> arg) {
           return py::try_number_conversion(self->get_card_value(
             std::get<0>(arg), std::get<1>(arg), std::get<2>(arg)));
         },
         pybind11::return_value_policy::take_ownership)
    .def("get_card_valueByIndex",
         [](std::shared_ptr<Keyword> self,
            int64_t iCard,
            int64_t iField,
            size_t field_size) {
           return py::try_number_conversion(
             self->get_card_value(iCard, iField, field_size));
         },
         "iCard"_a,
         "iField"_a,
         "field_size"_a = 0,
         pybind11::return_value_policy::take_ownership,
         keyword_get_card_valueByIndex_docs)
    .def("get_card_valueByName",
         [](std::shared_ptr<Keyword> self,
            const std::string& name,
            size_t _field_size) {
           return py::try_number_conversion(
             self->get_card_value(name, _field_size));
         },
         "name"_a,
         "field_size"_a = 0,
         pybind11::return_value_policy::take_ownership,
         keyword_get_card_valueByName_docs)
    .def("__setitem__",
         [](std::shared_ptr<Keyword> self,
            pybind11::tuple args,
            pybind11::object _value) {

           switch (args.size()) {
             case (0):
               throw(
                 std::invalid_argument("At least one argument is required."));
             case (1):
               // card index
               if (pybind11::isinstance<pybind11::int_>(args[0]))
                 self->set_card(
                   pybind11::cast<int64_t>(args[0]),
                   pybind11::cast<std::string>(pybind11::str(_value)));
               // field name
               else
                 self->set_card_value(
                   pybind11::cast<std::string>(pybind11::str(args[0])),
                   pybind11::cast<std::string>(pybind11::str(_value)));
               break;

             case (2):
               // card and field index
               if (pybind11::isinstance<pybind11::int_>(args[0]))
                 self->set_card_value(
                   pybind11::cast<int64_t>(args[0]),
                   pybind11::cast<int64_t>(args[1]),
                   pybind11::cast<std::string>(pybind11::str(_value)));
               // field name and field size
               else
                 self->set_card_value(
                   pybind11::cast<std::string>(pybind11::str(args[0])),
                   pybind11::cast<std::string>(pybind11::str(_value)),
                   pybind11::cast<size_t>(args[1]));
               break;

             // card and field index and field size
             case (3):
               self->set_card_value(
                 pybind11::cast<int64_t>(args[0]),
                 pybind11::cast<int64_t>(args[1]),
                 pybind11::cast<std::string>(pybind11::str(_value)),
                 "",
                 pybind11::cast<size_t>(args[2]));
               break;

             default:
               throw(std::invalid_argument(
                 "Invalid number of arguments. Valid are:\n"
                 "- card index (int)\n"
                 "- card and field index (int,int)\n"
                 "- card index, field index and field size (int,int,int)\n"
                 "- field name (str)\n"
                 "- field name and field size (str,int)\n"));
               break;
           }
         },
         keyword_setitem_docs)
    .def("__setitem__",
         [](std::shared_ptr<Keyword> self,
            const std::string& _field_name,
            pybind11::object _value) {
           self->set_card_value(
             _field_name, pybind11::cast<std::string>(pybind11::str(_value)));
         })
    .def("__setitem__",
         [](std::shared_ptr<Keyword> self,
            const std::string& _field_name,
            pybind11::object _value) {
           self->set_card_value(
             _field_name, pybind11::cast<std::string>(pybind11::str(_value)));
         })
    .def("__setitem__",
         [](std::shared_ptr<Keyword> self,
            int64_t iCard,
            const std::string& data) { self->set_card(iCard, data); })
    .def("__setitem__",
         [](std::shared_ptr<Keyword> self,
            int64_t iCard,
            int64_t iField,
            pybind11::object value) {
           self->set_card_value(
             iCard, iField, pybind11::cast<std::string>(pybind11::str(value)));
         })
    .def("__setitem__",
         [](std::shared_ptr<Keyword> self,
            int64_t iCard,
            int64_t iField,
            int64_t field_size,
            pybind11::object value) {
           self->set_card_value(
             iCard, iField, pybind11::cast<std::string>(pybind11::str(value)));
         })
    .def("set_card_valueByIndex",
         [](std::shared_ptr<Keyword> self,
            int64_t iCard,
            int64_t iField,
            pybind11::object value,
            const std::string& name,
            size_t field_size) {
           self->set_card_value(
             iCard,
             iField,
             pybind11::cast<std::string>(pybind11::str(value)),
             name,
             field_size);
         },
         "iCard"_a,
         "iField"_a,
         "value"_a,
         "name"_a = "",
         "field_size"_a = 0,
         keyword_set_card_valueByIndex_docs)
    .def("set_card_valueByName",
         [](std::shared_ptr<Keyword> self,
            const std::string& field_name,
            pybind11::object value,
            size_t field_size) {
           self->set_card_value(
             field_name,
             pybind11::cast<std::string>(pybind11::str(value)),
             field_size);
         },
         "name"_a,
         "value"_a,
         "field_size"_a = 0,
         keyword_set_card_valueByName_docs)
    .def("set_card_valueByDict",
         [](std::shared_ptr<Keyword> self,
            pybind11::dict fields,
            size_t field_size) {
           for (auto& item : fields) {

             if (pybind11::isinstance<pybind11::str>(item.first))
               self->set_card_value(
                 pybind11::cast<std::string>(item.first),
                 pybind11::cast<std::string>(pybind11::str(item.second)),
                 field_size);

             else if (pybind11::isinstance<pybind11::tuple>(item.first)) {
               auto tmp_tuple = pybind11::cast<pybind11::tuple>(item.first);
               if (tmp_tuple.size() < 2)
                 throw(std::invalid_argument(
                   "Setting a card value from a tuple requires at least two "
                   "indexes: iCard and iField."));
               auto iField = pybind11::cast<int64_t>(tmp_tuple[0]);
               auto iCard = pybind11::cast<int64_t>(tmp_tuple[1]);
               auto field_size = tmp_tuple.size() > 2
                                   ? pybind11::cast<size_t>(tmp_tuple[2])
                                   : 0;
               self->set_card_value(
                 iField,
                 iCard,
                 pybind11::cast<std::string>(pybind11::str(item.second)),
                 "",
                 field_size);
             }
           }
         },
         "fields"_a,
         "field_size"_a = 0,
         keyword_set_card_valueByDict_docs)
    .def("__len__",
         &Keyword::size,
         pybind11::return_value_policy::take_ownership,
         keyword_len_docs)
    .def("append_line",
         &Keyword::append_line,
         "line"_a,
         pybind11::return_value_policy::take_ownership,
         keyword_append_line_docs)
    .def("get_lines",
         (std::vector<std::string> & (Keyword::*)()) & Keyword::get_lines,
         pybind11::return_value_policy::take_ownership,
         keyword_get_lines_docs)
    .def("get_line",
         &Keyword::get_line<int64_t>,
         "iLine"_a,
         pybind11::return_value_policy::take_ownership,
         keyword_get_line_docs)
    .def("set_lines", &Keyword::set_lines, "lines"_a, keyword_set_lines_docs)
    .def("set_line",
         &Keyword::set_line<int64_t>,
         "iLine"_a,
         "line"_a,
         keyword_set_line_docs)
    .def("insert_line",
         &Keyword::insert_line<int64_t>,
         "iLine"_a,
         "line"_a,
         keyword_insert_line_docs)
    .def("remove_line",
         &Keyword::remove_line<int64_t>,
         "iLine"_a,
         keyword_remove_line_docs)
    .def_property("line_index",
                  &Keyword::get_line_index,
                  &Keyword::set_line_index,
                  keyword_line_index_docs)
    .def("switch_field_size",
         &Keyword::switch_field_size<size_t>,
         "skip_cards"_a = pybind11::list(),
         keyword_switch_field_size_docs)
    .def("switch_field_size",
         &Keyword::switch_field_size<size_t>,
         "skip_cards"_a = pybind11::tuple())
    .def("reformat_all",
         &Keyword::reformat_all<size_t>,
         "skip_cards"_a = pybind11::list(),
         keyword_reformat_all_docs)
    .def("reformat_all",
         &Keyword::reformat_all<size_t>,
         "skip_cards"_a = pybind11::tuple())
    .def("reformat_field",
         &Keyword::reformat_card_value<size_t>,
         "iCard"_a,
         "iField"_a,
         "field_size"_a = 0,
         "format_field"_a = true,
         "format_name"_a = true,
         keyword_reformat_field_docs)
    .def("has_long_fields",
         &Keyword::has_long_fields,
         keyword_has_long_fields_docs)
    .def("get_keyword_name",
         &Keyword::get_keyword_name,
         keyword_get_keyword_name_docs)
    .def_property_static(
      "name_delimiter",
      [](pybind11::object) { return Keyword::name_delimiter; },
      [](pybind11::object, char val) { Keyword::name_delimiter = val; })
    .def_property_static(
      "name_delimiter_used",
      [](pybind11::object) { return Keyword::name_delimiter_used; },
      [](pybind11::object, bool _arg) { Keyword::name_delimiter_used = _arg; },
      keyword_name_delimiter_docs,
      keyword_name_delimiter_used_docs)
    .def_property_static(
      "name_spacer",
      [](pybind11::object) { return Keyword::name_spacer; },
      [](pybind11::object, char val) { Keyword::name_spacer = val; },
      keyword_name_spacer_docs)
    .def_property_static(
      "field_alignment",
      [](pybind11::object) { return Keyword::field_alignment; },
      [](pybind11::object, Keyword::Align _align) {
        Keyword::field_alignment = _align;
      },
      keyword_field_alignment_docs)
    .def_property_static(
      "name_alignment",
      [](pybind11::object) { return Keyword::name_alignment; },
      [](pybind11::object, Keyword::Align _align) {
        Keyword::name_alignment = _align;
      },
      keyword_name_alignment_docs);

  node_keyword_py
    .def("add_node",
         (std::shared_ptr<Node>(NodeKeyword::*)(
           int64_t, float, float, float, const std::string&)) &
           NodeKeyword::add_node<int64_t>,
         "id"_a,
         "x"_a,
         "y"_a,
         "z"_a,
         "additional_card_data"_a = "",
         pybind11::return_value_policy::take_ownership,
         node_keyword_add_node_docs)
    .def("get_nNodes",
         &NodeKeyword::get_nNodes,
         pybind11::return_value_policy::take_ownership,
         node_keyword_get_nNodes_docs)
    .def("get_nodes",
         &NodeKeyword::get_nodes,
         pybind11::return_value_policy::take_ownership,
         node_keyword_get_nodes_docs)
    .def("get_node_ids",
         &NodeKeyword::get_node_ids,
         pybind11::return_value_policy::take_ownership,
         node_keyword_get_node_ids_docs)
    .def("load", &NodeKeyword::load, node_keyword_load_docs);

  element_keyword_py.def("get_elements", &ElementKeyword::get_elements)
    .def("get_nElements", &ElementKeyword::get_nElements)
    .def("get_elementByIndex",
         &ElementKeyword::get_elementByIndex<int64_t>,
         "index"_a)
    .def("add_elementByNodeID",
         &ElementKeyword::add_elementByNodeID<int64_t>,
         "id"_a,
         "part_id"_a,
         "node_ids"_a)
    .def("add_elementByNodeIndex",
         &ElementKeyword::add_elementByNodeIndex<int64_t>,
         "id"_a,
         "part_id"_a,
         "node_indexes"_a)
    .def("load", &ElementKeyword::load);

  part_keyword_py
    .def("add_part",
         (std::shared_ptr<Part>(PartKeyword::*)(
           int64_t, const std::string&, const std::vector<std::string>&)) &
           PartKeyword::add_part<int64_t>,
         "id"_a,
         "name"_a = "",
         "additional_lines"_a = std::vector<std::string>(),
         pybind11::return_value_policy::take_ownership,
         part_keyword_add_part_docs)
    .def("add_part",
         [](std::shared_ptr<PartKeyword> self,
            int64_t part_id,
            const std::string& name,
            const std::string& additional_lines) {
           return self->add_part(part_id, name, { additional_lines });
         },
         "id"_a,
         "name"_a = "",
         "additional_lines"_a = "",
         pybind11::return_value_policy::take_ownership)
    .def("get_partByIndex", &PartKeyword::get_partByIndex<int64_t>, "index"_a)
    .def("get_parts", &PartKeyword::get_parts)
    .def("get_nParts", &PartKeyword::get_nParts)
    .def("load", &PartKeyword::load)
    .def("__str__", &PartKeyword::str);

  include_path_keyword_py.def("is_relative", &IncludePathKeyword::is_relative)
    .def("get_include_dirs", &IncludePathKeyword::get_include_dirs);

  include_keyword_py.def("get_includes", &IncludeKeyword::get_includes)
    .def("load", &IncludeKeyword::load);

  // KeyFile
  pybind11::class_<KeyFile, FEMFile, std::shared_ptr<KeyFile>> keyfile_py(
    m, "QD_KeyFile", keyfile_description);
  keyfile_py
    .def("__init__",
         [](KeyFile& instance,
            const std::string& _filepath,
            bool read_generic_keywords,
            bool parse_mesh,
            bool load_includes,
            double encryption_detection_threshold) {

           new (&instance) KeyFile(_filepath,
                                   read_generic_keywords,
                                   parse_mesh,
                                   load_includes,
                                   encryption_detection_threshold);
           instance.load();

         },
         "filepath"_a,
         "read_keywords"_a = true,
         "parse_mesh"_a = false,
         "load_includes"_a = false,
         "encryption_detection"_a = 0.7,
         keyfile_constructor)
    .def("__str__", &KeyFile::str, keyfile_str_description)
    .def(
      "__getitem__",
      [](std::shared_ptr<KeyFile> self, std::string key) {
        auto kwrds = self->get_keywordsByName(key);

        // safe sex
        if (kwrds.empty())
          return pybind11::cast(kwrds);

        switch (kwrds[0]->get_keyword_type()) {

          // node keyword
          case (Keyword::KeywordType::NODE): {
            std::vector<std::shared_ptr<NodeKeyword>> ret(kwrds.size());
            for (size_t ii = 0; ii < kwrds.size(); ++ii)
              ret[ii] = std::static_pointer_cast<NodeKeyword>(kwrds[ii]);
            return pybind11::cast(ret);
          }

          // element keyword
          case (Keyword::KeywordType::ELEMENT): {
            std::vector<std::shared_ptr<ElementKeyword>> ret(kwrds.size());
            for (size_t ii = 0; ii < kwrds.size(); ++ii)
              ret[ii] = std::static_pointer_cast<ElementKeyword>(kwrds[ii]);
            return pybind11::cast(ret);
          }

          case (Keyword::KeywordType::PART): {
            std::vector<std::shared_ptr<PartKeyword>> ret(kwrds.size());
            for (size_t ii = 0; ii < kwrds.size(); ++ii)
              ret[ii] = std::static_pointer_cast<PartKeyword>(kwrds[ii]);
            return pybind11::cast(ret);
          }

          case (Keyword::KeywordType::INCLUDE_PATH): {
            std::vector<std::shared_ptr<IncludePathKeyword>> ret(kwrds.size());
            for (size_t ii = 0; ii < kwrds.size(); ++ii)
              ret[ii] = std::static_pointer_cast<IncludePathKeyword>(kwrds[ii]);
            return pybind11::cast(ret);
          }

          case (Keyword::KeywordType::INCLUDE): {
            std::vector<std::shared_ptr<IncludeKeyword>> ret(kwrds.size());
            for (size_t ii = 0; ii < kwrds.size(); ++ii)
              ret[ii] = std::static_pointer_cast<IncludeKeyword>(kwrds[ii]);
            return pybind11::cast(ret);
          }

          default:
            return pybind11::cast(kwrds);
        }

      },
      "name"_a,
      keyfile_getitem_description)
    .def("keys", &KeyFile::keys, keyfile_keys_description)
    .def("save", &KeyFile::save_txt, "filepath"_a, keyfile_save_description)
    .def("remove_keyword",
         [](std::shared_ptr<KeyFile> self,
            const std::string& name,
            int64_t index) { self->remove_keyword(name, index); },
         "name"_a,
         "index"_a,
         keyfile_remove_keyword_description)
    .def("remove_keyword",
         [](std::shared_ptr<KeyFile> self, const std::string& name) {
           self->remove_keyword(name);
         },
         "name"_a)
    .def("add_keyword",
         &KeyFile::add_keyword,
         "lines"_a,
         keyfile_add_keyword_description)
    .def(
      "get_includes", &KeyFile::get_includes, keyfile_get_includes_description)
    .def("get_include_dirs",
         [](std::shared_ptr<KeyFile> self) {
           return self->get_include_dirs(true);
         },
         keyfile_get_include_dirs_description);

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