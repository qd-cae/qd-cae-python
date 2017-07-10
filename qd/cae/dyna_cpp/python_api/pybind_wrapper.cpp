
#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/db/FEMFile.hpp>
#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/db/Part.hpp>
#include <dyna_cpp/dyna/D3plot.hpp>
#include <dyna_cpp/dyna/KeyFile.hpp>
#include <dyna_cpp/utility/PythonUtility.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <numpy/arrayobject.h>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace py::literals;

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
  pybind11::detail::list_caster<std::vector<std::shared_ptr<Node>>,
                                std::shared_ptr<Node>>;
using ListCasterElements =
  pybind11::detail::list_caster<std::vector<std::shared_ptr<Element>>,
                                std::shared_ptr<Element>>;
using ListCasterParts =
  pybind11::detail::list_caster<std::vector<std::shared_ptr<Part>>,
                                std::shared_ptr<Part>>;
namespace pybind11 {
namespace detail {
template<>
struct type_caster<std::vector<std::shared_ptr<Node>>> : ListCasterNodes
{
  static handle cast(const std::vector<std::shared_ptr<Node>>& src,
                     return_value_policy,
                     handle parent)
  {
    return ListCasterNodes::cast(
      src, return_value_policy::reference_internal, parent);
  }
  static handle cast(const std::vector<std::shared_ptr<Node>>* src,
                     return_value_policy pol,
                     handle parent)
  {
    return cast(*src, pol, parent);
  }
};

template<>
struct type_caster<std::vector<std::shared_ptr<Element>>> : ListCasterElements
{
  static handle cast(const std::vector<std::shared_ptr<Element>>& src,
                     return_value_policy,
                     handle parent)
  {
    return ListCasterElements::cast(
      src, return_value_policy::reference_internal, parent);
  }
  static handle cast(const std::vector<std::shared_ptr<Element>>* src,
                     return_value_policy pol,
                     handle parent)
  {
    return cast(*src, pol, parent);
  }
};

template<>
struct type_caster<std::vector<std::shared_ptr<Part>>> : ListCasterParts
{
  static handle cast(const std::vector<std::shared_ptr<Part>>& src,
                     return_value_policy,
                     handle parent)
  {
    return ListCasterParts::cast(
      src, return_value_policy::reference_internal, parent);
  }
  static handle cast(const std::vector<std::shared_ptr<Part>>* src,
                     return_value_policy pol,
                     handle parent)
  {
    return cast(*src, pol, parent);
  }
};
} // namespace detail
} // namespace pybind11

// PLUGIN: dyna_cpp
PYBIND11_PLUGIN(dyna_cpp)
{
  py::module m("dyna_cpp", "c++ python wrapper for ls-dyna module");

  // load numpy
  if (_import_array() < 0) {
    PyErr_SetString(PyExc_ImportError,
                    "numpy.core.multiarray failed to import");
    return nullptr;
  };

  // disable sigantures for documentation
  py::options options;
  options.disable_function_signatures();

  // Node
  py::class_<Node, std::shared_ptr<Node>> node_py(
    m, "Node", qd_node_class_docs);
  node_py
    .def("get_id",
         &Node::get_nodeID,
         py::return_value_policy::take_ownership,
         node_get_id_docs)
    .def("get_coords",
         &Node::get_coords_py,
         "iTimestep"_a = 0,
         py::return_value_policy::take_ownership,
         node_get_coords_docs)
    .def("get_disp",
         &Node::get_disp_py,
         py::return_value_policy::take_ownership,
         node_get_disp_docs)
    .def("get_vel",
         &Node::get_vel_py,
         py::return_value_policy::take_ownership,
         node_get_vel_docs)
    .def("get_accel",
         &Node::get_accel_py,
         py::return_value_policy::take_ownership,
         node_get_accel_docs)
    .def("get_elements",
         &Node::get_elements,
         py::return_value_policy::reference_internal,
         node_get_elements_docs);

  // Element
  py::class_<Element, std::shared_ptr<Element>> element_py(
    m, "Element", element_description);

  py::enum_<Element::ElementType>(element_py, "type", element_type_docs)
    .value("none", Element::ElementType::NONE)
    .value("beam", Element::ElementType::BEAM)
    .value("shell", Element::ElementType::SHELL)
    .value("solid", Element::ElementType::SOLID)
    .export_values();

  element_py
    .def("get_id",
         &Element::get_elementID,
         py::return_value_policy::take_ownership,
         element_get_id_docs)
    .def("get_coords",
         &Element::get_coords_py,
         py::return_value_policy::take_ownership,
         element_get_coords_docs)
    .def("get_energy",
         &Element::get_energy_py,
         py::return_value_policy::take_ownership,
         element_get_energy_docs)
    .def("get_stress_mises",
         &Element::get_stress_mises_py,
         py::return_value_policy::take_ownership,
         element_get_stress_mises_docs)
    .def("get_plastic_strain",
         &Element::get_plastic_strain_py,
         py::return_value_policy::take_ownership,
         element_get_plastic_strain_docs)
    .def("get_strain",
         &Element::get_strain_py,
         py::return_value_policy::take_ownership,
         element_get_strain_docs)
    .def("get_stress",
         &Element::get_stress_py,
         py::return_value_policy::take_ownership,
         element_get_stress_docs)
    .def("get_history_variables",
         &Element::get_history_vars_py,
         py::return_value_policy::take_ownership,
         element_get_history_docs)
    .def("is_rigid",
         &Element::get_is_rigid,
         py::return_value_policy::take_ownership,
         element_get_is_rigid_docs)
    .def("get_estimated_size",
         &Element::get_estimated_element_size,
         py::return_value_policy::take_ownership,
         element_get_estimated_size_docs)
    .def("get_type",
         &Element::get_elementType,
         py::return_value_policy::take_ownership,
         element_get_type_docs)
    .def("get_nodes",
         &Element::get_nodes,
         py::return_value_policy::reference_internal,
         element_get_nodes_docs);

  // Part
  py::class_<Part, std::shared_ptr<Part>> part_py(m, "QD_Part");
  part_py
    .def("get_name",
         &Part::get_name,
         py::return_value_policy::take_ownership,
         part_get_name_docs)
    .def("get_id",
         &Part::get_partID,
         py::return_value_policy::take_ownership,
         part_get_id_docs)
    .def("get_nodes",
         &Part::get_nodes,
         py::return_value_policy::reference_internal,
         part_get_nodes_docs)
    .def("get_elements",
         &Part::get_elements,
         "element_filter"_a = Element::ElementType::NONE,
         py::return_value_policy::reference_internal,
         part_get_elements_docs);

  // DB_Nodes
  py::class_<DB_Nodes, std::shared_ptr<DB_Nodes>> db_nodes_py(
    m, "DB_Nodes", dbnodes_description);
  db_nodes_py
    .def("get_nNodes",
         &DB_Nodes::get_nNodes,
         py::return_value_policy::take_ownership,
         dbnodes_get_nNodes_docs)
    .def("get_nodes",
         &DB_Nodes::get_nodes,
         py::return_value_policy::take_ownership,
         dbnodes_get_nodes_docs)
    .def("get_nodeByID",
         (std::shared_ptr<Node>(DB_Nodes::*)(long)) &
           DB_Nodes::get_nodeByID<long>,
         "id"_a,
         py::return_value_policy::reference_internal,
         dbnodes_get_nodeByID_docs)
    .def("get_nodeByID",
         (std::vector<std::shared_ptr<Node>>(DB_Nodes::*)(py::list)) &
           DB_Nodes::get_nodeByID_py,
         "id"_a,
         py::return_value_policy::reference_internal)
    .def("get_nodeByID",
         (std::vector<std::shared_ptr<Node>>(DB_Nodes::*)(py::tuple)) &
           DB_Nodes::get_nodeByID_py,
         "id"_a,
         py::return_value_policy::reference_internal)
    .def("get_nodeByIndex",
         (std::shared_ptr<Node>(DB_Nodes::*)(long)) &
           DB_Nodes::get_nodeByIndex<long>,
         "index"_a,
         py::return_value_policy::reference_internal,
         dbnodes_get_nodeByIndex_docs)
    .def("get_nodeByIndex",
         (std::vector<std::shared_ptr<Node>>(DB_Nodes::*)(py::list)) &
           DB_Nodes::get_nodeByIndex_py,
         "index"_a,
         py::return_value_policy::reference_internal)
    .def("get_nodeByIndex",
         (std::vector<std::shared_ptr<Node>>(DB_Nodes::*)(py::tuple)) &
           DB_Nodes::get_nodeByIndex_py,
         "index"_a,
         py::return_value_policy::reference_internal);

  // DB_Elements
  py::class_<DB_Elements, std::shared_ptr<DB_Elements>> db_elements_py(
    m, "DB_Elements", dbelems_description);
  db_elements_py
    .def("get_nElements",
         &DB_Elements::get_nElements,
         "element_type"_a = Element::NONE,
         py::return_value_policy::take_ownership,
         dbelems_get_nElements_docs)
    .def("get_elements",
         &DB_Elements::get_elements,
         "element_type"_a = Element::NONE,
         py::return_value_policy::take_ownership,
         get_elements_docs)
    .def(
      "get_elementByID",
      (std::shared_ptr<Element>(DB_Elements::*)(Element::ElementType, long)) &
        DB_Elements::get_elementByID<long>,
      "element_type"_a,
      "id"_a,
      py::return_value_policy::reference_internal)
    .def("get_elementByID",
         (std::vector<std::shared_ptr<Element>>(DB_Elements::*)(
           Element::ElementType, pybind11::list)) &
           DB_Elements::get_elementByID<long>,
         "element_type"_a,
         "id"_a,
         py::return_value_policy::reference_internal)
    .def("get_elementByID",
         (std::vector<std::shared_ptr<Element>>(DB_Elements::*)(
           Element::ElementType, pybind11::tuple)) &
           DB_Elements::get_elementByID<long>,
         "element_type"_a,
         "id"_a,
         py::return_value_policy::reference_internal)
    .def(
      "get_elementByIndex",
      (std::shared_ptr<Element>(DB_Elements::*)(Element::ElementType, long)) &
        DB_Elements::get_elementByIndex<long>,
      "element_type"_a,
      "index"_a,
      py::return_value_policy::reference_internal,
      dbelems_get_elementByIndex_docs)
    .def("get_elementByIndex",
         (std::vector<std::shared_ptr<Element>>(DB_Elements::*)(
           Element::ElementType, pybind11::list)) &
           DB_Elements::get_elementByIndex<long>,
         "element_type"_a,
         "index"_a,
         py::return_value_policy::reference_internal)
    .def("get_elementByIndex",
         (std::vector<std::shared_ptr<Element>>(DB_Elements::*)(
           Element::ElementType, pybind11::tuple)) &
           DB_Elements::get_elementByIndex<long>,
         "element_type"_a,
         "index"_a,
         py::return_value_policy::reference_internal);

  // DB_Parts
  py::class_<DB_Parts, std::shared_ptr<DB_Parts>> db_parts_py(
    m, "DB_Parts", dbparts_description);
  db_parts_py
    .def("get_nParts",
         &DB_Parts::get_nParts,
         py::return_value_policy::take_ownership,
         dbparts_get_nParts_docs)
    .def("get_parts",
         &DB_Parts::get_parts,
         py::return_value_policy::reference_internal,
         dbparts_get_parts_docs)
    .def("get_partByID",
         (std::shared_ptr<Part>(DB_Parts::*)(long)) &
           DB_Parts::get_partByID<long>,
         "id"_a,
         py::return_value_policy::reference_internal,
         dbparts_get_partByID_docs)
    .def("get_partByID",
         (std::vector<std::shared_ptr<Part>>(DB_Parts::*)(py::list)) &
           DB_Parts::get_partByID,
         "id"_a,
         py::return_value_policy::reference_internal)
    .def("get_partByID",
         (std::vector<std::shared_ptr<Part>>(DB_Parts::*)(py::tuple)) &
           DB_Parts::get_partByID,
         "id"_a,
         py::return_value_policy::reference_internal)
    .def("get_partByIndex",
         (std::shared_ptr<Part>(DB_Parts::*)(long)) &
           DB_Parts::get_partByIndex<long>,
         "id"_a,
         py::return_value_policy::reference_internal,
         dbparts_get_partByIndex_docs)
    .def("get_partByIndex",
         (std::vector<std::shared_ptr<Part>>(DB_Parts::*)(py::list)) &
           DB_Parts::get_partByIndex,
         "id"_a,
         py::return_value_policy::reference_internal)
    .def("get_partByIndex",
         (std::vector<std::shared_ptr<Part>>(DB_Parts::*)(py::tuple)) &
           DB_Parts::get_partByIndex,
         "id"_a,
         py::return_value_policy::reference_internal)
    .def("get_partByName",
         &DB_Parts::get_partByName,
         "name"_a,
         py::return_value_policy::reference_internal,
         dbparts_get_partByName_docs);

  // FEMFile
  py::class_<FEMFile, DB_Parts, DB_Nodes, DB_Elements, std::shared_ptr<FEMFile>>
    femfile_py(m, "FEMFile", py::multiple_inheritance());
  femfile_py.def("get_filepath",
                 &FEMFile::get_filepath,
                 py::return_value_policy::take_ownership,
                 femfile_get_filepath_docs);

  // D3plot
  py::class_<D3plot, FEMFile, std::shared_ptr<D3plot>> d3plot_py(
    m, "QD_D3plot", d3plot_description);
  d3plot_py
    .def(py::init<std::string, py::list, bool>(),
         "filepath"_a,
         "read_states"_a = py::list(),
         "use_femzip"_a = false,
         d3plot_constructor)
    .def(py::init<std::string, py::tuple, bool>(),
         "filepath"_a,
         "read_states"_a = py::tuple(),
         "use_femzip"_a = false)
    .def(py::init<std::string, std::string, bool>(),
         "filepath"_a,
         "read_states"_a = std::string(),
         "use_femzip"_a = false)
    .def("info", &D3plot::info, d3plot_info_docs)
    .def("read_states",
         (void (D3plot::*)(std::string)) & D3plot::read_states,
         d3plot_read_states_docs)
    .def("read_states", (void (D3plot::*)(py::list)) & D3plot::read_states)
    .def("read_states", (void (D3plot::*)(py::tuple)) & D3plot::read_states)
    .def("clear",
         (void (D3plot::*)(py::list)) & D3plot::clear,
         "variables"_a = py::list(),
         d3plot_clear_docs)
    .def("clear",
         (void (D3plot::*)(py::tuple)) & D3plot::clear,
         "variables"_a = py::tuple())
    .def("clear",
         (void (D3plot::*)(py::str)) & D3plot::clear,
         "variables"_a = py::str())
    .def("get_timesteps",
         &D3plot::get_timesteps_py,
         py::return_value_policy::take_ownership,
         d3plot_get_timesteps_docs)
    .def("get_nStates",
         &D3plot::get_nStates,
         py::return_value_policy::take_ownership,
         d3plot_get_nStates)
    .def("get_title",
         &D3plot::get_title,
         py::return_value_policy::take_ownership,
         d3plot_get_title_docs);

  // KeyFile
  py::class_<KeyFile, FEMFile, std::shared_ptr<KeyFile>> keyfile_py(
    m, "QD_KeyFile", keyfile_description);
  keyfile_py.def(py::init<std::string>(), "filepath"_a, keyfile_constructor);

  return m.ptr();
}