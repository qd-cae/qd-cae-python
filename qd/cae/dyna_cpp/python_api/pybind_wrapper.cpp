
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
#include <pybind11/stl_bind.h>

#include <numpy/arrayobject.h>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace py::literals;

// get the docstrings
#include <dyna_cpp/python_api/docstrings.cpp>

/* This hack ensures that a vector of instances is translated into a list of
 * instances in such a way, that it still references its d3plot in memory. If
 * not doing so then using the instance when the d3plot is dead results in an
 * error.
 */
// DEBUG: Somehow not working ...
/*
using ListCasterBase =
   pybind11::detail::list_caster<std::vector<std::shared_ptr<Node>>,
                                 std::shared_ptr<Node>>;
namespace pybind11 {
namespace detail {
template <>
struct type_caster<std::vector<std::shared_ptr<Node>>> : ListCasterBase {
 static handle cast(const std::vector<std::shared_ptr<Node>> &src,
                    return_value_policy, handle parent) {
   return ListCasterBase::cast(src, return_value_policy::reference_internal,
                               parent);
 }
 static handle cast(const std::vector<std::shared_ptr<Node>> *src,
                    return_value_policy pol, handle parent) {
   return cast(*src, pol, parent);
 }
};
}
}
*/

void
qd_test_d3plot(std::shared_ptr<D3plot> d3plot)
{
  auto vec = d3plot->get_nodeByID(1)->get_disp();
  size_t iRow = 0;
  for (const auto& subvec : vec) {
    std::cout << "iRow:" << iRow << " | " << subvec[0] << "  " << subvec[1]
              << "  " << subvec[2] << std::endl;
    iRow++;
  }
}

PYBIND11_PLUGIN(dyna_cpp)
{
  py::module m("dyna_cpp", "ls-dyna c++ python wrapper");

  // load numpy
  if (_import_array() < 0) {
    PyErr_SetString(PyExc_ImportError,
                    "numpy.core.multiarray failed to import");
    return nullptr;
  };

  // disable sigantures for docs
  py::options options;
  options.disable_function_signatures();

  // Node
  py::class_<Node, std::shared_ptr<Node>> node_py(
    m, "Node", qd_node_class_docs);
  node_py
    .def("get_id",
         &Node::get_nodeID,
         py::return_value_policy::take_ownership,
         get_node_id_docs)
    .def("get_coords",
         &Node::get_coords_py,
         "iTimestep"_a = 0,
         py::return_value_policy::take_ownership,
         get_node_coords_docs)
    .def("get_disp",
         &Node::get_disp_py,
         py::return_value_policy::take_ownership,
         get_node_disp_docs)
    .def("get_vel",
         &Node::get_vel_py,
         py::return_value_policy::take_ownership,
         get_node_vel_docs)
    .def("get_accel",
         &Node::get_accel_py,
         py::return_value_policy::take_ownership,
         get_node_accel_docs)
    .def("get_elements",
         &Node::get_elements,
         py::return_value_policy::reference_internal,
         get_node_elements_docs);

  // Element
  py::class_<Element, std::shared_ptr<Element>> element_py(
    m, "Element", qd_element_class_docs);

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
         get_element_id_docs)
    .def("get_coords",
         &Element::get_coords_py,
         py::return_value_policy::take_ownership,
         get_element_coords_docs)
    .def("get_energy",
         &Element::get_energy_py,
         py::return_value_policy::take_ownership,
         get_element_energy_docs)
    .def("get_stress_mises",
         &Element::get_stress_mises_py,
         py::return_value_policy::take_ownership,
         get_element_stress_mises_docs)
    .def("get_plastic_strain",
         &Element::get_plastic_strain_py,
         py::return_value_policy::take_ownership,
         get_element_plastic_strain_docs)
    .def("get_strain",
         &Element::get_strain_py,
         py::return_value_policy::take_ownership,
         get_element_strain_docs)
    .def("get_stress",
         &Element::get_stress_py,
         py::return_value_policy::take_ownership,
         get_element_stress_docs)
    .def("get_history_variables",
         &Element::get_history_vars_py,
         py::return_value_policy::take_ownership,
         get_element_history_docs)
    .def("is_rigid",
         &Element::get_is_rigid,
         py::return_value_policy::take_ownership,
         get_element_is_rigid_docs)
    .def("get_estimated_size",
         &Element::get_estimated_element_size,
         py::return_value_policy::take_ownership,
         get_element_estimated_size_docs)
    .def("get_type",
         &Element::get_elementType,
         py::return_value_policy::take_ownership,
         get_element_type_docs)
    .def("get_nodes",
         &Element::get_nodes,
         py::return_value_policy::reference_internal,
         get_element_nodes_docs);

  // Part
  py::class_<Part, std::shared_ptr<Part>> part_py(m, "Part");
  part_py
    .def("get_name",
         &Part::get_name,
         py::return_value_policy::take_ownership,
         get_part_name_docs)
    .def("get_id",
         &Part::get_partID,
         py::return_value_policy::take_ownership,
         get_part_id_docs)
    .def("get_nodes",
         &Part::get_nodes,
         py::return_value_policy::reference_internal,
         get_part_nodes_docs)
    .def("get_elements",
         &Part::get_elements,
         "element_filter"_a = Element::ElementType::NONE,
         py::return_value_policy::reference_internal,
         get_part_elements_docs);

  // DB_Nodes
  py::class_<DB_Nodes, std::shared_ptr<DB_Nodes>> db_nodes_py(
    m, "DB_Nodes", dbnodes_description);
  db_nodes_py
    .def("get_nNodes",
         &DB_Nodes::get_nNodes,
         py::return_value_policy::take_ownership,
         dbnodes_get_nNodes_docs)
    /* TODO: array versions (maybe also from numpy array)
.def("get_node_id_from_index", &DB_Nodes::get_id_from_index<long>,
    py::return_value_policy::take_ownership)
.def("get_node_index_from_id", &DB_Nodes::get_index_from_id<long>,
    py::return_value_policy::take_ownership)
    */
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
    m, "DB_Elements");
  db_elements_py
    .def("get_nElements",
         &DB_Elements::get_nElements,
         "element_type"_a = Element::NONE,
         py::return_value_policy::take_ownership)
    .def("get_elementByID",
         &DB_Elements::get_elementByID<long>,
         "element_type"_a,
         "id"_a,
         py::return_value_policy::reference_internal)
    .def("get_elementByIndex",
         &DB_Elements::get_elementByIndex<long>,
         "element_type"_a,
         "index"_a,
         py::return_value_policy::reference_internal);
  // TODO get_elementByID(py::list), get_elementByID(py::tuple)
  // TODO get_elementByIndex(py::list), get_elementByIndex(py::tuple)

  // DB_Parts
  py::class_<DB_Parts, std::shared_ptr<DB_Parts>> db_parts_py(m, "DB_Parts");
  db_parts_py
    .def("get_nParts",
         &DB_Parts::get_nParts,
         py::return_value_policy::take_ownership)
    .def("get_parts",
         &DB_Parts::get_parts,
         py::return_value_policy::reference_internal)
    .def("get_partByID",
         (std::shared_ptr<Part>(DB_Parts::*)(long)) &
           DB_Parts::get_partByID<long>,
         "id"_a,
         py::return_value_policy::reference_internal)
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
         py::return_value_policy::reference_internal)
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
         py::return_value_policy::reference_internal);

  // FEMFile
  py::class_<FEMFile, DB_Parts, DB_Nodes, DB_Elements, std::shared_ptr<FEMFile>>
    femfile_py(m, "FEMFile", py::multiple_inheritance());
  femfile_py.def("get_filepath",
                 &FEMFile::get_filepath,
                 py::return_value_policy::take_ownership);

  // D3plot
  py::class_<D3plot, FEMFile, std::shared_ptr<D3plot>> d3plot_py(m, "D3plot");
  d3plot_py
    .def(py::init<std::string, py::list, bool>(),
         "filepath"_a,
         "read_states"_a = py::list(),
         "use_femzip"_a = false)
    .def(py::init<std::string, py::tuple, bool>(),
         "filepath"_a,
         "read_states"_a = py::tuple(),
         "use_femzip"_a = false)
    .def(py::init<std::string, std::string, bool>(),
         "filepath"_a,
         "read_states"_a = std::string(),
         "use_femzip"_a = false)
    .def("info", &D3plot::info)
    .def("read_states", (void (D3plot::*)(std::string)) & D3plot::read_states)
    .def("read_states", (void (D3plot::*)(py::list)) & D3plot::read_states)
    .def("read_states", (void (D3plot::*)(py::tuple)) & D3plot::read_states)
    .def("clear",
         (void (D3plot::*)(py::list)) & D3plot::clear,
         "variables"_a = py::list())
    .def("clear",
         (void (D3plot::*)(py::tuple)) & D3plot::clear,
         "variables"_a = py::tuple())
    .def("clear",
         (void (D3plot::*)(py::str)) & D3plot::clear,
         "variables"_a = py::str())
    .def("get_timesteps",
         &D3plot::get_timesteps_py,
         py::return_value_policy::take_ownership)
    .def("get_nStates",
         &D3plot::get_nStates,
         py::return_value_policy::take_ownership)
    .def(
      "get_title", &D3plot::get_title, py::return_value_policy::take_ownership);

  // KeyFile
  py::class_<KeyFile, FEMFile, std::shared_ptr<KeyFile>> keyfile_py(m,
                                                                    "KeyFile");
  keyfile_py.def(py::init<std::string>(), "filepath"_a);

  // Test
  // m.def("qd_test_d3plot", &qd_test_d3plot);

  return m.ptr();
}