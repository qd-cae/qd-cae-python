
#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/db/FEMFile.hpp>
#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/db/Part.hpp>
#include <dyna_cpp/dyna/D3plot.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <numpy/arrayobject.h>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace py::literals;

void qd_test_d3plot(std::shared_ptr<D3plot> d3plot) {
  auto vec = d3plot->get_nodeByID(1)->get_disp();
  size_t iRow = 0;
  for (const auto& subvec : vec) {
    std::cout << "iRow:" << iRow << " | " << subvec[0] << "  " << subvec[1]
              << "  " << subvec[2] << std::endl;
    iRow++;
  }
}

PYBIND11_PLUGIN(dyna_cpp) {
  py::module m("dyna_cpp", "ls-dyna c++ python wrapper");

  // load numpy
  if (_import_array() < 0) {
    PyErr_SetString(PyExc_ImportError,
                    "numpy.core.multiarray failed to import");
    return nullptr;
  };

  // Node
  py::class_<Node, std::shared_ptr<Node>> node_py(m, "Node");
  node_py
      .def("get_id", &Node::get_nodeID, py::return_value_policy::take_ownership)
      .def("get_coords", &Node::get_coords_py, "iTimestep"_a = 0,
           py::return_value_policy::take_ownership)
      .def("get_disp", &Node::get_disp_py,
           py::return_value_policy::take_ownership)
      .def("get_vel", &Node::get_vel_py,
           py::return_value_policy::take_ownership)
      .def("get_accel", &Node::get_accel_py,
           py::return_value_policy::take_ownership);
  // TODO get_elements

  // Part
  py::class_<Part, std::shared_ptr<Part>> part_py(m, "Part");
  part_py
      .def("get_name", &Part::get_name, py::return_value_policy::take_ownership)
      .def("get_id", &Part::get_partID,
           py::return_value_policy::take_ownership);
  // TODO get_elements, get_nodes

  // Element
  py::class_<Element, std::shared_ptr<Element>> element_py(m, "Element");
  element_py
      .def("get_coords", &Element::get_coords_py,
           py::return_value_policy::take_ownership)
      .def("is_rigid", &Element::get_is_rigid,
           py::return_value_policy::take_ownership)
      .def("get_estimated_size", &Element::get_estimated_element_size,
           py::return_value_policy::take_ownership);
  // TODO get_nodes, get_coords, get_energy, get_stress_mises,
  // TODO get_plastic_strain, get_strain, get_stress, get_history
  // TODO is_rigid, get_type

  // DB_Nodes
  py::class_<DB_Nodes, std::shared_ptr<DB_Nodes>> db_nodes_py(m, "DB_Nodes");
  db_nodes_py
      .def(
          "get_nodeByID",
          (std::shared_ptr<Node>(DB_Nodes::*)(int)) & DB_Nodes::get_nodeByID_py,
          "id"_a, py::return_value_policy::reference_internal)
      .def("get_nodeByID",
           (std::vector<std::shared_ptr<Node>>(DB_Nodes::*)(py::list)) &
               DB_Nodes::get_nodeByID_py,
           "id"_a, py::return_value_policy::reference)
      .def("get_nodeByID",
           (std::vector<std::shared_ptr<Node>>(DB_Nodes::*)(py::tuple)) &
               DB_Nodes::get_nodeByID_py,
           "id"_a, py::return_value_policy::reference)
      .def("get_nNodes", &DB_Nodes::get_nNodes,
           py::return_value_policy::take_ownership);
  //

  // DB_Elements
  py::class_<DB_Elements, std::shared_ptr<DB_Elements>> db_elements_py(
      m, "DB_Elements");

  // DB_Parts
  py::class_<DB_Parts, std::shared_ptr<DB_Parts>> db_parts_py(m, "DB_Parts");
  db_parts_py.def("get_nParts", &DB_Parts::get_nParts,
                  py::return_value_policy::take_ownership);

  // FEMFile
  py::class_<FEMFile, DB_Parts, DB_Nodes, std::shared_ptr<FEMFile>> femfile_py(
      m, "FEMFile", py::multiple_inheritance());
  femfile_py
      .def("get_filepath", &FEMFile::get_filepath,
           py::return_value_policy::take_ownership)
      .def("is_d3plot", &FEMFile::is_d3plot,
           py::return_value_policy::take_ownership)
      .def("is_keyFile", &FEMFile::is_keyFile,
           py::return_value_policy::take_ownership)
      .def("get_nNodes", &FEMFile::get_nNodes,
           py::return_value_policy::take_ownership);

  // D3plot
  py::class_<D3plot, std::shared_ptr<D3plot>> d3plot_py(m, "D3plot",
                                                        femfile_py);
  d3plot_py
      .def(py::init<std::string, py::list, bool>(), "filepath"_a,
           "read_states"_a = py::list(), "use_femzip"_a = false)
      .def(py::init<std::string, py::tuple, bool>(), "filepath"_a,
           "read_states"_a = py::tuple(), "use_femzip"_a = false)
      .def("get_timesteps", &D3plot::get_timesteps_py,
           py::return_value_policy::take_ownership)
      .def("info", &D3plot::info)
      .def("read_states", (void (D3plot::*)(py::list)) & D3plot::read_states)
      .def("read_states", (void (D3plot::*)(py::tuple)) & D3plot::read_states)
      .def("clear", (void (D3plot::*)(py::list)) & D3plot::clear,
           "variables"_a = py::list())
      .def("clear", (void (D3plot::*)(py::tuple)) & D3plot::clear,
           "variables"_a = py::tuple())
      .def("clear", (void (D3plot::*)(py::str)) & D3plot::clear,
           "variables"_a = py::str());

  // Test
  m.def("qd_test_d3plot", &qd_test_d3plot);

  return m.ptr();
}