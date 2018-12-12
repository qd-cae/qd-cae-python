
#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/db/FEMFile.hpp>
#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/db/Part.hpp>
#include <dyna_cpp/dyna/binout/Binout.hpp>
#include <dyna_cpp/dyna/d3plot/D3plot.hpp>
#include <dyna_cpp/dyna/d3plot/FemzipBuffer.hpp>
#include <dyna_cpp/dyna/d3plot/RawD3plot.hpp>
#include <dyna_cpp/dyna/keyfile/ElementKeyword.hpp>
#include <dyna_cpp/dyna/keyfile/KeyFile.hpp>
#include <dyna_cpp/dyna/keyfile/Keyword.hpp>
#include <dyna_cpp/dyna/keyfile/NodeKeyword.hpp>
#include <dyna_cpp/dyna/keyfile/PartKeyword.hpp>
#include <dyna_cpp/math/Tensor.hpp>
#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/PythonUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

extern "C"
{
#include <pybind11/numpy.h>
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// #include <numpy/arrayobject.h>

#include <memory>
#include <string>
#include <vector>

using namespace pybind11::literals;

// get the docstrings
#include <dyna_cpp/python_api/docstrings.cpp>

/** Computes strides from a tensor shape
 *
 * @param shape
 * @return strides
 */
template<typename T>
std::vector<size_t>
shape_to_strides(const std::vector<size_t>& shape)
{
  std::vector<size_t> strides(shape.size());
  if (strides.size() != 0)
    strides.back() = static_cast<size_t>(sizeof(T));

  for (int32_t iDim = static_cast<int32_t>(strides.size()) - 2; iDim >= 0;
       --iDim)
    strides[iDim] =
      strides[iDim + 1] * static_cast<pybind11::ssize_t>(shape[iDim + 1]);

  return std::move(strides);
}

/*
namespace pybind11 {
namespace detail {

template<typename T>
struct type_caster<qd::Tensor<T>>
{
  PYBIND11_TYPE_CASTER(qd::Tensor<T>, _("Tensor<T>"));

public:

  // Conversion part 1 (Python -> C++)
  bool load(pybind11::handle src, bool convert)
  {
    if (!convert && !pybind11::array_t<T>::check_(src))
      return false;

    auto buf = pybind11::array_t<T,
                                 pybind11::array::c_style |
                                   pybind11::array::forcecast>::ensure(src);
    if (!buf)
      return false;

    const auto n_dims = buf.ndim();

    std::vector<size_t> shape(n_dims);
    for (int i = 0; i < n_dims; i++)
      shape[i] = buf.shape()[i];

    value = qd::Tensor<T>(shape, buf.data()); // copies! TODO fix

    return true;
  }

  // Conversion part 2 (C++ -> Python)
  static pybind11::handle cast(qd::Tensor<T>& src,
                               const pybind11::return_value_policy& policy,
                               pybind11::handle& parent)
  {

    const auto& shape = src.get_shape();
    auto strides = shape_to_strides<T>(shape);

    pybind11::array a(shape, std::move(strides), src.get_data().data());

    return a.release();
  }

};

} // namespace detail
} // namespace pybind11
*/

namespace qd {

/* Utility functions */
auto cast_kw = [](std::shared_ptr<Keyword> instance) {
  switch (instance->get_keyword_type()) {

    case (Keyword::KeywordType::NODE):
      return pybind11::cast(std::static_pointer_cast<NodeKeyword>(instance));

    case (Keyword::KeywordType::ELEMENT):
      return pybind11::cast(std::static_pointer_cast<ElementKeyword>(instance));

    case (Keyword::KeywordType::PART):
      return pybind11::cast(std::static_pointer_cast<PartKeyword>(instance));

    case (Keyword::KeywordType::INCLUDE_PATH):
      return pybind11::cast(
        std::static_pointer_cast<IncludePathKeyword>(instance));

    case (Keyword::KeywordType::INCLUDE):
      return pybind11::cast(std::static_pointer_cast<IncludeKeyword>(instance));

    default:
      return pybind11::cast(instance);
  }
};

// template<typename T>
// pybind11::array_t<T> f

/*========= PLUGIN: dyna_cpp =========*/
// pybind11 2.2
// PYBIND11_MODULE(dyna_cpp, m)
// old style
PYBIND11_PLUGIN(dyna_cpp)
{
  pybind11::module m("dyna_cpp", "c++ python wrapper for ls-dyna module");

  // pybind11 2.2
  // m.doc() = "c++ python wrapper for ls-dyna module";

  // load numpy
  // if (_import_array() < 0) {
  //   PyErr_SetString(PyExc_ImportError,
  //                   "numpy.core.multiarray failed to import");
  //   return nullptr;
  // };

  // disable sigantures for documentation
  pybind11::options options;
  options.disable_function_signatures();

  // Tensor
  pybind11::class_<Tensor<float>, std::shared_ptr<Tensor<float>>> tensor_f32_py(
    m, "Tensor_f32", pybind11::buffer_protocol());
  tensor_f32_py
    .def_buffer([](Tensor<float>& m) -> pybind11::buffer_info {
      const auto& shape = m.get_shape();
      auto strides = shape_to_strides<float>(shape);

      return pybind11::buffer_info(
        m.get_data().data(),                          // Pointer to buffer
        (pybind11::ssize_t)sizeof(float),             // Size of one scalar
        pybind11::format_descriptor<float>::format(), // Python struct-style
        static_cast<pybind11::ssize_t>(shape.size()), // Number of dims
        shape,                                        // Buffer dimensions
        strides // Strides (in bytes) for each index
      );
    })
    .def("print", &Tensor<float>::print)
    .def("shape", &Tensor<float>::get_shape);

  // Tensor
  pybind11::class_<Tensor<double>, std::shared_ptr<Tensor<double>>>
    tensor_f64_py(m, "Tensor_f64", pybind11::buffer_protocol());
  tensor_f64_py
    .def_buffer([](Tensor<double>& m) -> pybind11::buffer_info {
      const auto& shape = m.get_shape();
      auto strides = shape_to_strides<double>(shape);

      return pybind11::buffer_info(
        m.get_data().data(),                           // Pointer to buffer
        (pybind11::ssize_t)sizeof(double),             // Size of one scalar
        pybind11::format_descriptor<double>::format(), // Python struct-style
        static_cast<pybind11::ssize_t>(shape.size()),  // Number of dims
        shape,                                         // Buffer dimensions
        strides // Strides (in bytes) for each index
      );
    })
    .def("print", &Tensor<double>::print)
    .def("shape", &Tensor<double>::get_shape);

  pybind11::class_<Tensor<int8_t>, std::shared_ptr<Tensor<int8_t>>>
    tensor_i8_py(m, "Tensor_int8", pybind11::buffer_protocol());
  tensor_i8_py
    .def_buffer([](Tensor<int8_t>& m) -> pybind11::buffer_info {
      const auto& shape = m.get_shape();
      auto strides = shape_to_strides<int8_t>(shape);

      return pybind11::buffer_info(
        m.get_data().data(),                           // Pointer to buffer
        (pybind11::ssize_t)sizeof(int8_t),             // Size of one scalar
        pybind11::format_descriptor<int8_t>::format(), // Python struct-style
        static_cast<pybind11::ssize_t>(shape.size()),  // Number of dims
        shape,                                         // Buffer dimensions
        strides // Strides (in bytes) for each index
      );
    })
    .def("print", &Tensor<int8_t>::print)
    .def("shape", &Tensor<int8_t>::get_shape);

  pybind11::class_<Tensor<int16_t>, std::shared_ptr<Tensor<int16_t>>>
    tensor_i16_py(m, "Tensor_int16", pybind11::buffer_protocol());
  tensor_i16_py
    .def_buffer([](Tensor<int16_t>& m) -> pybind11::buffer_info {
      const auto& shape = m.get_shape();
      auto strides = shape_to_strides<int16_t>(shape);

      return pybind11::buffer_info(
        m.get_data().data(),                            // Pointer to buffer
        (pybind11::ssize_t)sizeof(int16_t),             // Size of one scalar
        pybind11::format_descriptor<int16_t>::format(), // Python struct-style
        static_cast<pybind11::ssize_t>(shape.size()),   // Number of dims
        shape,                                          // Buffer dimensions
        strides // Strides (in bytes) for each index
      );
    })
    .def("print", &Tensor<int16_t>::print)
    .def("shape", &Tensor<int16_t>::get_shape);

  pybind11::class_<Tensor<int32_t>, std::shared_ptr<Tensor<int32_t>>>
    tensor_i32_py(m, "Tensor_int32", pybind11::buffer_protocol());
  tensor_i32_py
    .def_buffer([](Tensor<int32_t>& m) -> pybind11::buffer_info {
      const auto& shape = m.get_shape();
      auto strides = shape_to_strides<int32_t>(shape);

      return pybind11::buffer_info(
        m.get_data().data(),                            // Pointer to buffer
        (pybind11::ssize_t)sizeof(int32_t),             // Size of one scalar
        pybind11::format_descriptor<int32_t>::format(), // Python struct-style
        static_cast<pybind11::ssize_t>(shape.size()),   // Number of dims
        shape,                                          // Buffer dimensions
        strides // Strides (in bytes) for each index
      );
    })
    .def("print", &Tensor<int32_t>::print)
    .def("shape", &Tensor<int32_t>::get_shape);

  pybind11::class_<Tensor<int64_t>, std::shared_ptr<Tensor<int64_t>>>
    tensor_i64_py(m, "Tensor_int64", pybind11::buffer_protocol());
  tensor_i64_py
    .def_buffer([](Tensor<int64_t>& m) -> pybind11::buffer_info {
      const auto& shape = m.get_shape();
      auto strides = shape_to_strides<int64_t>(shape);

      return pybind11::buffer_info(
        m.get_data().data(),                            // Pointer to buffer
        (pybind11::ssize_t)sizeof(int64_t),             // Size of one scalar
        pybind11::format_descriptor<int64_t>::format(), // Python struct-style
        static_cast<pybind11::ssize_t>(shape.size()),   // Number of dims
        shape,                                          // Buffer dimensions
        strides // Strides (in bytes) for each index
      );
    })
    .def("print", &Tensor<int64_t>::print)
    .def("shape", &Tensor<int64_t>::get_shape);

  pybind11::class_<Tensor<uint8_t>, std::shared_ptr<Tensor<uint8_t>>>
    tensor_uint8_py(m, "Tensor_uint8", pybind11::buffer_protocol());
  tensor_uint8_py
    .def_buffer([](Tensor<uint8_t>& m) -> pybind11::buffer_info {
      const auto& shape = m.get_shape();
      auto strides = shape_to_strides<uint8_t>(shape);

      return pybind11::buffer_info(
        m.get_data().data(),                            // Pointer to buffer
        (pybind11::ssize_t)sizeof(uint8_t),             // Size of one scalar
        pybind11::format_descriptor<uint8_t>::format(), // Python struct-style
        static_cast<pybind11::ssize_t>(shape.size()),   // Number of dims
        shape,                                          // Buffer dimensions
        strides // Strides (in bytes) for each index
      );
    })
    .def("print", &Tensor<uint8_t>::print)
    .def("shape", &Tensor<uint8_t>::get_shape);

  pybind11::class_<Tensor<uint16_t>, std::shared_ptr<Tensor<uint16_t>>>
    tensor_uint16_py(m, "Tensor_uint16", pybind11::buffer_protocol());
  tensor_uint16_py
    .def_buffer([](Tensor<uint16_t>& m) -> pybind11::buffer_info {
      const auto& shape = m.get_shape();
      auto strides = shape_to_strides<uint16_t>(shape);

      return pybind11::buffer_info(
        m.get_data().data(),                             // Pointer to buffer
        (pybind11::ssize_t)sizeof(uint16_t),             // Size of one scalar
        pybind11::format_descriptor<uint16_t>::format(), // Python struct-style
        static_cast<pybind11::ssize_t>(shape.size()),    // Number of dims
        shape,                                           // Buffer dimensions
        strides // Strides (in bytes) for each index
      );
    })
    .def("print", &Tensor<uint16_t>::print)
    .def("shape", &Tensor<uint16_t>::get_shape);

  pybind11::class_<Tensor<uint32_t>, std::shared_ptr<Tensor<uint32_t>>>
    tensor_uint32_py(m, "Tensor_uint32", pybind11::buffer_protocol());
  tensor_uint32_py
    .def_buffer([](Tensor<uint32_t>& m) -> pybind11::buffer_info {
      const auto& shape = m.get_shape();
      auto strides = shape_to_strides<uint32_t>(shape);

      return pybind11::buffer_info(
        m.get_data().data(),                             // Pointer to buffer
        (pybind11::ssize_t)sizeof(uint32_t),             // Size of one scalar
        pybind11::format_descriptor<uint32_t>::format(), // Python struct-style
        static_cast<pybind11::ssize_t>(shape.size()),    // Number of dims
        shape,                                           // Buffer dimensions
        strides // Strides (in bytes) for each index
      );
    })
    .def("print", &Tensor<uint32_t>::print)
    .def("shape", &Tensor<uint32_t>::get_shape);

  pybind11::class_<Tensor<uint64_t>, std::shared_ptr<Tensor<uint64_t>>>
    tensor_uint64_py(m, "Tensor_uint64", pybind11::buffer_protocol());
  tensor_uint64_py
    .def_buffer([](Tensor<uint64_t>& m) -> pybind11::buffer_info {
      const auto& shape = m.get_shape();
      auto strides = shape_to_strides<uint64_t>(shape);

      return pybind11::buffer_info(
        m.get_data().data(),                             // Pointer to buffer
        (pybind11::ssize_t)sizeof(uint64_t),             // Size of one scalar
        pybind11::format_descriptor<uint64_t>::format(), // Python struct-style
        static_cast<pybind11::ssize_t>(shape.size()),    // Number of dims
        shape,                                           // Buffer dimensions
        strides // Strides (in bytes) for each index
      );
    })
    .def("print", &Tensor<uint64_t>::print)
    .def("shape", &Tensor<uint64_t>::get_shape);

  /*
  pybind11::class_<Tensor<size_t>, std::shared_ptr<Tensor<size_t>>>
    tensor_size_t_py(m, "Tensor_size_t", pybind11::buffer_protocol());
  tensor_size_t_py
    .def_buffer([](Tensor<size_t>& m) -> pybind11::buffer_info {
      const auto& shape = m.get_shape();
      auto strides = shape_to_strides<size_t>(shape);

      return pybind11::buffer_info(
        m.get_data().data(),                           // Pointer to buffer
        (pybind11::ssize_t)sizeof(size_t),             // Size of one scalar
        pybind11::format_descriptor<size_t>::format(), // Python struct-style
        static_cast<pybind11::ssize_t>(shape.size()),  // Number of dims
        shape,                                         // Buffer dimensions
        strides // Strides (in bytes) for each index
      );
    })
    .def("print", &Tensor<size_t>::print)
    .def("shape", &Tensor<size_t>::get_shape);
    */

  // Node
  pybind11::class_<Node, std::shared_ptr<Node>> node_py(
    m, "Node", qd_node_class_docs);
  node_py
    .def("__repr__",
         [](std::shared_ptr<Node> self) {
           return "<qd.cae.dyna.Node id=" + std::to_string(self->get_nodeID()) +
                  '>';
         },
         pybind11::return_value_policy::take_ownership)
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
    .def("__repr__",
         [](std::shared_ptr<Element> self) {
           std::string etype_str;
           switch (self->get_elementType()) {
             case (Element::ElementType::SHELL):
               etype_str = "shell";
               break;
             case (Element::ElementType::SOLID):
               etype_str = "solid";
               break;
             case (Element::ElementType::BEAM):
               etype_str = "beam";
               break;
             case (Element::ElementType::TSHELL):
               etype_str = "tshell";
               break;
             case (Element::ElementType::NONE):
               etype_str = "none";
               break;
             default:
               etype_str = "unknown";
               break;
           }

           return "<qd.cae.dyna.Element id=" +
                  std::to_string(self->get_elementID()) + " type=" + etype_str +
                  '>';
         },
         pybind11::return_value_policy::take_ownership)
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
         pybind11::return_value_policy::reference_internal,
         element_get_part_id_docs)
    .def("get_node_ids",
         &Element::get_node_ids,
         pybind11::return_value_policy::reference_internal,
         element_get_node_ids_docs);

  // Part
  pybind11::class_<Part, std::shared_ptr<Part>> part_py(m, "QD_Part");
  part_py
    .def("__repr__",
         [](std::shared_ptr<Part> self) {
           return "<qd.cae.dyna.Part id=" + std::to_string(self->get_partID()) +
                  " name=" + self->get_name() + '>';
         },
         pybind11::return_value_policy::take_ownership)
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
         part_get_elements_docs)
    .def("get_nNodes", &Part::get_nNodes, part_get_nNodes_docs)
    .def("get_nElements", &Part::get_nElements, part_get_nElements_docs)
    .def("get_element_node_ids",
         //  &Part::get_element_node_ids,
         [](std::shared_ptr<Part> self,
            Element::ElementType element_type,
            size_t nNodes) {
           return py::tensor_to_nparray(
             self->get_element_node_ids(element_type, nNodes));
         },
         "element_type"_a,
         "nNodes"_a,
         pybind11::return_value_policy::reference_internal,
         part_get_element_node_ids_docs)
    .def("get_element_node_indexes",
         //  &Part::get_element_node_indexes,
         [](std::shared_ptr<Part> self,
            Element::ElementType element_type,
            size_t nNodes) {
           return py::tensor_to_nparray(
             self->get_element_node_indexes(element_type, nNodes));
         },
         "element_type"_a,
         "nNodes"_a,
         pybind11::return_value_policy::reference_internal,
         part_get_element_node_indexes_docs)
    .def("get_node_ids",
         [](std::shared_ptr<Part> self) {
           return py::tensor_to_nparray(self->get_node_ids());
         },
         part_get_node_ids_docs)
    .def("get_node_indexes",
         [](std::shared_ptr<Part> self) {
           return py::tensor_to_nparray(self->get_node_indexes());
         },
         part_get_node_indexes_docs)
    .def("get_element_ids",
         [](std::shared_ptr<Part> self, Element::ElementType element_filter) {
           return py::tensor_to_nparray(self->get_element_ids(element_filter));
         },
         "element_filter"_a = Element::ElementType::NONE,
         part_get_element_ids_docs);

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
         //  pybind11::call_guard<pybind11::gil_scoped_release>(),
         dbnodes_get_nodes_docs)
    .def("get_nodeByID",
         (std::shared_ptr<Node>(DB_Nodes::*)(long)) &
           DB_Nodes::get_nodeByID<long>,
         "id"_a,
         pybind11::return_value_policy::reference_internal,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         dbnodes_get_nodeByID_docs)
    .def("get_nodeByID",
         [](std::shared_ptr<DB_Nodes> _db_nodes, pybind11::list _ids) {
           std::vector<int32_t> tmp = qd::py::container_to_vector<int32_t>(
             _ids, "An entry of the list was not a fully fledged integer.");

           pybind11::gil_scoped_release release;
           return _db_nodes->get_nodeByID(tmp);
         },
         "id"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_nodeByID",
         [](std::shared_ptr<DB_Nodes> _db_nodes, pybind11::tuple _ids) {
           auto tmp = qd::py::container_to_vector<int32_t>(
             _ids, "An entry of the list was not a fully fledged integer.");

           pybind11::gil_scoped_release release;
           return _db_nodes->get_nodeByID(tmp);
         },
         "id"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_nodeByIndex",
         (std::shared_ptr<Node>(DB_Nodes::*)(long)) &
           DB_Nodes::get_nodeByIndex<long>,
         "index"_a,
         pybind11::return_value_policy::reference_internal,
         dbnodes_get_nodeByIndex_docs)
    .def("get_nodeByIndex",
         [](std::shared_ptr<DB_Nodes> _db_nodes, pybind11::list _indexes) {
           auto tmp = qd::py::container_to_vector<int32_t>(
             _indexes, "An entry of the list was not a fully fledged integer.");

           pybind11::gil_scoped_release release;
           return _db_nodes->get_nodeByIndex(tmp);
         },
         "index"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_nodeByIndex",
         [](std::shared_ptr<DB_Nodes> _db_nodes, pybind11::tuple _indexes) {
           auto tmp = qd::py::container_to_vector<int32_t>(
             _indexes, "An entry of the list was not a fully fledged integer.");

           pybind11::gil_scoped_release release;
           return _db_nodes->get_nodeByIndex(tmp);
         },
         "index"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_node_coords",
         [](std::shared_ptr<DB_Nodes> db_nodes) {
           return py::tensor_to_nparray(db_nodes->get_node_coords());
         },
         pybind11::return_value_policy::take_ownership,
         dbnodes_get_node_coords_docs)
    .def("get_node_velocity",
         [](std::shared_ptr<DB_Nodes> db_nodes) {
           return py::tensor_to_nparray(db_nodes->get_node_velocity());
         },
         pybind11::return_value_policy::take_ownership,
         dbnodes_get_node_velocity_docs)
    .def("get_node_acceleration",
         [](std::shared_ptr<DB_Nodes> db_nodes) {
           return py::tensor_to_nparray(db_nodes->get_node_acceleration());
         },
         pybind11::return_value_policy::take_ownership,
         dbnodes_get_node_acceleration_docs)
    .def("get_node_ids",
         [](std::shared_ptr<DB_Nodes> db_nodes) {
           return py::tensor_to_nparray(db_nodes->get_node_ids());
         },
         pybind11::return_value_policy::take_ownership,
         dbnodes_get_node_ids_docs);

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
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         get_elements_docs)
    .def(
      "get_elementByID",
      (std::shared_ptr<Element>(DB_Elements::*)(Element::ElementType, long)) &
        DB_Elements::get_elementByID<long>,
      "element_type"_a,
      "id"_a,
      // pybind11::call_guard<pybind11::gil_scoped_release>(),
      pybind11::return_value_policy::reference_internal)
    .def("get_elementByID",
         [](std::shared_ptr<DB_Elements> _db_elems,
            Element::ElementType _eType,
            pybind11::list _list) {
           auto tmp = qd::py::container_to_vector<long>(
             _list, "An entry of the id list was not an integer.");

           pybind11::gil_scoped_release release;
           return _db_elems->get_elementByID(_eType, tmp);
         },
         "element_type"_a,
         "id"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_elementByID",
         [](std::shared_ptr<DB_Elements> _db_elems,
            Element::ElementType _eType,
            pybind11::tuple _list) {
           auto tmp = qd::py::container_to_vector<long>(
             _list, "An entry of the id list was not an integer.");

           pybind11::gil_scoped_release release;
           return _db_elems->get_elementByID(_eType, tmp);
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
         [](std::shared_ptr<DB_Elements> _db_elems,
            Element::ElementType _eType,
            pybind11::tuple _list) {
           auto tmp = qd::py::container_to_vector<long>(
             _list, "An entry of the index list was not an integer.");

           pybind11::gil_scoped_release release;
           return _db_elems->get_elementByIndex(_eType, tmp);
         },
         "element_type"_a,
         "index"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_elementByIndex",
         [](std::shared_ptr<DB_Elements> _db_elems,
            Element::ElementType _eType,
            pybind11::list _list) {
           auto tmp = qd::py::container_to_vector<long>(
             _list, "An entry of the index list was not an integer.");

           pybind11::gil_scoped_release release;
           return _db_elems->get_elementByIndex(_eType, tmp);
         },
         "element_type"_a,
         "index"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_element_ids",
         [](std::shared_ptr<DB_Elements> self,
            Element::ElementType element_filter) {
           return py::tensor_to_nparray(self->get_element_ids());
         },
         "element_filter"_a = Element::ElementType::NONE,
         dbelems_get_element_ids_docs)
    .def("get_element_node_ids",
         [](std::shared_ptr<DB_Elements> self,
            Element::ElementType element_type,
            size_t n_nodes) {
           return py::tensor_to_nparray(
             self->get_element_node_ids(element_type, n_nodes));
         },
         "element_type"_a = Element::ElementType::NONE,
         "n_nodes"_a,
         dbelems_get_element_node_ids_docs)
    .def("get_element_energy",
         [](std::shared_ptr<DB_Elements> self,
            Element::ElementType element_filter) {
           return py::tensor_to_nparray(
             self->get_element_energy(element_filter));
         },
         "element_filter"_a = Element::ElementType::NONE,
         dbelems_get_element_energy)
    .def("get_element_plastic_strain",
         [](std::shared_ptr<DB_Elements> self,
            Element::ElementType element_filter) {
           return py::tensor_to_nparray(
             self->get_element_plastic_strain(element_filter));
         },
         "element_filter"_a = Element::ElementType::NONE,
         dbelems_get_plastic_strain)
    .def("get_element_stress_mises",
         [](std::shared_ptr<DB_Elements> self,
            Element::ElementType element_filter) {
           return py::tensor_to_nparray(
             self->get_element_stress_mises(element_filter));
         },
         "element_filter"_a = Element::ElementType::NONE,
         dbelems_get_element_stress_mises)
    .def("get_element_stress",
         [](std::shared_ptr<DB_Elements> self,
            Element::ElementType element_filter) {
           return py::tensor_to_nparray(
             self->get_element_stress(element_filter));
         },
         "element_filter"_a = Element::ElementType::NONE,
         dbelems_get_element_stress)
    .def("get_element_strain",
         [](std::shared_ptr<DB_Elements> self,
            Element::ElementType element_filter) {
           return py::tensor_to_nparray(
             self->get_element_strain(element_filter));
         },
         "element_filter"_a = Element::ElementType::NONE,
         dbelems_get_element_strain)
    .def("get_element_coords",
         [](std::shared_ptr<DB_Elements> self,
            Element::ElementType element_filter) {
           return py::tensor_to_nparray(
             self->get_element_coords(element_filter));
         },
         "element_filter"_a = Element::ElementType::NONE,
         dbelems_get_element_coords)
    .def(
      "get_element_history_vars",
      [](std::shared_ptr<DB_Elements> self, Element::ElementType element_type) {
        return py::tensor_to_nparray(
          self->get_element_history_vars(element_type));
      },
      "element_type"_a = Element::ElementType::NONE,
      dbelems_get_element_history_vars);

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
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         dbparts_get_parts_docs)
    .def("get_partByID",
         (std::shared_ptr<Part>(DB_Parts::*)(long)) &
           DB_Parts::get_partByID<long>,
         "id"_a,
         pybind11::return_value_policy::reference_internal,
         dbparts_get_partByID_docs)
    .def("get_partByID",
         [](std::shared_ptr<DB_Parts> _db_parts, pybind11::list _list) {
           auto tmp = qd::py::container_to_vector<int32_t>(
             _list, "An entry of the index list was not an integer.");

           pybind11::gil_scoped_release release;
           return _db_parts->get_partByID(tmp);
         },
         "id"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_partByID",
         [](std::shared_ptr<DB_Parts> _db_parts, pybind11::tuple _list) {
           auto tmp = qd::py::container_to_vector<int32_t>(
             _list, "An entry of the index list was not an integer.");

           pybind11::gil_scoped_release release;
           return _db_parts->get_partByID(tmp);
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
         [](std::shared_ptr<DB_Parts> _db_parts, pybind11::list _list) {
           auto tmp = qd::py::container_to_vector<int32_t>(
             _list, "An entry of the index list was not an integer.");

           pybind11::gil_scoped_release release;
           return _db_parts->get_partByIndex(tmp);
         },
         "index"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_partByIndex",
         [](std::shared_ptr<DB_Parts> _db_parts, pybind11::tuple _list) {
           auto tmp = qd::py::container_to_vector<int32_t>(
             _list, "An entry of the index list was not an integer.");

           pybind11::gil_scoped_release release;
           return _db_parts->get_partByIndex(tmp);
         },
         "index"_a,
         pybind11::return_value_policy::reference_internal)
    .def("get_partByName",
         &DB_Parts::get_partByName,
         "name"_a,
         pybind11::return_value_policy::reference_internal,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         dbparts_get_partByName_docs);

  // FEMFile
  pybind11::
    class_<FEMFile, DB_Parts, DB_Nodes, DB_Elements, std::shared_ptr<FEMFile>>
      femfile_py(m, "FEMFile", pybind11::multiple_inheritance());
  femfile_py.def("get_filepath",
                 &FEMFile::get_filepath,
                 pybind11::return_value_policy::take_ownership,
                 // pybind11::call_guard<pybind11::gil_scoped_release>(),
                 femfile_get_filepath_docs);

  // D3plot
  pybind11::class_<D3plot, FEMFile, std::shared_ptr<D3plot>> d3plot_py(
    m, "QD_D3plot", d3plot_description);
  d3plot_py
    // .def(pybind11::init<std::string, std::string>(),
    //      "filepath"_a,
    //      "read_states"_a = std::string(),
    //      pybind11::call_guard<pybind11::gil_scoped_release>(),
    //      d3plot_constructor)
    // .def(pybind11::init([](std::string _filepath, pybind11::list _variables)
    // {
    //        auto tmp = qd::py::container_to_vector<std::string>(
    //          _variables, "An entry of read_states was not of type str");

    //        pybind11::gil_scoped_release release;
    //        return std::make_shared<D3plot>(_filepath, tmp);
    //      }),
    //      "filepath"_a,
    //      "read_states"_a = pybind11::list())
    // .def(pybind11::init([](std::string _filepath, pybind11::tuple _variables)
    // {
    //        auto tmp = qd::py::container_to_vector<std::string>(
    //          _variables, "An entry of read_states was not of type str");

    //        pybind11::gil_scoped_release release;
    //        return std::make_shared<D3plot>(_filepath, tmp);
    //      }),
    //      "filepath"_a,
    //      "read_states"_a = pybind11::tuple())
    // DEPRECATED BEGIN
    .def("__init__",
         [](D3plot& instance,
            std::string _filepath,
            pybind11::list _variables,
            bool use_femzip) {
           // std::cout << "DeprecationWarning: Argument 'use_femzip' is not "
           //              "needed anymore and will be "
           //              "removed in the future.\n";

           auto tmp = qd::py::container_to_vector<std::string>(
             _variables, "An entry of read_states was not of type str");

           pybind11::gil_scoped_release release;
           new (&instance) D3plot(_filepath, tmp, use_femzip);
         },
         "filepath"_a,
         "read_states"_a = pybind11::list(),
         "use_femzip"_a = false)
    .def("__init__",
         [](D3plot& instance,
            std::string _filepath,
            pybind11::tuple _variables,
            bool use_femzip) {
           // std::cout << "DeprecationWarning: Argument 'use_femzip' is not "
           //              "needed anymore and will be "
           //              "removed in the future.\n";

           auto tmp = qd::py::container_to_vector<std::string>(
             _variables, "An entry of read_states was not of type str");

           pybind11::gil_scoped_release release;
           new (&instance) D3plot(_filepath, tmp, use_femzip);
         },
         "filepath"_a,
         "read_states"_a = pybind11::tuple(),
         "use_femzip"_a = false)
    .def("__init__",
         [](D3plot& instance,
            std::string _filepath,
            std::string var_name,
            bool use_femzip) {
           //  std::cout << "DeprecationWarning: Argument 'use_femzip' is not
           //  "
           //               "needed anymore and will be "
           //               "removed in the future.\n";

           pybind11::gil_scoped_release release;
           new (&instance) D3plot(_filepath, var_name, use_femzip);
         },
         "filepath"_a,
         "read_states"_a = std::string(),
         "use_femzip"_a = false,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         d3plot_constructor)
    // DEPRECATED END
    .def("info", &D3plot::info, d3plot_info_docs)
    .def("read_states",
         (void (D3plot::*)(const std::string&)) & D3plot::read_states,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         d3plot_read_states_docs)
    .def("read_states",
         [](std::shared_ptr<D3plot> _d3plot, pybind11::list _list) {
           auto tmp = qd::py::container_to_vector<std::string>(
             _list, "An entry of read_states was not of type str");

           pybind11::gil_scoped_release release;
           _d3plot->read_states(tmp);
         })
    .def("read_states",
         [](std::shared_ptr<D3plot> _d3plot, pybind11::tuple _list) {
           auto tmp = qd::py::container_to_vector<std::string>(
             _list, "An entry of read_states was not of type str");

           pybind11::gil_scoped_release release;
           _d3plot->read_states(tmp);
         })
    .def("clear",
         [](std::shared_ptr<D3plot> _d3plot, pybind11::list _list) {
           auto tmp = qd::py::container_to_vector<std::string>(
             _list, "An entry of read_states was not of type str");

           pybind11::gil_scoped_release release;
           _d3plot->clear(tmp);
         },
         "variables"_a = pybind11::list(),
         d3plot_clear_docs)
    .def("clear",
         [](std::shared_ptr<D3plot> _d3plot, pybind11::tuple _list) {
           auto tmp = qd::py::container_to_vector<std::string>(
             _list, "An entry of read_states was not of type str");

           pybind11::gil_scoped_release release;
           _d3plot->clear(tmp);
         },
         "variables"_a = pybind11::tuple())
    .def("clear",
         (void (D3plot::*)(const std::string&)) & D3plot::clear,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         "variables"_a = pybind11::str())
    .def("get_timesteps",
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
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
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
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         rawd3plot_get_string_data_docs)
    .def("_get_int_names",
         &RawD3plot::get_int_names,
         pybind11::return_value_policy::take_ownership,
         rawd3plot_get_int_names_docs)
    .def("_get_int_data",
         [](std::shared_ptr<RawD3plot> self, std::string& name) {
           return py::tensor_to_nparray(self->get_int_data(name));
         },
         "name"_a,
         rawd3plot_get_int_data_docs)
    .def("_get_float_names",
         &RawD3plot::get_float_names,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         rawd3plot_get_float_names_docs)
    .def("_get_float_data",
         [](std::shared_ptr<RawD3plot> self, std::string& name) {
           return py::tensor_to_nparray(self->get_float_data(name));
         },
         "name"_a,
         rawd3plot_get_float_data_docs)
    .def("_set_float_data",
         [](std::shared_ptr<RawD3plot> _d3plot,
            std::string _entry_name,
            pybind11::array_t<float> _data) {
           auto shape_sizet = std::vector<size_t>(_data.ndim());
           for (size_t ii = 0; ii < _data.ndim(); ++ii)
             shape_sizet[ii] = _data.shape()[ii];

           pybind11::gil_scoped_release release;
           _d3plot->set_float_data(_entry_name, shape_sizet, _data.data());
         },
         "name"_a,
         "data"_a)
    .def("_set_int_data",
         [](std::shared_ptr<RawD3plot> _d3plot,
            std::string _entry_name,
            pybind11::array_t<int> _data) {
           auto shape_sizet = std::vector<size_t>(_data.ndim());
           for (size_t ii = 0; ii < _data.ndim(); ++ii)
             shape_sizet[ii] = _data.shape()[ii];

           pybind11::gil_scoped_release release;
           _d3plot->set_int_data(_entry_name, shape_sizet, _data.data());
         },
         "name"_a,
         "data"_a)
    .def("_set_string_data",
         [](std::shared_ptr<RawD3plot> _d3plot,
            std::string _entry_name,
            pybind11::list _data) {
           auto data = qd::py::container_to_vector<std::string>(_data);

           pybind11::gil_scoped_release release;
           _d3plot->set_string_data(_entry_name, data);
         },
         "name"_a,
         "data"_a)
    .def("_set_string_data",
         [](std::shared_ptr<RawD3plot> _d3plot,
            std::string _entry_name,
            pybind11::tuple _data) {
           auto data = qd::py::container_to_vector<std::string>(_data);

           pybind11::gil_scoped_release release;
           _d3plot->set_string_data(_entry_name, data);
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
         "position"_a = 0,
         keyword_constructor_docs)
    .def(pybind11::init<std::vector<std::string>, int64_t>(),
         "lines"_a,
         "position"_a = 0)
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
                 " - card index (int)\n"
                 " - card and field index (int,int)\n"
                 " - card index, field index and field size (int,int,int)\n"
                 " - field name (str)\n"
                 " - field name and field size (str,int)\n"));
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
    .def_property("position",
                  &Keyword::get_position,
                  &Keyword::set_position,
                  keyword_position_docs)
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
    .def_property("field_size",
                  &Keyword::get_field_size,
                  &Keyword::set_field_size<int64_t>,
                  keyfile_field_size_description)
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
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         node_keyword_get_nodes_docs)
    .def("get_node_ids",
         &NodeKeyword::get_node_ids,
         pybind11::return_value_policy::take_ownership,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         node_keyword_get_node_ids_docs)
    .def("load",
         &NodeKeyword::load,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         node_keyword_load_docs);

  element_keyword_py
    .def("get_elements",
         &ElementKeyword::get_elements,
         pybind11::return_value_policy::take_ownership,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         element_keyword_get_elements_docs)
    .def("get_nElements",
         &ElementKeyword::get_nElements,
         pybind11::return_value_policy::take_ownership,
         element_keyword_get_nElements_docs)
    .def("add_elementByNodeID",
         [](std::shared_ptr<ElementKeyword> self,
            int64_t element_id,
            int64_t part_id,
            const std::vector<int32_t>& node_ids,
            const std::string& additional_card_data) {
           return self->add_elementByNodeID<int64_t>(
             element_id, part_id, node_ids, { additional_card_data });
         },
         "element_id"_a,
         "part_id"_a,
         "node_ids"_a,
         "additional_card_data"_a = "",
         pybind11::return_value_policy::take_ownership,
         element_keyword_add_elementByNodeID_docs)
    .def("add_elementByNodeID",
         &ElementKeyword::add_elementByNodeID<int64_t>,
         "element_id"_a,
         "part_id"_a,
         "node_ids"_a,
         "additional_card_data"_a = "",
         pybind11::return_value_policy::take_ownership)
    .def("add_elementByNodeIndex",
         [](std::shared_ptr<ElementKeyword> self,
            int64_t element_id,
            int64_t part_id,
            const std::vector<size_t>& node_indexes,
            const std::string& additional_card_data) {
           return self->add_elementByNodeIndex<int64_t>(
             element_id, part_id, node_indexes, { additional_card_data });
         },
         "element_id"_a,
         "part_id"_a,
         "node_indexes"_a,
         "additional_card_data"_a = "",
         pybind11::return_value_policy::take_ownership,
         element_keyword_add_elementByNodeIndex_docs)
    .def("add_elementByNodeIndex",
         &ElementKeyword::add_elementByNodeIndex<int64_t>,
         "id"_a,
         "part_id"_a,
         "node_indexes"_a,
         "additional_card_data"_a = "",
         pybind11::return_value_policy::take_ownership,
         element_keyword_add_elementByNodeIndex_docs)
    .def("load",
         &ElementKeyword::load,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         element_keyword_load_docs);

  part_keyword_py
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
    .def("add_part",
         (std::shared_ptr<Part>(PartKeyword::*)(
           int64_t, const std::string&, const std::vector<std::string>&)) &
           PartKeyword::add_part<int64_t>,
         "id"_a,
         "name"_a = "",
         "additional_lines"_a = std::vector<std::string>(),
         pybind11::return_value_policy::take_ownership,
         part_keyword_add_part_docs)
    .def("get_parts",
         &PartKeyword::get_parts,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         part_keyword_get_parts_docs)
    .def("get_nParts", &PartKeyword::get_nParts, part_keyword_get_nParts_docs)
    .def("load",
         &PartKeyword::load,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         part_keyword_load_docs);

  include_path_keyword_py
    .def("is_relative",
         &IncludePathKeyword::is_relative,
         pybind11::return_value_policy::take_ownership,
         include_path_is_relative_docs)
    .def("get_include_dirs",
         &IncludePathKeyword::get_include_dirs,
         pybind11::return_value_policy::take_ownership,
         include_path_keyword_get_include_dirs_docs);

  include_keyword_py
    .def("get_includes",
         &IncludeKeyword::get_includes,
         pybind11::return_value_policy::take_ownership,
         include_keyword_get_includes_docs)
    // .def("load", &IncludeKeyword::load, include_keyword_load_docs)
    .def("load",
         [](std::shared_ptr<IncludeKeyword> self) { self->load(); },
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         include_keyword_load_docs);

  // KeyFile
  pybind11::class_<KeyFile, FEMFile, std::shared_ptr<KeyFile>> keyfile_py(
    m, "QD_KeyFile", keyfile_description);

  keyfile_py
    .def("__init__",
         [](KeyFile& instance,
            const std::string& _filepath,
            bool read_generic_keywords,
            bool parse_mesh,
            bool load_includes) {
           new (&instance) KeyFile(
             _filepath, read_generic_keywords, parse_mesh, load_includes);
           if (!str_has_content(_filepath))
             instance.load();
         },
         "filepath"_a = std::string(),
         "read_keywords"_a = true,
         "parse_mesh"_a = false,
         "load_includes"_a = false,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         keyfile_constructor)
    .def("__init__",
         [](KeyFile& instance,
            const std::string& _filepath,
            bool read_generic_keywords,
            bool parse_mesh,
            bool load_includes,
            double encryption_detection_threshold) {
           if (encryption_detection_threshold != 0.7)
             std::cout << "DeprecationWarning: Argument 'encryption_detection' "
                          "is not needed "
                          "anymore and will be "
                          "removed in the future.\n";

           pybind11::gil_scoped_release release;
           new (&instance) KeyFile(
             _filepath, read_generic_keywords, parse_mesh, load_includes);
           if (!str_has_content(_filepath))
             instance.load();
         },
         "filepath"_a = std::string(),
         "read_keywords"_a = true,
         "parse_mesh"_a = false,
         "load_includes"_a = false,
         "encryption_detection"_a = 0.7)
    .def("__str__",
         &KeyFile::str,
         pybind11::return_value_policy::take_ownership,
         keyfile_str_description)
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
      pybind11::return_value_policy::take_ownership,
      keyfile_getitem_description)
    .def("keys",
         &KeyFile::keys,
         pybind11::return_value_policy::take_ownership,
         keyfile_keys_description)
    .def("save",
         &KeyFile::save_txt,
         "filepath"_a,
         // pybind11::call_guard<pybind11::gil_scoped_release>(),
         keyfile_save_description)
    .def("remove_keyword",
         [](std::shared_ptr<KeyFile> self,
            const std::string& name,
            int64_t index) { self->remove_keyword(name, index); },
         "name"_a,
         "index"_a,
         pybind11::return_value_policy::take_ownership,
         keyfile_remove_keyword_description)
    .def("remove_keyword",
         [](std::shared_ptr<KeyFile> self, const std::string& name) {
           self->remove_keyword(name);
         },
         "name"_a,
         pybind11::return_value_policy::take_ownership)
    .def("add_keyword",
         [](std::shared_ptr<KeyFile> self,
            const std::string& lines,
            int64_t position) {
           return cast_kw(
             self->add_keyword(string_to_lines(lines, true), position));
         },
         "lines"_a,
         "position"_a = -1,
         pybind11::return_value_policy::take_ownership)
    .def("add_keyword",
         [](std::shared_ptr<KeyFile> self,
            const std::vector<std::string>& lines,
            int64_t position) {
           return cast_kw(self->add_keyword(lines, position));
         },
         "lines"_a,
         "position"_a = -1,
         pybind11::return_value_policy::take_ownership,
         keyfile_add_keyword_description)
    .def("get_includes",
         &KeyFile::get_includes,
         pybind11::return_value_policy::take_ownership,
         keyfile_get_includes_description)
    .def("get_include_dirs",
         [](std::shared_ptr<KeyFile> self) {
           return self->get_include_dirs(true);
         },
         pybind11::return_value_policy::take_ownership,
         keyfile_get_include_dirs_description)
    .def("get_end_keyword_position",
         &KeyFile::get_end_keyword_position,
         pybind11::return_value_policy::take_ownership);

// Binout
#ifdef QD_USE_C_BINOUT
  const char* empty_description = { "\0" };
  pybind11::class_<Binout, std::shared_ptr<Binout>> binout_py(
    m, "QD_Binout", empty_description);

  pybind11::enum_<Binout::EntryType>(binout_py, "entry_type")
    .value("unknown", Binout::EntryType::UNKNOWN)
    .value("directory", Binout::EntryType::DIRECTORY)
    .value("int8", Binout::EntryType::INT8)
    .value("int16", Binout::EntryType::INT16)
    .value("int32", Binout::EntryType::INT32)
    .value("int64", Binout::EntryType::INT64)
    .value("uint8", Binout::EntryType::UINT8)
    .value("uint16", Binout::EntryType::UINT16)
    .value("uint32", Binout::EntryType::UINT32)
    .value("uint64", Binout::EntryType::UINT64)
    .value("float32", Binout::EntryType::FLOAT32)
    .value("float64", Binout::EntryType::FLOAT64)
    .value("link", Binout::EntryType::LINK)
    .export_values();

  binout_py.def(pybind11::init<std::string>(), "filepath"_a)
    .def("cd", &Binout::cd, "path"_a)
    .def("exists", &Binout::exists, "path"_a)
    .def("has_children", &Binout::has_children, "path"_a)
    .def("is_variable", &Binout::is_variable, "path"_a)
    .def("get_children", &Binout::get_children, "path"_a = ".")
    .def("get_type_id", &Binout::get_type_id, "path"_a)
    .def(
      "read_variable",
      [](std::shared_ptr<Binout> self, const std::string& path) {
        // what do we have?
        auto type_id = self->get_type_id(path);

        // magic
        switch (type_id) {
          case Binout::EntryType::DIRECTORY:
            throw(std::invalid_argument(
              "Path points to a directory, not a variable."));
          case Binout::EntryType::INT8:
            return py::tensor_to_nparray(self->read_variable<int8_t>(path));
          case Binout::EntryType::INT16:
            return py::tensor_to_nparray(self->read_variable<int16_t>(path));
          case Binout::EntryType::INT32:
            return py::tensor_to_nparray(self->read_variable<int32_t>(path));
          case Binout::EntryType::INT64:
            return py::tensor_to_nparray(self->read_variable<int64_t>(path));
          case Binout::EntryType::UINT8:
            return py::tensor_to_nparray(self->read_variable<uint8_t>(path));
          case Binout::EntryType::UINT16:
            return py::tensor_to_nparray(self->read_variable<uint16_t>(path));
          case Binout::EntryType::UINT32:
            return py::tensor_to_nparray(self->read_variable<uint32_t>(path));
          case Binout::EntryType::UINT64:
            return py::tensor_to_nparray(self->read_variable<uint64_t>(path));
          case Binout::EntryType::FLOAT32:
            return py::tensor_to_nparray(self->read_variable<float>(path));
          case Binout::EntryType::FLOAT64:
            return py::tensor_to_nparray(self->read_variable<double>(path));
          case Binout::EntryType::LINK:
            throw(
              std::invalid_argument("Path points to a link, not a variable."));
          default:
            throw(std::invalid_argument("Path points to something ... I "
                                        "honestly don't know what exactly."));
        }
      },
      "path"_a);
#endif

  // module functions
  m.def("get_file_entropy",
        [](std::string _filepath) {
          std::vector<char> buffer = read_binary_file(_filepath);
          return get_entropy(buffer);
        },
        pybind11::return_value_policy::take_ownership,
        // pybind11::call_guard<pybind11::gil_scoped_release>(),
        module_get_file_entropy_description);
#ifdef QD_USE_FEMZIP
  m.def("is_femzipped",
        &FemzipBuffer::is_femzipped,
        pybind11::return_value_policy::take_ownership);
#endif

#ifdef QD_VERSION
  m.attr("__version__") = QD_VERSION;
#endif

  return m.ptr();
}

} //  namespace qd
