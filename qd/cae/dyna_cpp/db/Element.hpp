
#ifndef ELEMENT_HPP
#define ELEMENT_HPP

// includes
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace qd {

// forward declarations
class Node;
class DB_Nodes;
class DB_Elements;

class Element
{
  friend class DB_Nodes;

public:
  enum ElementType
  {
    NONE,
    BEAM,
    SHELL,
    SOLID,
    TSHELL
  };

private:
  int32_t elementID;
  int32_t part_id;
  bool is_rigid;
  std::vector<int32_t> node_ids;
  std::vector<float> energy;
  std::vector<float> stress_mises;
  std::vector<float> plastic_strain;
  std::vector<std::vector<float>> strain;
  std::vector<std::vector<float>> stress;
  std::vector<std::vector<float>> history_vars;
  ElementType elemType;
  DB_Elements* db_elements;

  std::mutex _element_mutex;

  void remove_node(int32_t _node_id);

public:
  explicit Element(int32_t _id,
                   int32_t _part_id,
                   ElementType _etype,
                   const std::vector<int32_t>& _node_ids,
                   DB_Elements* db_elements);
  virtual ~Element();
  bool operator<(const Element& other) const;
  inline std::string str()
  {
    return "<Element type:" + std::to_string(elemType) +
           " id:" + std::to_string(elementID) + ">";
  };
  void check() const;

  // getter
  ElementType get_elementType() const;
  int32_t get_elementID() const;
  int32_t get_part_id() const;
  bool get_is_rigid() const;
  float get_estimated_element_size() const; // fast
  size_t get_nNodes() const;
  std::vector<std::shared_ptr<Node>> get_nodes() const;
  const std::vector<int32_t>& get_node_ids() const;
  std::vector<size_t> get_node_indexes() const;
  std::vector<float> get_energy() const;
  std::vector<float> get_stress_mises() const;
  std::vector<float> get_plastic_strain() const;
  std::vector<std::vector<float>> get_coords() const;
  std::vector<std::vector<float>> get_strain() const;
  std::vector<std::vector<float>> get_stress() const;
  std::vector<std::vector<float>> get_history_vars() const;

  // setter
  void set_is_rigid(bool _is_rigid);
  void add_energy(float _energy);
  void add_stress_mises(float _stress_mises);
  void add_plastic_strain(float _plastic_strain);
  void add_stress(std::vector<float> _stress);
  void add_strain(std::vector<float> _strain);
  void add_history_vars(std::vector<float> vars, size_t iTimestep);

  // clear memory
  void clear_energy();
  void clear_plastic_strain();
  void clear_stress();
  void clear_stress_mises();
  void clear_strain();
  void clear_history_vars();
};

} // namespace qd

#endif
