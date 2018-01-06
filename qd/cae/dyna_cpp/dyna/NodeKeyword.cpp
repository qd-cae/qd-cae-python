
#include <dyna_cpp/dyna/NodeKeyword.hpp>

namespace qd {

NodeKeyword::NodeKeyword(std::shared_ptr<DB_Nodes> _db_nodes,
                         std::vector<std::string> _lines,
                         int64_t _iLine)
  : Keyword(_lines, _iLine)
  , db_nodes(_db_nodes)
{
  // Do the thing
}

} // NAMESPACE:qd