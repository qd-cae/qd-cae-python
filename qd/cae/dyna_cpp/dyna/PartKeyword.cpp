
#include <iomanip>

#include <dyna_cpp/dyna/PartKeyword.hpp>

namespace qd {

/** Constructor of a part keyword
 *
 * @param _db_parts : parent database
 * @param _lines : keyword lines
 * @param _iLine : line index in file
 */
PartKeyword::PartKeyword(DB_Parts* _db_parts,
                         const std::vector<std::string>& _lines,
                         int64_t _iLine)
  : Keyword(_lines, _iLine)
  , db_parts(_db_parts)
{

  field_size = has_long_fields() ? 20 : 10;
}

/** Load the data from the string data
 *
 * This function loads the data from the string data.
 * The string data is removed while the data is being parsed.
 */
void
PartKeyword::load()
{
  // name
  auto kw_name = to_lower_copy(get_keyword_name());

  // find first card line
  size_t header_size = iCard_to_iLine(0, false);
  size_t iLine = header_size;

  // how much data to read
  bool is_part_inertia = false;
  bool one_more_card = false;
  size_t nAdditionalLines = 0;
  if (kw_name.find("inertia", 6) != std::string::npos) {
    nAdditionalLines += 3;
    is_part_inertia = true;
  }
  if (kw_name.find("reposition", 6) != std::string::npos)
    ++nAdditionalLines;
  if (kw_name.find("contact", 6) != std::string::npos)
    ++nAdditionalLines;
  if (kw_name.find("print", 6) != std::string::npos)
    ++nAdditionalLines;
  if (kw_name.find("attachment_nodes", 6) != std::string::npos)
    ++nAdditionalLines;

#ifdef QD_DEBUG
  std::cout << "PartKeyword\nname: " << kw_name
            << "\nnAdditionalLines: " << nAdditionalLines
            << "\nis_part_inertia: " << is_part_inertia << '\n';
#endif

  std::string part_name;
  int32_t part_id;
  for (; iLine < lines.size(); iLine += 2 + nAdditionalLines + one_more_card) {
    one_more_card = false;

    // TODO: comment treatment inbetween
    const auto& line = lines[iLine];
    if (line.empty())
      break;

    if (iLine + 1 >= lines.size())
      throw(std::runtime_error(
        "Parsing error in line: " +
        std::to_string(static_cast<size_t>(line_index) + iLine) +
        "\nerror: Part is supposed to have a card with a part id."));
    const auto& next_line = lines[iLine + 1];

    try {
      part_name = line.substr(0, 7 * field_size);
      part_id = std::stoi(next_line.substr(0, field_size));
      std::string remaining_data(next_line.begin() + field_size,
                                 next_line.end());
      for (size_t iExtraLine = 0; iExtraLine < nAdditionalLines + one_more_card;
           ++iExtraLine) {
        remaining_data += '\n' + lines[iLine + 2 + iExtraLine];
        if (is_part_inertia && iExtraLine == 1) {
          auto flag_ircs = trim(
            lines[iLine + 2 + iExtraLine].substr(4 * field_size, field_size));
          if (flag_ircs.empty() || std::stod(flag_ircs) == 1)
            one_more_card = true;
        }
      }

      db_parts->add_partByID(part_id, part_name);
      part_ids.push_back(part_id);
      unparsed_data.push_back(remaining_data);

    } catch (const std::exception& err) {
      std::cerr << "Parsing error in line: "
                << (static_cast<size_t>(line_index) + iLine) << '\n'
                << "error:" << err.what() << '\n'
                << "line :" << line << '\n';
    }
  }

  // trailing shit
  for (; iLine < lines.size(); ++iLine)
    trailing_lines.push_back(lines[iLine]);

  // remove all lines below keyword
  lines.resize(header_size);
}

/** Get the keyword as a string
 *
 * @return str : keyword as string (again)
 */
std::string
PartKeyword::str() const
{
  std::stringstream ss;

  // write headers
  for (const auto& entry : lines)
    ss << entry << '\n';

  // write parts
  switch (Keyword::field_alignment) {
    case (Keyword::Align::LEFT):
      ss.setf(std::ios::left);
      break;
    case (Keyword::Align::RIGHT):
      ss.setf(std::ios::right);
      break;
    default:
      ss.setf(std::ios::internal);
  }
  ss.precision(7);
  for (size_t iPart = 0; iPart < part_ids.size(); ++iPart) {
    auto part = db_parts->get_partByID(part_ids[iPart]);
    ss << std::setw(7 * field_size) << part->get_name() << '\n'
       << std::setw(field_size) << part->get_partID() << unparsed_data[iPart]
       << '\n';
  }

  // write trailing lines
  for (const auto& entry : trailing_lines)
    ss << entry << '\n';

  return ss.str();
}

} // namespace:qd