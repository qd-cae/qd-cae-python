
#include <iomanip>

#include <dyna_cpp/dyna/keyfile/PartKeyword.hpp>

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
  if (db_parts == nullptr)
    return;

  // name
  auto kw_name = to_lower_copy(get_keyword_name());

  // find first card line
  size_t header_size = get_line_index_of_next_card(0);
  size_t iLine = header_size;

  if (iLine == lines.size())
    return;

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
  while (iLine < lines.size()) {

    // depends on a specific card value
    one_more_card = false;

    const auto& line = lines[iLine];

    // if (std::all_of(line.begin(), line.end(), std::isspace))
    //   break;

    // find second card for parsing
    const auto iNextLine = get_line_index_of_next_card(iLine);
    if (iNextLine == lines.size())
      throw(std::runtime_error(
        "Parsing error in line: " +
        std::to_string(static_cast<size_t>(position) + 1 + iLine) +
        "\nerror: Second card is missing."));

    // eventually there are comments inbetween first and second card
    std::string comment_block;
    for (size_t iCLine = 1; iCLine < iNextLine - iLine; ++iCLine)
      comment_block += lines[iLine + iCLine] + '\n';
    comments_between_card0_and_card1.push_back(comment_block);

    const auto& next_line = lines[iNextLine];

    // do the thing (parse card 0 and card 1)
    // then save remaining card data
    try {
      part_name = line.substr(0, 7 * field_size);
      part_id = std::stoi(next_line.substr(0, field_size));
      std::string remaining_data(next_line.begin() + field_size,
                                 next_line.end());

      // find end of part block
      iLine = iNextLine + 1;
      size_t iCardCount = 0;
      while (iCardCount < nAdditionalLines + one_more_card &&
             iLine < lines.size()) {

        if (!is_comment(lines[iLine])) {

          // if ircs field is 1, then we have another card line
          if (is_part_inertia && iCardCount == 0) {
            const auto flag_ircs =
              trim_copy(lines[iLine].substr(4 * field_size, field_size));
            if (flag_ircs.empty() || std::stod(flag_ircs) == 1)
              one_more_card = true;
          }

          ++iCardCount;
        }

        // copy line
        remaining_data += '\n' + lines[iLine++];
      }

      // create part and save data
      db_parts->add_partByID(part_id, part_name);
      part_ids.push_back(part_id);
      unparsed_data.push_back(remaining_data);
    } catch (const std::exception& err) {
      std::cerr << "Parsing error in line: "
                << (static_cast<size_t>(position) + 1 + iLine) << '\n'
                << "error:" << err.what() << '\n'
                << "line :" << line << '\n';
    }

    // update line counter
    ++iLine;
  }

  // trailing shit
  auto iLine_trailing = iLine;
  for (; iLine < lines.size(); ++iLine)
    trailing_lines.push_back(lines[iLine]);

  // remove all lines below keyword
  lines.resize(iLine_trailing);
}

/** Get the keyword as a string
 *
 * @return str : keyword as string (again)
 */
std::string
PartKeyword::str()
{
  std::stringstream ss;

  // write headers
  auto iLine = get_line_index_of_next_card(0);
  for (size_t ii = 0; ii < iLine; ++ii)
    ss << lines[ii] << '\n';
  // for (const auto& entry : lines)
  //   ss << entry << '\n';

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

  const auto kw_name = get_keyword_name();
  bool is_part_inertia = false;
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

  ss.precision(7);
  for (size_t iPart = 0; iPart < part_ids.size(); ++iPart) {
    const auto part = db_parts->get_partByID(part_ids[iPart]);

    // const auto& comment_block = comments_between_card0_and_card1[iPart];

    ss << std::setw(7 * field_size) << part->get_name() << '\n';

    const size_t iLineNextKeyword = get_line_index_of_next_card(iLine);
    for (size_t jLine = iLine + 1; jLine < iLineNextKeyword; ++jLine)
      ss << lines[jLine] << '\n';

    std::string remaining_line_data =
      (iLineNextKeyword < lines.size() &&
       lines[iLineNextKeyword].size() > field_size)
        ? lines[iLineNextKeyword].substr(field_size,
                                         lines[iLineNextKeyword].size())
        : "";

    ss << std::setw(field_size) << part->get_partID() << remaining_line_data
       << '\n';

    iLine = iLineNextKeyword + 1;
    size_t iCardCount = 0;
    bool one_more_card = false;
    try {
      while (iCardCount < nAdditionalLines + one_more_card &&
             iLine < lines.size()) {

        if (!is_comment(lines[iLine])) {

          // if ircs field is 1, then we have another card line
          if (is_part_inertia && iCardCount == 0) {
            const auto flag_ircs =
              trim_copy(lines[iLine].substr(4 * field_size, field_size));
            if (flag_ircs.empty() || std::stod(flag_ircs) == 1)
              one_more_card = true;
          }

          ++iCardCount;
        }
        ss << lines[iLine] << '\n';
        ++iLine;
      }
    } catch (const std::exception& err) {
      throw(std::invalid_argument("Error while trying to convert part data "
                                  "beyond second card to string: " +
                                  std::string(err.what())));
    }
  }

  // write trailing lines
  for (const auto& entry : trailing_lines)
    ss << entry << '\n';

  return ss.str();
}

} // namespace:qd