
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>

#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/db/Part.hpp>
#include <dyna_cpp/dyna/ElementKeyword.hpp>
#include <dyna_cpp/dyna/KeyFile.hpp>
#include <dyna_cpp/dyna/NodeKeyword.hpp>
#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

namespace qd {

/**
 * Constructor for a LS-Dyna input file.
 */
KeyFile::KeyFile() {}

/** Constructor for reading a LS-Dyna input file.
 *
 * @param string filepath : filepath of a key file to read
 */
KeyFile::KeyFile(const std::string& _filepath,
                 bool _load_includes,
                 double _encryption_detection,
                 bool _parse_mesh)
  : FEMFile(_filepath)
  , load_includes(_load_includes)
  , encryption_detection_threshold(_encryption_detection)
{
  // check encryption
  if (encryption_detection_threshold < 0 || encryption_detection_threshold > 1)
    throw(std::invalid_argument(
      "Encryption detection threshold must be between 0 and 1."));

  // Read the mesh
  // this->read_mesh(this->get_filepath());
  try {
    this->parse_file(this->get_filepath(), _parse_mesh);
  } catch (const std::exception& ex) {
    std::cout << "Error:" << ex.what() << '\n';
  }
}

/** Parse a keyfile
 *
 * @param _filepath
 * @param _parse_mesh : whether the mesh shall be parsed
 */
void
KeyFile::parse_file(const std::string& _filepath, bool _parse_mesh)
{

  // File directory for Includes
  std::string directory = "";
  size_t pos = _filepath.find_last_of("/\\");
  if (pos != std::string::npos)
    directory = _filepath.substr(0, pos) + "/";
#ifdef QD_DEBUG
  std::cout << "Basic directory for *INCLUDE: " << directory << std::endl;
#endif

  // read file
  std::vector<char> char_buffer = read_binary_file(_filepath);
#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  // test for encryption
  if ((get_entropy(char_buffer) / 8.) > this->encryption_detection_threshold) {
#ifdef QD_DEBUG
    std::cout << "Skipping file " << _filepath << " with normalized entropy of "
              << (get_entropy(char_buffer) / 8) << std::endl;
#endif
    return;
  }

  // convert buffer into blocks
  size_t iLine = 0;
  std::string last_keyword;
  std::vector<std::string> line_buffer;
  std::vector<std::string> line_buffer_tmp;
  std::queue<std::tuple<std::vector<std::string>, size_t, std::string>>
    buffer_queue;
  std::queue<size_t> buffer_iLine_queue;

  std::string line;
  std::stringstream st(std::string(char_buffer.begin(), char_buffer.end()));
  for (; std::getline(st, line); ++iLine) {

    // remove windows file ending ... I hate it ...
    if (line.size() != 0)
      if (line[line.size() - 1] == '\r')
        line.pop_back();

    // new keyword
    if (line[0] == '*') {

      if (!line_buffer.empty() && !last_keyword.empty()) {

        // transfer possible header for following keyword (see function)
        transfer_comment_header(line_buffer, line_buffer_tmp);

        // get type
        auto kw_type = Keyword::determine_keyword_type(last_keyword);

        // elements will be done after parsing since we might miss parts
        // or nodes
        if (_parse_mesh && kw_type == Keyword::KeywordType::ELEMENT) {
          buffer_queue.push(std::make_tuple(line_buffer, iLine, last_keyword));
        } else {
          auto kw =
            create_keyword(line_buffer,
                           kw_type,
                           iLine - line_buffer.size() - line_buffer_tmp.size(),
                           _parse_mesh);
          keywords[kw->get_keyword_name()].push_back(kw);
        }

        // transfer cropped data
        line_buffer = line_buffer_tmp;
      }

      // we always trim keywords
      trim_right(line);
      last_keyword = line;

    } // IF:line[0] == '*'

    // we stupidly add every line to the buffer
    line_buffer.push_back(line);

  } // for:line

  // allocate last block
  if (!line_buffer.empty()) {
    auto kw =
      create_keyword(line_buffer,
                     Keyword::determine_keyword_type(last_keyword),
                     iLine - line_buffer.size() - line_buffer_tmp.size(),
                     _parse_mesh);
    keywords[kw->get_keyword_name()].push_back(kw);
  }

  // get rid of work in queue
  while (buffer_queue.size() > 0) {

    auto& data = buffer_queue.front();
    const auto& block = std::get<0>(data);
    iLine = std::get<1>(data);
    last_keyword = std::get<2>(data);

    auto kw = create_keyword(block,
                             Keyword::determine_keyword_type(last_keyword),
                             iLine - block.size() - block.size(),
                             _parse_mesh);
    keywords[kw->get_keyword_name()].push_back(kw);

    // hihihi popping ...
    buffer_queue.pop();
  }
}

/** Extract a comment header, which may belong to another keyword block
 *
 * @param _old : old string buffer with lines containing coments
 * @param _new : buffer in which the extracted comments will be written
 *
 * The routine does the following:
 * *KEYWORD
 * I'm some data
 * $-------------------------------------
 * $ I should be removed from the upper
 * $ because im the header for the lower
 * $-------------------------------------
 * *ANOTHER_KEYWORD
 *
 * If there are comments DIRECTLY ABOVE a keyword, they ought to be transferred
 * to the following one, because people just defined a header. People never
 * define comments at the end of a keyword, and if they do they suck!
 */
void
KeyFile::transfer_comment_header(std::vector<std::string>& _old,
                                 std::vector<std::string>& _new)
{
#ifdef QD_DEBUG
  std::cout << "KeyFile::transfer_comment_header ... ";
#endif

  size_t iCount = 0;
  size_t nTransferLines = 0;
  if (_old.size() > 0)
    for (size_t iCount = _old.size() - 1; iCount > 0 && _old[iCount][0] == '$';
         --iCount)
      ++nTransferLines;

  _new.resize(nTransferLines);
  std::copy(_old.end() - nTransferLines, _old.end(), _new.begin());
  //_new = std::vector<std::string>(_old.end() - nTransferLines, _old.end());
  _old.resize(_old.size() - nTransferLines);

#ifdef QD_DEBUG
  std::cout << "done" << '\n';
#endif
}

/** Create a keyword from it's line buffer
 *
 * @param _lines buffer
 * @param _keyword_name name flag
 * @param _iLine line index of the block
 */
std::shared_ptr<Keyword>
KeyFile::create_keyword(const std::vector<std::string>& _lines,
                        Keyword::KeywordType _keyword_type,
                        size_t _iLine,
                        bool _parse_mesh)
{

#ifdef QD_DEBUG
  std::cout << "Creating:" << _keyword_name << '\n';
#endif

  // NODE
  if (_parse_mesh && _keyword_type == Keyword::KeywordType::NODE)
    return std::make_shared<NodeKeyword>(
      this->get_db_nodes(), _lines, static_cast<int64_t>(_iLine));
  // ELEMENT
  else if (_parse_mesh && _keyword_type == Keyword::KeywordType::ELEMENT)
    return std::make_shared<ElementKeyword>(
      this->get_db_elements(), _lines, static_cast<int64_t>(_iLine));
  // GENERIC
  else
    return std::make_shared<Keyword>(_lines, _iLine);
}

/** Resolve an include
 *
 * @param _filepath path of the include
 * @return filepath resolved filepath
 */
std::string
KeyFile::resolve_include(const std::string& _filepath)
{

  if (check_ExistanceAndAccess(_filepath))
    return _filepath;

  for (const auto& dir : base_dirs) {
    if (check_ExistanceAndAccess(dir + _filepath))
      return dir + _filepath;
  }

  throw(std::runtime_error("Can not resolve include:" + _filepath));
}

/** Read the mesh from the file given in the filepath
 *
 * @param string filepath : filepath of a key file to read
 */
void
KeyFile::read_mesh(const std::string& _filepath)
{
#ifdef QD_DEBUG
  std::cout << " === Parsing File: " << _filepath << std::endl;
#endif

  // File directory for Includes
  std::string directory = "";
  size_t pos = _filepath.find_last_of("/\\");
  if (pos != std::string::npos)
    directory = _filepath.substr(0, pos) + "/";
#ifdef QD_DEBUG
  std::cout << "Basic directory for *INCLUDE: " << directory << std::endl;
#endif

// Read the lines
#ifdef QD_DEBUG
  std::cout << "Filling IO-Buffer ... " << std::flush;
#endif

  // read file
  std::vector<char> _buffer = read_binary_file(_filepath);
#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  // test for encryption
  if ((get_entropy(_buffer) / 8) > this->encryption_detection_threshold) {
#ifdef QD_DEBUG
    std::cout << "Skipping file " << _filepath << " with normalized entropy of "
              << (get_entropy(_buffer) / 8) << std::endl;
#endif
    return;
  }

  // convert buffer and release memory
  std::vector<std::string> lines = convert_chars_to_lines(_buffer);
  _buffer.clear();
  _buffer.shrink_to_fit();

  // Get databases
  auto db_parts = this->get_db_parts();
  auto db_nodes = this->get_db_nodes();
  auto db_elements = this->get_db_elements();

  // Time to do the thing
  KeyFile::KeywordType keyword = KeywordType::NONE;
  std::string line;
  std::string line_trimmed;
  std::vector<float> coords(3);
  std::vector<int32_t> elemNodes_beam(2, 0);
  std::vector<int32_t> elemNodes_shell(4, 0);
  std::vector<int32_t> elemNodes_solid(8, 0);
  int32_t id = -1;
  int32_t partID = -1;
  size_t iCardLine = 0;
  std::string title;
  bool line_has_keyword = false;

  for (size_t iLine = 0; iLine != lines.size(); iLine++) {
    // Remove comments, etc
    // line = preprocess_string_dyna(lines[iLine]);
    line = lines[iLine];

    // Skip comment lines
    if (line[0] == '$')
      continue;

    line_trimmed = trim_copy(line);
    line_has_keyword = (line_trimmed.find('*') != std::string::npos);

    /* INCLUDE */
    if (line_trimmed == "*INCLUDE") {
      keyword = KeywordType::INCLUDE;
#ifdef QD_DEBUG
      std::cout << "*INCLUDE in line: " << (iLine + 1) << std::endl;
#endif
    } else if (keyword == KeywordType::INCLUDE && this->load_includes) {
      this->read_mesh(directory +
                      line_trimmed); // basic directory is this file's
      keyword = KeywordType::NONE;
    }

    /* NODES */
    if (line_trimmed == "*NODE") {
      keyword = KeywordType::NODE;
#ifdef QD_DEBUG
      std::cout << "Starting *NODE in line: " << (iLine + 1) << std::endl;
#endif
    } else if ((keyword == KeywordType::NODE) && !line_has_keyword &&
               (!line_trimmed.empty())) {
      try {
        coords[0] = std::stof(line.substr(8, 16));
        coords[1] = std::stof(line.substr(24, 16));
        coords[2] = std::stof(line.substr(40, 16));

        db_nodes->add_node(std::stoi(line.substr(0, 8)), coords);
      } catch (const std::exception& ex) {
        std::cerr << "Error reading node in line " << (iLine + 1) << ": "
                  << ex.what() << std::endl;
        keyword = KeywordType::NODE;
      } catch (...) {
        std::cerr << "Error reading node in line " << (iLine + 1)
                  << ": Unknown error." << std::endl;
        keyword = KeywordType::NODE;
      }
    } else if ((keyword == KeywordType::NODE) &&
               (line_has_keyword | line.empty())) {
      keyword = KeywordType::NONE;
#ifdef QD_DEBUG
      std::cout << "*NODE finished in line: " << (iLine + 1) << std::endl;
#endif
    }

    /* ELEMENTS SHELL */
    if (line_trimmed == "*ELEMENT_SHELL") {
      keyword = KeywordType::ELEMENT_SHELL;
#ifdef QD_DEBUG
      std::cout << "Starting *ELEMENT_SHELL in line: " << (iLine + 1)
                << std::endl;
#endif
    } else if ((keyword == KeywordType::ELEMENT_SHELL) && !line_has_keyword &&
               (!line.empty())) {

      try {
        id = std::stoi(line.substr(0, 8));
        partID = std::stoi(line.substr(8, 8));
        elemNodes_shell[0] = std::stoi(line.substr(16, 8));
        elemNodes_shell[1] = std::stoi(line.substr(24, 8));
        elemNodes_shell[2] = std::stoi(line.substr(32, 8));
        elemNodes_shell[3] = std::stoi(line.substr(40, 8));
        db_elements->add_element_byKeyFile(
          Element::SHELL, id, partID, elemNodes_shell);
      } catch (const std::exception& ex) {
        std::cerr << "Error reading shell in line " << (iLine + 1) << ":"
                  << ex.what() << std::endl;
        keyword = KeywordType::NONE;
      } catch (...) {
        std::cerr << "Error reading shell in line " << (iLine + 1)
                  << ": Unknown error." << std::endl;
        keyword = KeywordType::NONE;
      }
    } else if ((keyword == KeywordType::ELEMENT_SHELL) &&
               (line_has_keyword | line.empty())) {
      keyword = KeywordType::NONE;
#ifdef QD_DEBUG
      std::cout << "*ELEMENT_SHELL finished in line: " << (iLine + 1)
                << std::endl;
#endif
    }

    /* ELEMENTS SOLID */
    if (line_trimmed == "*ELEMENT_SOLID") {
      keyword = KeywordType::ELEMENT_SOLID;
      iCardLine = 0;
#ifdef QD_DEBUG
      std::cout << "Starting *ELEMENT_SOLID in line: " << (iLine + 1)
                << std::endl;
#endif
    } else if ((keyword == KeywordType::ELEMENT_SOLID) && !line_has_keyword &&
               !line.empty()) {
      try {
        if (iCardLine == 0) {
          id = std::stoi(line.substr(0, 8));
          partID = std::stoi(line.substr(8, 8));
          ++iCardLine;
        } else if (iCardLine == 1) {
          elemNodes_solid[0] = std::stoi(line.substr(0, 8));
          elemNodes_solid[1] = std::stoi(line.substr(8, 8));
          elemNodes_solid[2] = std::stoi(line.substr(16, 8));
          elemNodes_solid[3] = std::stoi(line.substr(24, 8));
          elemNodes_solid[4] = std::stoi(line.substr(32, 8));
          elemNodes_solid[5] = std::stoi(line.substr(40, 8));
          elemNodes_solid[6] = std::stoi(line.substr(48, 8));
          elemNodes_solid[7] = std::stoi(line.substr(56, 8));
          db_elements->add_element_byKeyFile(
            Element::SOLID, id, partID, elemNodes_solid);
          iCardLine = 0;
        }
      } catch (const std::exception& ex) {
        std::cerr << "Error reading solid in line " << (iLine + 1) << ":"
                  << ex.what() << std::endl;
        keyword = KeywordType::NONE;
      } catch (...) {
        std::cerr << "Error reading solid in line " << (iLine + 1)
                  << ": Unknown error." << std::endl;
        keyword = KeywordType::NONE;
      }
    } else if ((keyword == KeywordType::ELEMENT_SOLID) &&
               (line_has_keyword | line.empty())) {
      keyword = KeywordType::NONE;
#ifdef QD_DEBUG
      std::cout << "*ELEMENT_SOLID finished in line: " << (iLine + 1)
                << std::endl;
#endif
    }

    /* ELEMENTS BEAM */
    if (line_trimmed.substr(0, std::string("*ELEMENT_BEAM").size()) ==
        "*ELEMENT_BEAM") {
      keyword = KeywordType::ELEMENT_BEAM;
      iCardLine = 0;
#ifdef QD_DEBUG
      std::cout << "Starting *ELEMENT_BEAM in line: " << (iLine + 1)
                << std::endl;
#endif
    } else if ((keyword == KeywordType::ELEMENT_BEAM) && !line_has_keyword &&
               (!line.empty())) {
      try {
        if (iCardLine == 0) {
          id = std::stoi(line.substr(0, 8));
          partID = std::stoi(line.substr(8, 8));
          elemNodes_beam[0] = std::stoi(line.substr(16, 8));
          elemNodes_beam[1] = std::stoi(line.substr(24, 8));
          db_elements->add_element_byKeyFile(
            Element::BEAM, id, partID, elemNodes_beam);
          ++iCardLine;
        } else if (iCardLine == 1) {
          iCardLine = 0;
        }
      } catch (const std::exception& ex) {
        std::cerr << "Error reading beam in line " << (iLine + 1) << ":"
                  << ex.what() << std::endl;
        keyword = KeywordType::ELEMENT_BEAM;
      } catch (...) {
        std::cerr << "Error reading beam in line " << (iLine + 1)
                  << ": Unknown error." << std::endl;
        keyword = KeywordType::ELEMENT_BEAM;
      }
    } else if ((keyword == KeywordType::ELEMENT_BEAM) &&
               (line_has_keyword | line.empty())) {
      keyword = KeywordType::ELEMENT_BEAM;
#ifdef QD_DEBUG
      std::cout << "*ELEMENT_BEAM finished in line: " << (iLine + 1)
                << std::endl;
#endif
    }

    /* PART */
    if (line_trimmed.substr(0, 5) == "*PART") {
      keyword = KeywordType::PART;
#ifdef QD_DEBUG
      std::cout << "Starting *PART in line: " << (iLine + 1) << std::endl;
#endif
      iCardLine = 0;
    } else if ((keyword == KeywordType::PART) && !line_has_keyword &&
               (!line.empty())) {

      if (iCardLine == 0) {
        title = line_trimmed;
        ++iCardLine;
      } else if (iCardLine == 1) {

        try {
          id = std::stoi(line.substr(0, 10));

          try {
            db_parts->get_partByID(id)->set_name(title);
          } catch (std::invalid_argument&) {
            db_parts->add_partByID(id, title);
          }

          ++iCardLine;
        } catch (const std::exception& ex) {
          std::cerr << "Error reading part in line " << (iLine + 1) << ":"
                    << ex.what() << std::endl;
          keyword = KeywordType::NONE;
        }
      } else if ((keyword == KeywordType::PART) &&
                 (line_has_keyword | (iCardLine > 1))) {
        keyword = KeywordType::NONE;
#ifdef QD_DEBUG
        std::cout << "*PART finished in line: " << (iLine + 1) << std::endl;
#endif
      }
    }

  } // for lines
#ifdef QD_DEBUG
  std::cout << "parsing of file " << _filepath << " done." << std::endl;
#endif
}

/** Convert the keyfile to a string
 *
 * @return str : keyfile as string
 */
std::string
KeyFile::str() const
{

  std::map<int64_t, std::shared_ptr<Keyword>> kwrds_sorted;
  for (auto& kv : keywords) {
    for (auto kw : kv.second) {
      kwrds_sorted.insert(std::make_pair(kw->get_line_index(), kw));
    }
  }

  std::stringstream ss;
  for (auto kv : kwrds_sorted)
    ss << kv.second->str();

  return std::move(ss.str());
}

/** Save a keyfile again
 *
 * @param _filepath : path to new file
 * @param _save_includes : save also include files in same dir
 * @param _save_all_in_one : save all include data in the master file
 */
void
KeyFile::save_txt(const std::string& _filepath,
                  bool _save_includes,
                  bool _save_all_in_one)
{
  save_file(_filepath, str());
}

} // namespace qd