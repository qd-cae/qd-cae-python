
#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/db/Part.hpp>
#include <dyna_cpp/dyna/KeyFile.hpp>
#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

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
                 double _encryption_detection)
  : FEMFile(_filepath)
  , load_includes(_load_includes)
  , encryption_detection_threshold(_encryption_detection)
{
  // check encryption
  if (encryption_detection_threshold < 0 || encryption_detection_threshold > 1)
    throw(std::invalid_argument(
      "Encryption detection threshold must be between 0 and 1."));

  // Read the mesh
  this->read_mesh(this->get_filepath());
  // this->parse_file(this->get_filepath());
}

void
KeyFile::parse_file(const std::string& _filepath)
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
  std::string last_keyword;
  bool is_keyword = false;
  size_t iLine = 0;
  std::vector<std::string> line_buffer;
  std::vector<std::string> line_buffer_tmp;

  std::stringstream st(std::string(char_buffer.begin(), char_buffer.end()));
  for (std::string line; std::getline(st, line); ++iLine) {

    // some data is there
    // if (!line.empty()) {

    // new keyword
    if (line[0] == '*') {

      // check for previous header comments
      // wrongly assigned to previous block
      // people usually never define comments
      // at the end of a previous block:
      //
      // KEYWORD
      // CARD
      // ------- <- Comment belongs to lower not upper
      // COMMENT
      // KEYWORD
      if (!line_buffer.empty()) {

        // remove comment lines from previous block
        line_buffer_tmp.clear();
        while (line_buffer.size() > 0 && line_buffer.back()[0] == '$') {
          line_buffer_tmp.push_back(line_buffer.back());
          line_buffer.pop_back();
        }

        // create a new keyword from previous data
        auto kw = std::make_shared<Keyword>(line_buffer,
                                            last_keyword,
                                            iLine - line_buffer.size() -
                                              line_buffer_tmp.size());
        if (kw->has_long_fields())
          last_keyword = last_keyword.substr(0, last_keyword.size() - 1);
        keywords[last_keyword].push_back(kw);

        // transfer cropped data
        line_buffer = line_buffer_tmp;
      }

      is_keyword = true;
      trim_right(line);
      last_keyword = line;
    } // IF:line[0] == '*'

    // we stupidly add every line to the buffer
    line_buffer.push_back(line);
    /*
    }
    // line empty (is a block separator)
    else {

      // ADD: KEYWORD BLOCK
      if (is_keyword) {
        auto kw = std::make_shared<Keyword>(
          line_buffer, last_keyword, iLine - line_buffer.size());
        if (kw->has_long_fields())
          last_keyword = last_keyword.substr(0, last_keyword.size() - 1);
        keywords[last_keyword].push_back(kw);
        is_keyword = false;
        line_buffer.clear();
      }
      // ADD: COMMENT BLOCK
      else {
        // (!) create comment/text block
      }
    }
    */
  } // for:line
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

} // namespace qd