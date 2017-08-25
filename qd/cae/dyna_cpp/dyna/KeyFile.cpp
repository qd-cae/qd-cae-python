
#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/db/Part.hpp>
#include <dyna_cpp/dyna/KeyFile.hpp>
#include <dyna_cpp/utility/BoostException.hpp>
#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace qd {

// Enumeration
namespace Keyword {
enum Keyword
{
  NONE,
  NODE,
  ELEMENT_BEAM,
  ELEMENT_SHELL,
  ELEMENT_SOLID,
  PART,
  INCLUDE
};
}

/**
 * Constructor for a LS-Dyna input file.
 */
KeyFile::KeyFile()
{
}

/** Constructor for reading a LS-Dyna input file.
 *
 * @param string filepath : filepath of a key file to read
 */
KeyFile::KeyFile(std::string _filepath)
  : FEMFile(_filepath)
{
  // Read the mesh
  this->read_mesh(this->get_filepath());
}

/** Read the mesh from the file given in the filepath
 *
 * @param string filepath : filepath of a key file to read
 */
void
KeyFile::read_mesh(std::string _filepath)
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
  std::vector<std::string> lines = read_textFile(_filepath);
#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  // Get databases
  auto db_parts = this->get_db_parts();
  auto db_nodes = this->get_db_nodes();
  auto db_elements = this->get_db_elements();

  // Time to do the thing
  Keyword::Keyword keyword = Keyword::NONE;
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
      keyword = Keyword::INCLUDE;
#ifdef QD_DEBUG
      std::cout << "*INCLUDE in line: " << (iLine + 1) << std::endl;
#endif
    } else if (keyword == Keyword::INCLUDE) {
      this->read_mesh(directory +
                      line_trimmed); // basic directory is this file's
      keyword = Keyword::NONE;
    }

    /* NODES */
    if (line_trimmed == "*NODE") {
      keyword = Keyword::NODE;
#ifdef QD_DEBUG
      std::cout << "Starting *NODE in line: " << (iLine + 1) << std::endl;
#endif

    } else if ((keyword == Keyword::NODE) && !line_has_keyword &&
               (!line_trimmed.empty())) {
      try {
        coords[0] = std::stof(line.substr(8, 16));
        coords[1] = std::stof(line.substr(24, 16));
        coords[2] = std::stof(line.substr(40, 16));

        db_nodes->add_node(std::stoi(line.substr(0, 8)), coords);

      } catch (const std::exception& ex) {
        std::cerr << "Error reading node in line " << (iLine + 1) << ": "
                  << ex.what() << std::endl;
        keyword = Keyword::NODE;

      } catch (...) {
        std::cerr << "Error reading node in line " << (iLine + 1)
                  << ": Unknown error." << std::endl;
        keyword = Keyword::NODE;
      }

    } else if ((keyword == Keyword::NODE) &&
               (line_has_keyword | line.empty())) {
      keyword = Keyword::NONE;
#ifdef QD_DEBUG
      std::cout << "*NODE finished in line: " << (iLine + 1) << std::endl;
#endif
    }

    /* ELEMENTS SHELL */
    if (line_trimmed == "*ELEMENT_SHELL") {
      keyword = Keyword::ELEMENT_SHELL;
#ifdef QD_DEBUG
      std::cout << "Starting *ELEMENT_SHELL in line: " << (iLine + 1)
                << std::endl;
#endif
    } else if ((keyword == Keyword::ELEMENT_SHELL) && !line_has_keyword &&
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
        keyword = Keyword::NONE;
      } catch (...) {
        std::cerr << "Error reading shell in line " << (iLine + 1)
                  << ": Unknown error." << std::endl;
        keyword = Keyword::NONE;
      }

    } else if ((keyword == Keyword::ELEMENT_SHELL) &&
               (line_has_keyword | line.empty())) {
      keyword = Keyword::NONE;
#ifdef QD_DEBUG
      std::cout << "*ELEMENT_SHELL finished in line: " << (iLine + 1)
                << std::endl;
#endif
    }

    /* ELEMENTS SOLID */
    if (line_trimmed == "*ELEMENT_SOLID") {
      keyword = Keyword::ELEMENT_SOLID;
      iCardLine = 0;
#ifdef QD_DEBUG
      std::cout << "Starting *ELEMENT_SOLID in line: " << (iLine + 1)
                << std::endl;
#endif
    } else if ((keyword == Keyword::ELEMENT_SOLID) && !line_has_keyword &&
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
        keyword = Keyword::NONE;
      } catch (...) {
        std::cerr << "Error reading solid in line " << (iLine + 1)
                  << ": Unknown error." << std::endl;
        keyword = Keyword::NONE;
      }

    } else if ((keyword == Keyword::ELEMENT_SOLID) &&
               (line_has_keyword | line.empty())) {
      keyword = Keyword::NONE;
#ifdef QD_DEBUG
      std::cout << "*ELEMENT_SOLID finished in line: " << (iLine + 1)
                << std::endl;
#endif
    }

    /* ELEMENTS BEAM */
    if (line_trimmed.substr(0, std::string("*ELEMENT_BEAM").size()) ==
        "*ELEMENT_BEAM") {
      keyword = Keyword::ELEMENT_BEAM;
      iCardLine = 0;
#ifdef QD_DEBUG
      std::cout << "Starting *ELEMENT_BEAM in line: " << (iLine + 1)
                << std::endl;
#endif
    } else if ((keyword == Keyword::ELEMENT_BEAM) && !line_has_keyword &&
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
        keyword = Keyword::ELEMENT_BEAM;
      } catch (...) {
        std::cerr << "Error reading beam in line " << (iLine + 1)
                  << ": Unknown error." << std::endl;
        keyword = Keyword::ELEMENT_BEAM;
      }

    } else if ((keyword == Keyword::ELEMENT_BEAM) &&
               (line_has_keyword | line.empty())) {
      keyword = Keyword::ELEMENT_BEAM;
#ifdef QD_DEBUG
      std::cout << "*ELEMENT_BEAM finished in line: " << (iLine + 1)
                << std::endl;
#endif
    }

    /* PART */
    if (line_trimmed.substr(0, 5) == "*PART") {
      keyword = Keyword::PART;
#ifdef QD_DEBUG
      std::cout << "Starting *PART in line: " << (iLine + 1) << std::endl;
#endif
      iCardLine = 0;

    } else if ((keyword == Keyword::PART) && !line_has_keyword &&
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
          keyword = Keyword::NONE;
        }

      } else if ((keyword == Keyword::PART) &&
                 (line_has_keyword | (iCardLine > 1))) {
        keyword = Keyword::NONE;
#ifdef QD_DEBUG
        std::cout << "*PART finished in line: " << (iLine + 1) << std::endl;
#endif
      }
    }

  } // for lines
#ifdef QD_DEBUG
  std::cout << "parsing of text-file done." << std::endl;
#endif
}

} // namespace qd