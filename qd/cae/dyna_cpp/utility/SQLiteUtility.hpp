
#ifndef SQLITE_UTILITY_HPP
#define SQLITE_UTILITY_HPP

#include <string>
#include <vector>

extern "C" {
#include <dyna_cpp/sqlite/sqlite3.h>
}

namespace qd {
namespace sqlite {

sqlite3*
open_database(const std::string& filepath);

std::string
get_table_creation_string(const std::string& _table_name,
                          const std::vector<std::string> _sql_vars);

void
create_table(sqlite3* db,
             const std::string& _table_name,
             const std::vector<std::string> _sql_vars);

std::string
prepare_insertion_string(const std::string& _table_name,
                         const std::vector<std::string> _sql_vars);

sqlite3_stmt*
prepare_insertion(sqlite3* db,
                  const std::string& _table_name,
                  const std::vector<std::string> _sql_vars);

void
close_database(sqlite3* db, sqlite3_stmt* sql_statement = nullptr);

void
begin_insertion(sqlite3* db);

void
perform_step(sqlite3* db, sqlite3_stmt* sql_statement);

void
end_insertion(sqlite3* db, sqlite3_stmt* sql_statement);

} // namespace sqlite
} // namespace qd

#endif