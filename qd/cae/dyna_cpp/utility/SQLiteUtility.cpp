

#include <stdexcept>
#include <type_traits>

#include <dyna_cpp/utility/SQLiteUtility.hpp>

namespace qd {
namespace sqlite {

/** Open a database from a filepath
 *
 * @param filepath : path to database
 * @return db : database handle
 */
sqlite3*
open_database(const std::string& filepath)
{

  sqlite3* db = nullptr;
  auto return_code = sqlite3_open(filepath.c_str(), &db);

  if (return_code) {
    sqlite3_close(db);
    throw(std::invalid_argument("Can not open database " + filepath +
                                std::string("\nSQLite Error: ") +
                                std::string(sqlite3_errmsg(db))));
  }

  return db;
}

/** Create a statement for table allocation from a list of variable names with
 * types
 *
 * @param _table_name : name of table in sqlite db
 * @param _sql_vars : list of variables
 *
 * Variables must consist of: "name type", e.g. "id INT"
 */
std::string
get_table_creation_string(const std::string& _table_name,
                          const std::vector<std::string> _sql_vars)
{

  std::string sql_statement_str = "CREATE TABLE " + _table_name + "(";
  for (size_t iVar = 0; iVar < _sql_vars.size(); ++iVar) {
    sql_statement_str += _sql_vars[iVar];
    if (iVar < _sql_vars.size() - 1)
      sql_statement_str += ",";
  }
  sql_statement_str += ");";

  return sql_statement_str;
}

/** Create a sqlite table
 *
 * @param db : sqlite database handle
 * @param _table_name : name of table in the file
 * @param _sql_vars : vars in the table with name and type
 *
 * Variables must consist of: "name type", e.g. "id INT"
 */
void
create_table(sqlite3* db,
             const std::string& _table_name,
             const std::vector<std::string> _sql_vars)
{

  // checks
  if (db == nullptr)
    throw(std::invalid_argument("Can not create table: " + _table_name +
                                ": database pointer null."));
  if (_table_name.size() == 0)
    throw(std::invalid_argument("Can not create table with no name."));

  // assemble creation string
  std::string sql_statement = get_table_creation_string(_table_name, _sql_vars);

  // allocate
  char* error_message = nullptr;
  auto return_code =
    sqlite3_exec(db, sql_statement.c_str(), nullptr, nullptr, &error_message);
  if (return_code) {
    sqlite3_close(db);
    throw(std::runtime_error(std::string("Could not create table :") +
                             _table_name + std::string("\nSQLite Error: ") +
                             std::string(error_message)));
  }
}

/** Assemble the string used to tell sqlite that we want to make an insertion
 *
 * @param _table_name : name of the table
 * @param _sql_vars : sqlite variable name list (e.g. "id")
 */
std::string
prepare_insertion_string(const std::string& _table_name,
                         const std::vector<std::string> _sql_vars)
{

  std::string sql_statement_str = "INSERT INTO " + _table_name + " VALUES (";
  for (size_t iVar = 0; iVar < _sql_vars.size(); ++iVar) {

    sql_statement_str += _sql_vars[iVar];
    if (iVar < _sql_vars.size() - 1)
      sql_statement_str += ",";
  }
  sql_statement_str += ");";

  return sql_statement_str;
}

/** Prepare insetion of multiple vars
 *
 * @param db : sqlite database handle
 * @param _table_name : name of the table
 * @param _sql_vars : sqlite variable name list (e.g. "id")
 */
sqlite3_stmt*
prepare_insertion(sqlite3* db,
                  const std::string& _table_name,
                  const std::vector<std::string> _sql_vars)
{
  // checks
  if (db == nullptr)
    throw(std::invalid_argument("Can not create table: " + _table_name +
                                ": database pointer null."));
  if (_table_name.size() == 0)
    throw(std::invalid_argument("Can not create table with no name."));

  // vars
  char* error_message = nullptr;
  sqlite3_stmt* sql_statement = nullptr;

  // build string from var list
  std::string sql_statement_str =
    prepare_insertion_string(_table_name, _sql_vars);

  // do the preparation
  sqlite3_prepare_v2(db,
                     sql_statement_str.c_str(),
                     static_cast<int32_t>(sql_statement_str.size()),
                     &sql_statement,
                     nullptr);

  return sql_statement;
}

/** Close the sqlite database
 *
 * @param db : sqlite database handle
 * @param sql_statement : sqlite insertion statement handle
 *
 * the statement handle is optional
 */
void
close_database(sqlite3* db, sqlite3_stmt* sql_statement)
{
  sqlite3_finalize(sql_statement);
  sqlite3_close(db);
}

/** Begin the insertion transaction
 *
 * @param db : sqlite database handle
 */
void
begin_insertion(sqlite3* db)
{
  char* error_message = nullptr;

  auto return_code =
    sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, &error_message);
  if (return_code) {
    sqlite3_close(db);
    throw(std::runtime_error("Error while starting SQL transaction." +
                             std::string("\nSQLite Error: ") +
                             std::string(error_message)));
  }
}

/** Perform an sqlite step (write data into db)
 *
 * @param db : sqlite database handle
 * @param sql_statement : sqlite insertion statement handle
 */
void
perform_step(sqlite3* db, sqlite3_stmt* sql_statement)
{

  if (sqlite3_step(sql_statement) != SQLITE_DONE) {
    close_database(db, sql_statement);
    throw(std::runtime_error("SQLite stepping failed."));
  }
}

/** End the insertion transaction
 *
 * @param db : sqlite database handle
 */
void
end_insertion(sqlite3* db, sqlite3_stmt* sql_statement)
{
  char* error_message = nullptr;

  auto return_code =
    sqlite3_exec(db, "END TRANSACTION", nullptr, nullptr, &error_message);
  if (return_code) {
    close_database(db, sql_statement);
    throw(
      std::runtime_error(std::string("Error while ending SQL transaction.") +
                         "\nSQLite Error: " + std::string(error_message)));
  }
}

} // namespace sqlite
} // namespace qd
