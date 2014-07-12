#ifndef _ARGUMENT_PARSER_H_
# define _ARGUMENT_PARSER_H_ 1
///////////////
// Libraries //
///////////////
#include <map>		/* std::map */
#include <string>	/* std::string */

#include <string.h>		/* strcmp */
#include <getopt.h>		/* no_argument, optional_argument, etc. */

///////////////////////
// Enums and Structs //
///////////////////////

/**
 * Type of argument required by option.
 */
enum class ArgReqType: int {
  NOARG = no_argument,		///< no argument is required
    OPTIONAL = optional_argument, ///< argument is optional
    REQUIRED = required_argument,	///< argument is obligatory
    };

/**
 * Container for storing option's attributes.
 */
typedef struct {
  char m_short;			///< short option form
  const char *m_long;		///< long option form
  const char *m_desc;		///< option's description

  ArgReqType m_arg_type;	///< type of argument
  const char *m_arg;		///< pointer to option's argument
} option_t;

///////////
// Class //
///////////

/**
 * @brief class for adding and parsing command line options
 *
 * This class provides methods for adding command line arguments to programs
 * and parsing those arguments on command line.
 */
class ArgumentParser {
private:
  /// pair of character and option pointer
  typedef std::pair<const signed char, option_t *> char_opt_t;
  /// mapping of character corresponding to option to option's description
  typedef std::map<const signed char, option_t *> char2opt_t;

  /// map from option's short name to the address of that option
  char2opt_t m_char2opt;
  /// counter of options with no short names
  signed char m_no_short_cnt;
  /// counter of options with optional argument
  unsigned m_opt_arg_cnt;
  /// counter of options with required arguments
  unsigned m_req_arg_cnt;
  /// C-style array of option structs (@see unistd.h)
  struct option *m_options;
  /// string representing short options
  char *m_optstring;

  /// usage message for given program
  std::string m_usage;

public:
  /// class constructor
  ArgumentParser(const char *a_desc);

  /// class destructor
  ~ArgumentParser();

  /// add option description to parser
  void addOption(const char a_short, const char *a_long, const char *a_desc, \
		 const ArgReqType a_arg_type = ArgReqType::NOARG, \
		 const char *a_default = NULL);

  /// process command line arguments
  int parseOptions(int& argc, char *const argv[]);

  /// return \c true if option is set and \c false otherwise
  bool getBool(const char *a_name) const;

  /// return integer value of the argument associated with the option
  int getInt(const char *a_name) const;

  /// return pointer to string argument associated with the option
  const char *getCStr(const char *a_name) const;

private:
  /// store option's character and option's address in \c m_char2opt
  void storeOption(option_t *a_opt);

  /// construct array of option struct's from options stroed in \c m_char2opt
  struct option *createOptions();
};
#endif
