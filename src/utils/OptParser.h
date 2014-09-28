/** @file OptParser.h
 *
 *  @brief Interface of OptParser class.
 *
 *  @author Uladzimir Sidarenka <sidarenk@uni-potsdam.de>
 */

#ifndef __OPTPRASER_H__
# define __OPTPRASER_H__

///////////////
// Libraries //
///////////////
# include <map>		/* for std::map */
# include <memory>	/* for std::shared_ptr */
# include <string>	/* for std::string */

////////////
// Macros //
////////////

/////////////////
// Declaration //
/////////////////

/**
 * Class containing functionality for parsing options.
 */
class OptParser
{
 private:
  ////////////////
  // Data Types //
  ////////////////
  /**
   * Enum class describing possible types of option arguments.
   */
  enum class ArgType {
    /** enum value representing none value */
    NONE,
      /** enum value representing pointer to char */
      CHAR_PTR,
      /** enum value representing int type */
      SHORT,
      /** enum value representing int type */
      INT,
      /** enum value representing float type */
      FLOAT,
      /** enum value representing double type */
      DOUBLE,
      /** enum value representing long type */
      LONG,
      };

  /** Synonym for ArgType. */
  typedef ArgType arg_type_t;

  /** Union for holding values of option arguments. */
  typedef union {
    /** member holding pointer to char */
    const char *m_char_ptr;
    /** member holding short int */
    short int  m_short;
    /** member holding int value */
    int  m_int;
    /** member holding float value */
    float  m_float;
    /** member holding double value */
    double  m_double;
    /** member holding long value */
    long m_long;
  } arg_value_t;

  /** Struct describing single option. */
  struct Option {
    /** Long name of the option */
    const char *m_long_name;
    /** Short name of the option */
    const char m_short_name;
    /** Option's type */
    arg_type_t m_type;
    /** Option's value */
    arg_value_t m_value;
  };

  //////////////////
  // Data Members //
  //////////////////

  /**
   * Map from short option name to option pointer.
   */
  std::map<const char, std::shared_ptr<const Option *>> m_short2opt;

  /**
   * Map from long option name to option pointer.
   */
  std::map<const char *, std::shared_ptr<const Option *>> m_long2opt;

  /////////////
  // Methods //
  /////////////
  /**
   * Output help on options and exit.
   */
  void usage();

  /**
   * Process long option.
   *
   * @param a_arg - argument to be processed
   */
  void process_long(const char *a_arg);

 public:

  //////////////////
  // Data Members //
  //////////////////

  /** Program description */
  std::string m_desc;
  /** Program description */
  std::string m_name;
  /** Number of processed options */
  int m_parsed;

  /////////////
  // Methods //
  /////////////

  /**
   * Default constructor.
   *
   * @param a_desc - description of the program
   */
  OptParser(const char *a_desc = "");

  /**
   * Parse operation.
   *
   * @param a_argc - number of arguments
   * @param a_argv - array of pointers to arguments
   *
   * @return index of the next unprocessed options (-1 if an error occurred)
   */
  void parse(const int a_argc, char *a_argv[]);

  /**
   * Function for adding option.
   *
   * @param a_short - short name of the option
   * @param a_long - long name of the option
   * @param a_desc - option's description
   * @param a_default - default value for option's argument
   */
  void add_option(const char a_short, const char *a_long, const char *a_desc, \
		  arg_type_t a_type = ArgType::NONE, void *a_default = nullptr);

  /**
   * Obtain value of option's argument.
   *
   * @param a_short - short name of the option
   * @param a_trg - target variable in which option's value should be stored
   *
   * @return \c -1 if neither option nor default value for its
   * argument were specified
   */
  int get_argument(const char a_short, void *a_trg) const;

  /**
   * Obtain value of option's argument.
   *
   * @param a_short - long name of the option
   * @param a_trg - target variable in which option's value should be stored
   *
   * @return \c -1 if neither option nor default value for its
   * argument were specified
   */
  int get_argument(const char *a_long, void *a_trg) const;

 private:
  /**
   * Parse long optionn.
   *
   * @param a_argc - number of arguments
   * @param a_argv - array of pointers to arguments
   *
   * @return index of the next unprocessed options (-1 if an error occurred)
   */
  int parse_long(const char *a_opt, const int a_argc, char *a_argv[], int &a_cnt);
};

#endif /*__OPTPRASER_H__ */
