/** @file OptParser.h
 *
 *  @brief Interface declaration of OptParser class.
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
 * @brief Class containing functionality for parsing options
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
      INT,
      /** enum value representing float type */
      FLOAT,
      /** enum value representing double type */
      DOUBLE,
      /** enum value representing long type */
      LONG,
      /** enum value representing long long type */
      LLONG,
      };

  /** Synonym for ArgType. */
  typedef ArgType arg_type_t;

  /** Union for holding values of option arguments. */
  typedef union {
    /** member holding pointer to char */
    const char *m_char_ptr;
    /** member holding int value */
    int  m_int;
    /** member holding float value */
    float  m_float;
    /** member holding double value */
    double  m_double;
    /** member holding long value */
    long m_long;
    /** member holding long value */
    long long m_llong;
  } arg_value_t;

  /** Struct describing single option. */
  struct Option {
    /* Data members */

    /** Short name of the option */
    const char m_short_name;
    /** Long name of the option */
    const char *m_long_name;
    /** Option's type */
    arg_type_t m_type;
    /** Option's value */
    arg_value_t m_value;
    /** Option's check status */
    bool m_specified;

    /* Methods */

    /** option's constructor */
    Option(const char a_short, const char *a_long, arg_type_t a_type, const void *a_default);

    /** obtain option's value */
    const void *get_value() const;

    /** set option's value from string */
    void set_value(const char *a_value);

    /** set option's value directly from corresponding value */
    void set_value(const void *a_value);
  };

  /**
   * Type for storing shared pouinters to options.
   */
  typedef std::shared_ptr<Option> opt_shptr_t;
  /**
   * Type for storing mapping from short option name to option pointer.
   */
  typedef std::map<const char, opt_shptr_t> char2opt_t;

  /**
   * Pair representing key and value type of char2opt_t map.
   */
  typedef std::pair<std::string, opt_shptr_t> char_opt_t;

  /**
   * Type for storing mapping from long option name to option pointer.
   */
  typedef std::map<std::string, opt_shptr_t> str2opt_t;

  /**
   * Pair representing key and value type of str2opt_t map.
   */
  typedef std::pair<std::string, opt_shptr_t> str_opt_t;

  //////////////////
  // Data Members //
  //////////////////

  /**
   * Map from short option name to option pointer.
   */
  char2opt_t m_short2opt;

  /**
   * Map from long option name to option pointer.
   */
  str2opt_t m_long2opt;

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
   * @return \c int - index of the next unprocessed options (-1 if an error occurred)
   */
  int parse(const int a_argc, char *a_argv[]);

  /**
   * Function for adding option to parser.
   *
   * @param a_short - short name of the option
   * @param a_long - long name of the option
   * @param a_desc - option's description
   * @param a_type - type of option's argument
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
   * @param a_long - long name of the option
   * @param a_trg - target variable in which option's value should be stored
   *
   * @return \c -1 if neither option nor default value for its
   * argument were specified
   */
  int get_argument(const char *a_long, void *a_trg) const;

 private:
  /**
   * Parse long option.
   *
   * @param a_argc - number of arguments
   * @param a_argv - array of pointers to arguments
   *
   * @return index of the next unprocessed options (-1 if an error occurred)
   * @throws std::invalid_argument if option is not recognized or no argument is supplied
   */
  int parse_long(const char *a_opt_start, const int a_argc, char *a_argv[], int &a_cnt);

  /**
   * Parse short option.
   *
   * @param a_argc - number of arguments
   * @param a_argv - array of pointers to arguments
   *
   * @return index of the next unprocessed options (-1 if an error occurred)
   * @throws std::invalid_argument if option is not recognized or no argument is supplied
   */
  int parse_short(const char *a_opt_start, const int a_argc, char *a_argv[], int &a_cnt);

  /**
   * Obtain option's argument.
   *
   * @param a_opt - option for which the value should be stored
   * @param a_arg_start - pointer to the start of option's argument on CL
   *
   * @return \c void
   */
  void get_arg_value(opt_shptr_t a_opt, const char *a_arg_start);
};

#endif /*__OPTPRASER_H__ */
