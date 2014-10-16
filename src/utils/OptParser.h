/** @file OptParser.h
 *
 *  @brief Interface declaration of OptParser class.
 *
 *  @author Uladzimir Sidarenka <sidarenk@uni-potsdam.de>
 */

#ifndef SOCMEDIA_UTILS_OPTPARSER_H_
# define SOCMEDIA_UTILS_OPTPARSER_H_

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
 public:
  ////////////////
  // Data Types //
  ////////////////
  /**
   * Enum class describing possible types of option arguments.
   */
  enum class ArgType: char {
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

 private:
  /** Abstract class with common option interface. */
  struct OptionBase {
    /* Data members */
    /** Short name of the option */
    const char m_short_name;
    /** Long name of the option */
    const char *m_long_name;
    /** Option description */
    const char *m_desc;

    /** Option's type */
    arg_type_t m_type;
    /** Option's check status */
    bool m_specified;

    /* Methods */
    /** default constructor */
    OptionBase();

    /** constructor */
    OptionBase(const char a_short, const char *a_long, const char *a_desc, \
	   arg_type_t a_type = ArgType::NONE);

    /** set option's value from string argument */
    virtual void parse_arg(const char *a_value) = 0;

    /** return pointer to option's argument */
    virtual const void* get_arg() const = 0;

    /** return string representation of options' argument type */
    virtual const char* atype2str() const = 0;
  };

  /** Struct describing single option. */
  template<typename T>
    struct Option final: public OptionBase {

    /* Data members */
    /** Option's value */
    T m_value;

    /* Methods */
    /** option's constructor */
    explicit Option(const char a_short, const char *a_long, const char *a_desc, \
		    arg_type_t a_type = ArgType::NONE, const void *a_default = nullptr);

    /** obtain option's value */
    T get_value() const;

    /** set option's value directly from corresponding value */
    void set_value(T a_value);

    /** set option's value from string argument */
    void parse_arg(const char *a_value);

    /** return pointer to option's argument */
    const void* get_arg()
      const
    {
      return static_cast<const void*>(&m_value);
    }

    /** return string representation of options' argument type */
    const char* atype2str() const;
  };

  /**
   * Type for storing shared pointers to options.
   */
  typedef std::shared_ptr<OptionBase> opt_shptr_t;

  /**
   * Type for storing mapping from short option name to option pointer.
   */
  typedef std::map<const char, opt_shptr_t> char2opt_t;

  /**
   * Pair representing key and value type of char2opt_t map.
   */
  typedef std::pair<char, opt_shptr_t> char_opt_t;

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

 public:

  //////////////////
  // Data Members //
  //////////////////

  /** Program description */
  const char *m_desc;
  /** Program description */
  const char *m_name;
  /** Program usage */
  mutable std::string m_usage;

  /////////////
  // Methods //
  /////////////

  /**
   * Default constructor.
   *
   * @param a_desc - description of the program
   */
  OptParser(const char *a_desc = nullptr);

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
		  arg_type_t a_type = ArgType::NONE, const void *a_default = nullptr);

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
   * Output usage information and exit.
   *
   * @param a_ostream - output stream for usage
   *
   * @return \c void
   */
  void usage(std::ostream& a_ostream) const;

  /**
   * Obtain value of option's argument.
   *
   * @param a_short - short name of the option
   *
   * @return nullptr if option is unknown and pointer to option's value otherwise
   */
  const void *get_arg(const char a_short) const;

  /**
   * Obtain value of option's argument.
   *
   * @param a_long - long name of the option
   *
   * @return nullptr if option is unknown and pointer to option's value otherwise
   */
  const void *get_arg(const char *a_long) const;

  /**
   * Output help on options and exit.
   */
  void usage();

  /**
   * Check if option was specified on command line.
   *
   * @param a_short - short name of the option
   *
   * @return \c true if option was specified, \c false otherwise
   */
  inline bool check(const char a_short)
    const
  {
    char2opt_t::const_iterator cit = m_short2opt.find(a_short);

    if (cit == m_short2opt.end())
      return false;
    else
      return cit->second->m_specified;
  }

  /**
   * Check if option was specified on command line.
   *
   * @param a_long - long name of the option
   *
   * @return \c true if option was specified, \c false otherwise
   */
  inline bool check(const char *a_long)
    const
  {
    str2opt_t::const_iterator cit = m_long2opt.find(std::string(a_long));

    if (cit == m_long2opt.end())
      return false;
    else
      return cit->second->m_specified;
  }

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

  /**
   * Generate description of an option.
   *
   * @return C++ string holding option description
   */
  std::string generate_opt_description() const;
};
#endif /* SOCMEDIA_UTILS_OPTPARSER_H_ */
