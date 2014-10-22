/** @file OptionParser.h
 *
 *  @brief Interface declaration of Option::Parser class.
 *
 *  @author Uladzimir Sidarenka <sidarenk@uni-potsdam.de>
 */

#ifndef SOCMEDIA_UTILS_OPTIONPARSER_H_
# define SOCMEDIA_UTILS_OPTIONPARSER_H_

///////////////
// Libraries //
///////////////
#include "Option.h"		/* Option::Option() etc.*/

# include <map>			/* for std::map */
# include <memory>		/* for std::shared_ptr */
# include <string>		/* for std::string */

////////////
// Macros //
////////////

/////////////////
// Declaration //
/////////////////

/**
 * @brief Namespace containing auxiliary classes for handling options.
 */
namespace Option {
  /**
   * @brief Class containing functionality for parsing options
   */
  class Parser
  {
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

    /** Map from short option name to option pointer. */
    char2opt_t m_short2opt;

    /** Map from long option name to option pointer. */
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
    Parser(const char *a_desc = nullptr);

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
     * @return copy of C++ string holding option description
     */
    std::string generate_opt_description() const;
  };
}
#endif /* SOCMEDIA_UTILS_OPTIONPARSER_H_ */
