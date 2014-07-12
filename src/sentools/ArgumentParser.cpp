/** @file Options.cpp

    @brief File implementing class for handling options.

    This file implements a class for handling options.
*/

///////////////
// Libraries //
///////////////
#include "ArgumentParser.h"

#include <iostream>		// std::cerr

#include <error.h>		// error()
#include <stdlib.h>		// malloc()

///////////////
// Variables //
///////////////
extern char *optarg;
extern int optind;

static option_t help_opt {'h', "help", "show this screen and exit", ArgReqType::NOARG, NULL};

///////////////////
// Class Methods //
///////////////////


/**
 * Class constructor.
 *
 * @param a_desc - description of the program
 */
ArgumentParser::ArgumentParser(const char *a_desc = NULL):
  m_char2opt{}, m_no_short_cnt{0}, m_opt_arg_cnt{0}, m_req_arg_cnt{0}, \
  m_options{NULL}, m_optstring{NULL}, m_usage{}
{
  storeOption(&help_opt);
}

/**
 * Delete dynamically created options and array of long option structs.
 */
ArgumentParser::~ArgumentParser() {
  // clear options referenced by map
  char2opt_t::iterator it, end = m_char2opt.end();
  for (it = m_char2opt.begin(); it != end; ++it)
    delete it->second;

  // clear C-style array of option structs
  delete[] m_options;
}

void ArgumentParser::addOption(const char a_short, const char *a_long,	\
			       const char *a_desc = NULL,		\
			       const ArgReqType a_arg_type,		\
			       const char *a_default) {
  // new option cannot be added, once the C array of option structs was
  // populated
  if (m_options)
    error(2, 0, "Option '%s' is added after all command line options were parsed.", a_long);

  // create an option_t from passed arguments
  option_t *opt = (option_t *) malloc(sizeof(option_t));
  if (! opt)
    error(3, errno, "Could not allocate memory for option '%s'.\n", a_long);

  // populate option's members
  if (! a_short)
    // assign a uniq character as option's short name
    opt->m_short = --m_no_short_cnt;
  else if (a_short < 0)
    error(4, errno, "Invalid short option character '%c' (%d) (char code should be >0).\n", \
	  a_short, a_short);

  opt->m_long = a_long;
  opt->m_desc = a_desc;
  opt->m_arg_type = a_arg_type;
  opt->m_arg = a_default ? a_default : NULL;

  if (opt->m_arg_type == ArgReqType::OPTIONAL)
    ++m_opt_arg_cnt;
  else if (opt->m_arg_type == ArgReqType::REQUIRED)
    ++m_req_arg_cnt;

  // remember that option under its short name in `m_char2opt`
  storeOption(opt);
}

int ArgumentParser::parseOptions(int& argc, char *const argv[]) {
  // construct a struct option array for storing options if not already
  // constructed
  if (! m_options)
    m_options = createOptions();

  int c, option_index;
  // parse obtained options with `getopt_long`
  while ((c = getopt_long(argc, argv, m_optstring, m_options, &option_index)) != -1) {
    switch(c) {
    case '?':
      std::cerr << "Unrecognized option '" << c << "'." << std::endl;
      goto error_exit;

    default:
      break;
    };
  }
  return 0;

 error_exit:
  return -1;
}

struct option *ArgumentParser::createOptions() {
  struct option *options = (struct option *) calloc(m_char2opt.size() + 1, \
						    sizeof(struct option));
  if (! options)
    error(4, errno, "Could not allocate memory for storing option array.\n");

  m_optstring = (char *) malloc((m_char2opt.size() + 2 * m_opt_arg_cnt + \
				 m_req_arg_cnt + 1) * sizeof(char));

  int i = 0, j = 0;
  char2opt_t::iterator it = m_char2opt.begin(), end = m_char2opt.end();
  option_t *opt;
  for (; it != end; ++it, ++i, ++j) {
    opt = it->second;
    options[i] = {opt->m_long, (int) opt->m_arg_type, NULL, (int) opt->m_short};
    // put character in constructed optstring
    m_optstring[j] = opt->m_short;
    if (opt->m_arg_type == ArgReqType::OPTIONAL) {
      m_optstring[++j]=':';
      m_optstring[++j]=':';
    } else if (opt->m_arg_type == ArgReqType::REQUIRED) {
      m_optstring[++j]=':';
    }
  }
  // put sentinel as the last element of the array
  options[i] = {0, 0, NULL, 0};
  m_optstring = '\0';
  return options;
}

/**
 * Store correspondence of option's character to its index in vector \c m_options.
 *
 * @param a_opt - option to be inserted
 */
void ArgumentParser::storeOption(option_t *a_opt) {
  static std::pair<char2opt_t::iterator, bool> ret;
  signed char shortName = a_opt->m_short;

  ret = m_char2opt.insert(char_opt_t(shortName, a_opt));
  if (! ret.second)
    error(1, 0, "Duplicate definition for option '%c' ('%s').", a_opt->m_short, \
	  a_opt->m_long);
}
