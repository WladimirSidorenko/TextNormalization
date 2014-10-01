/** @file OptParser.cpp
 *
 *  @brief Implementation of OptParser class.
 *
 *  @author Uladzimir Sidarenka <sidarenk@uni-potsdam.de>
 */

///////////////
// Libraries //
///////////////
#include "OptParser.h"

#include <stdexcept>		// std::invalid_argument
#include <cstdlib>		// atol(), atof() etc.
#include <cstring>		// memchr()

/////////////////
// Definitions //
/////////////////
OptParser::OptParser(const char *a_desc):
  m_short2opt{}, m_long2opt{}, m_desc{a_desc}, m_name{nullptr}, m_parsed{0}
{
}

int OptParser::parse(const int a_argc, char *a_argv[])
{
  int i = -1;
  char *chp;
  // set program name
  m_name = a_argv[0];
  // iterate over arguments
  for (i = 1; i < a_argc; ++i) {
    switch (*a_argv[i]) {
    case '-':
      chp = a_argv[i] + 1;
      if (*chp) {
	if (*chp == '-') {
	  if (*(++chp)) {
	    i = parse_long(chp, a_argc, a_argv, i);
	  } else {
	    // `--' terminates options
	    return ++i;
	  }
	} else {
	  i = parse_short(chp, a_argc, a_argv, i);
	}
      } else {
	return i;
      }
      break;

    default:
      return i;
    }
  }
  return i;
}

void OptParser::add_option(const char a_short, const char *a_long, \
			   const char *a_desc, arg_type_t a_type, \
			   void *a_default)
{
}

int OptParser::get_argument(const char a_short, void *a_trg)
  const
{
  return 0;
}

int OptParser::get_argument(const char *a_long, void *a_trg)
  const
{
  return 0;
}

int OptParser::parse_long(const char *a_opt_start, const int a_argc, char *a_argv[], int &a_cnt)
{
  str2opt_t::iterator it;
  // search option for the first occurrence of `=` character
  const char *opt_end = a_opt_start + strlen(a_opt_start);
  const char *opt_name_end = (const char *) memchr((const void *) a_opt_start, '=', \
						     opt_end - a_opt_start);
  if (opt_name_end == nullptr)
    opt_name_end = opt_end;

  std::string opt_name(a_opt_start, opt_name_end - a_opt_start);
  // check the name of the option in the long option map
  if ((it = m_long2opt.find(opt_name)) == m_long2opt.end()) {
    throw std::invalid_argument(std::string("Unrecognized option --") + opt_name);
  } else {
    opt_shptr_t opt = it->second;
    opt->m_specified = true;
    // obtain option's argument if it requires one
    if (opt->m_type != ArgType::NONE) {
      const char *arg_start = opt_name_end + 1;
      if (arg_start > opt_end) {
	if (++a_cnt >= a_argc)
	  throw std::invalid_argument(std::string("Argument missing for option --") + opt_name);

        arg_start = a_argv[a_cnt];
      }
      // convert argument to the required type
      get_arg_value(opt, arg_start);
    }
  }
  return ++a_cnt;
}

int OptParser::parse_short(const char *a_opt_start, const int a_argc, char *a_argv[], int &a_cnt)
{
  char2opt_t::iterator it;
  const char *opt_end = a_opt_start + strlen(a_opt_start);
  const char *opt_name_end = a_opt_start + 1;

  if ((it = m_short2opt.find(*a_opt_start)) == m_short2opt.end()) {
    throw std::invalid_argument(std::string("Unrecognized option -") + *a_opt_start);
  } else {
    opt_shptr_t opt = it->second;
    opt->m_specified = true;
    if (opt->m_type != ArgType::NONE) {
      const char *arg_start = opt_name_end;
      if (arg_start >= opt_end) {
	if (++a_cnt >= a_argc)
	  throw std::invalid_argument(std::string("Argument missing for option -") + *a_opt_start);

        arg_start = a_argv[a_cnt];
      }
      get_arg_value(opt, arg_start);
    } else if (*opt_end) {
      return parse_short(a_opt_start + 1, a_argc, a_argv, a_cnt);
    }
  }
  return ++a_cnt;
}

void OptParser::get_arg_value(opt_shptr_t a_opt, const char *a_arg_start) {
  switch (a_opt->m_type) {
  case ArgType::CHAR_PTR:
    a_opt->m_value.m_char_ptr = a_arg_start;
    break;
  case ArgType::INT:
    a_opt->m_value.m_int = atoi(a_arg_start);
    break;
  case ArgType::FLOAT:
    a_opt->m_value.m_float = (float) atof(a_arg_start);
    break;
  case ArgType::DOUBLE:
    a_opt->m_value.m_double = atof(a_arg_start);
    break;
  case ArgType::LONG:
    a_opt->m_value.m_long = atol(a_arg_start);
    break;
  case ArgType::LLONG:
    a_opt->m_value.m_llong = atoll(a_arg_start);
    break;
  }
}
