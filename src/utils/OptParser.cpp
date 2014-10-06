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
OptParser::Option::Option(const char a_short, const char *a_long, arg_type_t a_type, \
			  const void *a_default):
  m_short_name{a_short}, m_long_name{a_long}, m_type{a_type}, m_value{}, m_specified{false}
{
  if (a_default)
    set_value(a_default);
}

const void *OptParser::Option::get_value()
  const
{
  switch (m_type) {
  case ArgType::CHAR_PTR:
    return &m_value.m_char_ptr;
  case ArgType::INT:
    return &m_value.m_int;
  case ArgType::FLOAT:
    return &m_value.m_float;
  case ArgType::DOUBLE:
    return &m_value.m_double;
  case ArgType::LONG:
    return &m_value.m_long;
  case ArgType::LLONG:
    return &m_value.m_llong;
  default:
    return nullptr;
  }
}

void OptParser::Option::set_value(const char *a_arg)
{
  switch (m_type) {
  case ArgType::CHAR_PTR:
    m_value.m_char_ptr = a_arg;
    break;
  case ArgType::INT:
    m_value.m_int = atoi(a_arg);
    break;
  case ArgType::FLOAT:
    m_value.m_float = (float) atof(a_arg);
    break;
  case ArgType::DOUBLE:
    m_value.m_double = atof(a_arg);
    break;
  case ArgType::LONG:
    m_value.m_long = atol(a_arg);
    break;
  case ArgType::LLONG:
    m_value.m_llong = atoll(a_arg);
    break;
  // default:
  //   throw
  }
}

void OptParser::Option::set_value(const void *a_arg)
{
  switch (m_type) {
  case ArgType::CHAR_PTR:
    m_value.m_char_ptr = std::static_cast<const char *>(a_arg);
    break;
  case ArgType::INT:
    m_value.m_int = std::static_cast<int>(a_arg);
    break;
  case ArgType::FLOAT:
    m_value.m_float = std::static_cast<float>(a_arg);
    break;
  case ArgType::DOUBLE:
    m_value.m_double = std::static_cast<double>(a_arg);
    break;
  case ArgType::LONG:
    m_value.m_long = std::static_cast<long>(a_arg);
    break;
  case ArgType::LLONG:
    m_value.m_llong = std::static_cast<long long>(a_arg);
    break;
  }
}

OptParser::OptParser(const char *a_desc):
  m_short2opt{}, m_long2opt{}, m_desc{a_desc}, m_name{nullptr}, m_parsed{0}
{ }

void OptParser::add_option(const char a_short, const char *a_long, \
			   const char *a_desc, arg_type_t a_type, \
			   void *a_default)
{
  // check taht option is not already defined
  if ((a_short && m_short2opt.find(a_short) != m_short2opt.end())) ||	\
    throw std::invalid_argument(std::string("Option '-") + a_short + std::string("' already defined."));;

  if (a_long && m_long2opt.find(a_long) != m_short2opt.end())
    throw std::invalid_argument(std::string("Option '--") + a_long + std::string("' already defined."));;

  if (! a_short && ! a_long)
    throw std::invalid_argument("No option name specified.");;

  // create option
  std::shared_ptr<Option> iopt = std::make_shared<Option>(a_short, a_long, a_type, a_default);

  // insert option in map
  if (a_short)
    m_short2opt.insert(char_opt_t(a_short, iopt));

  if (a_long)
    m_long2opt.insert(str_opt_t(std::string(a_long), iopt));
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
      opt->set_value(arg_start)
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
      opt->set_value(arg_start);
    } else if (*opt_end) {
      return parse_short(a_opt_start + 1, a_argc, a_argv, a_cnt);
    }
  }
  return ++a_cnt;
}
