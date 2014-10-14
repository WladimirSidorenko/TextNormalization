/**
 * @file OptParser.cpp
 *
 * @brief Implementation of OptParser class.
 *
 * @author Uladzimir Sidarenka <sidarenk@uni-potsdam.de>
 */

///////////////
// Libraries //
///////////////
#include "OptParser.h"

#include <stdexcept>		// std::invalid_argument
#include <cstdlib>		// atol(), atof() etc.
#include <cstring>		// memchr()
#include <ostream>		// std::endl

/////////////////
// Definitions //
/////////////////

// Option
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

void OptParser::Option::set_value(const void *a_arg)
{
  switch (m_type) {
  case ArgType::CHAR_PTR:
    m_value.m_char_ptr = static_cast<const char *>(a_arg);
    break;
  case ArgType::INT:
    m_value.m_int = *(static_cast<const int *>(a_arg));
    break;
  case ArgType::FLOAT:
    m_value.m_float = *(static_cast<const float *>(a_arg));
    break;
  case ArgType::DOUBLE:
    m_value.m_double = *(static_cast<const double *>(a_arg));
    break;
  case ArgType::LONG:
    m_value.m_long = *(static_cast<const long*>(a_arg));
    break;
  case ArgType::LLONG:
    m_value.m_llong = *(static_cast<const long long *>(a_arg));
    break;
  default:
    if (m_short_name)
      throw std::invalid_argument(std::string("Option '-") + m_short_name + \
				  std::string("' does not accept argument."));
    else
      throw std::invalid_argument(std::string("Option '--") + m_long_name + \
				  std::string("' does not accept argument."));
  }
}

void OptParser::Option::parse_arg(const char *a_arg)
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
  default:
    if (m_short_name)
      throw std::invalid_argument(std::string("Option '-") + m_short_name + \
				  std::string("' does not accept argument."));
    else
      throw std::invalid_argument(std::string("Option '--") + m_long_name + \
				  std::string("' does not accept argument."));
  }
}

const char* OptParser::Option::arg_type2str(const arg_type_t a_type)
{
  switch (a_type) {
  case ArgType::CHAR_PTR:
    return "STR";
  case ArgType::INT:
    return "INT";
  case ArgType::FLOAT:
    return "FLOAT";
  case ArgType::DOUBLE:
    return "DOUBLE";
  case ArgType::LONG:
    return "LONG";
  case ArgType::LLONG:
    return "LONGLONG";
  default:
    return "";
  }
}

OptParser::OptParser(const char *a_desc):
  m_short2opt{}, m_long2opt{}, m_desc{a_desc}, m_name{nullptr}, m_usage{}
{ }

void OptParser::add_option(const char a_short, const char *a_long, \
			   const char *a_desc, arg_type_t a_type, \
			   void *a_default)
{
  std::string lname = a_long;
  // check taht option is not already defined
  if (! a_short && ! a_long)
    throw std::invalid_argument("No option name specified.");

  if (a_short && m_short2opt.find(a_short) != m_short2opt.end())
    throw std::invalid_argument(std::string("Option '-") + a_short + std::string("' already defined."));

  if (a_long && m_long2opt.find(lname) != m_long2opt.end())
    throw std::invalid_argument(std::string("Option '--") + a_long + std::string("' already defined."));

  // create option
  std::shared_ptr<Option> iopt = std::make_shared<Option>(a_short, a_long, a_type, a_default);

  // insert option in map
  if (a_short)
    m_short2opt.insert(char_opt_t(a_short, iopt));

  if (a_long)
    m_long2opt.insert(str_opt_t(lname, iopt));
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

void OptParser::usage(std::ostream& a_ostream)
  const
{
  // compose description message if necessary
  if (! *m_usage.c_str()) {
    // add program description
    m_usage += "DESCRIPTION:\n";
    m_usage += m_desc;
    // add USAGE message
    if (m_name == nullptr) {
      m_usage += "\nUSAGE\n";
      m_usage += m_name;
      m_usage += " [OPTION] [FILEs]\n";
    }
    // add option description
    m_usage += "OPTIONS:\n";
    m_usage += generate_opt_description();
  }
  a_ostream << m_usage << std::endl;
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

std::string& OptParser::generate_opt_description()
  const
{
  std::string idesc;
  opt_shptr_t iopt;

  // first add options according to their short names
  char2opt_t::const_iterator ch_cit = m_short2opt.begin(), ch_cit_e = m_short2opt.end();

  for (; ch_cit != ch_cit_e; ++ch_cit) {
    iopt = ch_cit->second;
    idesc += iopt->m_short_name;

    if (iopt->m_long_name) {
      idesc += "|";
      idesc += iopt->m_long_name;
    }

    if (iopt->m_type != ArgType::NONE) {
      idesc += Option::arg_type2str(iopt->m_type);
    }

    if (iopt->m_desc) {
      idesc += iopt->m_desc;
    }
    idesc += "\n";
  }

  // then add options which don't have short names
  str2opt_t::const_iterator s_cit = m_long2opt.begin(), s_cit_e = m_long2opt.end();
  for (; ch_cit != ch_cit_e; ++ch_cit) {
    if (iopt->m_short_name)
      continue;

    idesc += "--";
    idesc += iopt->m_long_name;

    if (iopt->m_type != ArgType::NONE) {
      idesc += Option::arg_type2str(iopt->m_type);
    }

    if (iopt->m_desc) {
      idesc += "\t";
      idesc += iopt->m_desc;
    }
    idesc += "\n";
  }

  return idesc;
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
      opt->parse_arg(arg_start);
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
      opt->parse_arg(arg_start);
    } else if (*opt_end) {
      return parse_short(a_opt_start + 1, a_argc, a_argv, a_cnt);
    }
  }
  return ++a_cnt;
}
