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
OptParser::OptionBase::OptionBase():
  m_short_name{}, m_long_name{}, m_desc{}, m_type{}, m_specified{}
{}

OptParser::OptionBase::OptionBase(const char a_short, const char *a_long, \
				  const char *a_desc, arg_type_t a_type):
  m_short_name{a_short}, m_long_name{a_long}, m_desc{a_desc},	\
  m_type{a_type}, m_specified{false}
{}

template<typename T>
OptParser::Option<T>::Option(const char a_short, const char *a_long,	\
			     const char *a_desc, arg_type_t a_type, void *a_default):
  OptionBase{a_short, a_long, a_desc, a_type}, m_value{}
{
  if (a_default)
    set_value(*(static_cast<T *>(a_default)));
}

template<typename T>
T OptParser::Option<T>::get_value()
  const
{
  return m_value;
}

template<typename T>
void OptParser::Option<T>::set_value(T a_value)
{
  m_value = a_value;
}

// Template specialization for argument parse functions
template<>
void OptParser::Option<const char *>::parse_arg(const char *a_value)
{
  m_value = a_value;
}

template<>
void OptParser::Option<int>::parse_arg(const char *a_value)
{
  m_value = atoi(a_value);
}

template<>
void OptParser::Option<float>::parse_arg(const char *a_value)
{
  m_value = static_cast<float>(atof(a_value));
}

template<>
void OptParser::Option<double>::parse_arg(const char *a_value)
{
  m_value = atof(a_value);
}

template<>
void OptParser::Option<long>::parse_arg(const char *a_value)
{
  m_value = atol(a_value);
}

template<>
void OptParser::Option<long long>::parse_arg(const char *a_value)
{
  m_value = atoll(a_value);
}

template<typename T>
void OptParser::Option<T>::parse_arg(const char *a_value)
{
  std::string err_msg = "Unknown argument type for option  -";
  if (m_short_name) {
    err_msg += m_short_name;
  } else {
    err_msg += '-';
    err_msg += m_long_name;
  }
  err_msg += "'.";

  throw std::invalid_argument(err_msg);
}

template<>
const char* OptParser::Option<const char *>::atype2str()
  const
{
  return "STR";
}

template<>
const char* OptParser::Option<int>::atype2str()
  const
{
  return "INT";
}

template<>
const char* OptParser::Option<float>::atype2str()
  const
{
  return "FLOAT";
}

template<>
const char* OptParser::Option<double>::atype2str()
  const
{
  return "DOUBLE";
}

template<>
const char* OptParser::Option<long>::atype2str()
  const
{
  return "LONG";
}

template<>
const char* OptParser::Option<long long>::atype2str()
  const
{
  return "LONGLONG";
}

template<typename T>
const char* OptParser::Option<T>::atype2str()
  const
{
  return "";
}

// OptParser
OptParser::OptParser(const char *a_desc):
  m_short2opt{}, m_long2opt{}, m_desc{a_desc}, m_name{nullptr}, m_usage{}
{ }

void OptParser::add_option(const char a_short, const char *a_long, \
			   const char *a_desc, arg_type_t a_type, \
			   void *a_default)
{
  std::string lname{a_long};
  // check that option is specified and is not already defined
  if (! a_short && ! a_long)
    throw std::invalid_argument("No option name specified.");

  if (a_short && m_short2opt.find(a_short) != m_short2opt.end())
    throw std::invalid_argument(std::string("Option '-") + a_short + \
  				std::string("' already defined."));

  if (a_long && m_long2opt.find(lname) != m_long2opt.end())
    throw std::invalid_argument(std::string("Option '--") + a_long + \
  				std::string("' already defined."));

  // create option
  opt_shptr_t iopt = nullptr;
  switch(a_type){
  case ArgType::NONE:
    iopt = std::make_shared<Option<void *>>(a_short, a_long, a_desc, a_type, a_default);
    break;
  case ArgType::CHAR_PTR:
    iopt = std::make_shared<Option<char *>>(a_short, a_long, a_desc, a_type, a_default);
    break;
  case ArgType::INT:
    iopt = std::make_shared<Option<int>>(a_short, a_long, a_desc, a_type, a_default);
    break;
  case ArgType::FLOAT:
    iopt = std::make_shared<Option<float>>(a_short, a_long, a_desc, a_type, a_default);
    break;
  case ArgType::DOUBLE:
    iopt = std::make_shared<Option<double>>(a_short, a_long, a_desc, a_type, a_default);
    break;
  case ArgType::LONG:
    iopt = std::make_shared<Option<long>>(a_short, a_long, a_desc, a_type, a_default);
    break;
  case ArgType::LLONG:
    iopt = std::make_shared<Option<long long>>(a_short, a_long, a_desc, a_type, a_default);
    break;
  }

  // insert option ponter in map
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

int OptParser::get_arg(const char a_short, void *a_trg)
  const
{
  return 0;
}

int OptParser::get_arg(const char *a_long, void *a_trg)
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
      idesc += iopt->atype2str();
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
      idesc += iopt->atype2str();
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
