/**
 * @file OptionParser.cpp
 *
 * @brief Implementation of Option::Parser class.
 *
 * @author Uladzimir Sidarenka <sidarenk@uni-potsdam.de>
 */

///////////////
// Libraries //
///////////////
#include "OptionParser.h"

#include <libgen.h>		// basename()
#include <cstring>		// memchr()

#include <iostream>		// std::endl
#include <ostream>		// std::endl
#include <stdexcept>		// std::invalid_argument

/////////////////
// Definitions //
/////////////////
namespace Option {

  // Parser
  Parser::Parser(const char *a_desc):
    m_short2opt{}, m_long2opt{}, m_desc{a_desc}, m_name{nullptr}, m_usage{}
{ }

  void Parser::add_option(const char a_short, const char *a_long, \
			     const char *a_desc, arg_type_t a_type, \
			     const void *a_default)
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
      iopt = std::make_shared<Option<const char *>>(a_short, a_long, a_desc, a_type, a_default);
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

    // insert option pointer in map
    if (a_short)
      m_short2opt.insert(char_opt_t(a_short, iopt));

    if (a_long)
      m_long2opt.insert(str_opt_t(lname, iopt));
  }

  int Parser::parse(const int a_argc, char *a_argv[])
  {
    int i = -1;
    char *chp;
    // set program name
    m_name = basename(a_argv[0]);
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

  void Parser::usage(std::ostream& a_ostream)
    const
  {
    // compose description message if necessary
    if (! *m_usage.c_str()) {
      // add program description
      m_usage += "DESCRIPTION:\n";
      m_usage += m_desc;
      // add USAGE message
      if (m_name != nullptr) {
	m_usage += "\n\nUSAGE\n";
	m_usage += m_name;
	m_usage += " [OPTION] [FILEs]";
      }
      // add option description
      m_usage += "\n\nOPTIONS:\n";
      m_usage += generate_opt_description();
    }
    a_ostream << m_usage;
  }

  const void *Parser::get_arg(const char a_short)
    const
  {
    char2opt_t::const_iterator ch_cit = m_short2opt.find(a_short);
    if (ch_cit == m_short2opt.end())
      return nullptr;
    else
      return ch_cit->second->get_arg();
  }

  const void *Parser::get_arg(const char *a_long)
    const
  {
    str2opt_t::const_iterator ch_cit = m_long2opt.find(a_long);
    if (ch_cit == m_long2opt.end())
      return nullptr;
    else
      return ch_cit->second->get_arg();
  }

  std::string Parser::generate_opt_description()
    const
  {
    std::string idesc;
    opt_shptr_t iopt;

    // first add options according to their short names
    char2opt_t::const_iterator ch_cit = m_short2opt.begin(), ch_cit_e = m_short2opt.end();

    for (; ch_cit != ch_cit_e; ++ch_cit) {
      iopt = ch_cit->second;
      idesc += "-";
      idesc += iopt->m_short_name;

      if (iopt->m_long_name) {
	idesc += "|--";
	idesc += iopt->m_long_name;
      }

      if (iopt->m_type != ArgType::NONE) {
	idesc += "=";
	idesc += iopt->atype2str();
      }

      if (iopt->m_desc) {
	idesc += "\t";
	idesc += iopt->m_desc;
      }
      idesc += "\n";
    }

    // then add options which don't have short names
    str2opt_t::const_iterator s_cit = m_long2opt.begin(), s_cit_e = m_long2opt.end();
    for (; s_cit != s_cit_e; ++s_cit) {
      iopt = s_cit->second;

      if (iopt->m_short_name)
	continue;

      idesc += "--";
      idesc += iopt->m_long_name;

      if (iopt->m_type != ArgType::NONE) {
	idesc += "=";
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

  int Parser::parse_long(const char *a_opt_start, const int a_argc, char *a_argv[], int &a_cnt)
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

  int Parser::parse_short(const char *a_opt_start, const int a_argc, char *a_argv[], int &a_cnt)
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
}
