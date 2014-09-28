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
  for (i = 1; i < argc; ++i) {
    switch (*argv[i]) {
    case '-':
      chp = argv[i] + 1;
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
