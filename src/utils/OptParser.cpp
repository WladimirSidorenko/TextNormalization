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
OptParser::~OptParser() { };

void OptParser::parse(const int a_argc, char *a_argv[])
{
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
