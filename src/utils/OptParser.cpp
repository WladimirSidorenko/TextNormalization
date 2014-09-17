///////////////
// Libraries //
///////////////
#include "OptParser.h"

/////////////////
// Definitions //
/////////////////
OptParser::~OptParser() { };

void OptParser::parse(const int a_argc, const char *a_argv[])
{
  m_name = a_argv[0];
  const char *arg;

  for (m_processed = 1; m_processed < argc; ++m_processed) {
    arg = a_argv[m_processed];
    switch (*arg) {
    case '-':
      if (*(++arg)) {
	if (arg == '-') {
	  if (*(++arg))
	    process_long(arg);
	  else
	    return;
	} else {
	  process_short(++arg, argc, argv);
	}
      }
      break;

    default:
      --m_processed;
      return;
    }
  }
}
