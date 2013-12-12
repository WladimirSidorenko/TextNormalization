/** @file misspelling_restorer.cpp

    @brief Correct colloquial spellings in input according to rules.

    Read input lines in one word per line format and change words
    which appear to be unknown to dictionary, if these words become
    valid in-vocabulary terms after applying rules' transformations to
    them.
*/

///////////////
// Libraries //
///////////////
#include <assert.h>		// assert()
#include <error.h>		// error()
#include <getopt.h>		// struct option
#include <stdio.h>		// stderr, FILE
#include <stdlib.h>		// exit(), atexit()
#include <stddef.h>		// ssize_t
#include <string.h>		// strncmp()

#include <iostream>		// std::cerr

#include "Dictionary.h"		 // Dictionary()
#include "MisspellingRestorer.h" // MisspellingRestorer()
#include "StringTools.h"	 // rstrip()

////////////
// Macros //
////////////
/// maximum allowed length of input line
#define MAX_LINE_LEN 2000

///////////////
// Variables //
///////////////
/** @brief name of this program */
static char const *progname = "";
/** @brief exit status of the command >0 on error */
static int exit_status;
/** @brief boolean variable indicating whether XML lines should be skipped */
static bool skip_xml;
/** @brief boolean variable indicating whether XML lines should be skipped */
static const char* skip_line;
/** @brief encoding of input text */
static const char *ienc_p;
/** @brief language of input text */
static const char *ilang_p;
/** @brief variables needed for reading input lines */
static ssize_t ilinelen;
static size_t ilinesize = 2000;
static char* iline;

/** @brief external variable storing argv indices of options' arguments */
extern int optind;
/** @brief external variable pointing to options' argument */
extern char *optarg;
/** string describing allowed short options */
static char const *SHORTOPTS = "he:l:s:mX";
/** long option structures needed for processing options */
static const struct option LONGOPTS[] = {
  {"help",      no_argument, NULL, 'h'},
  {"encoding",  required_argument, NULL, 'e'},
  {"language",  required_argument, NULL, 'l'},
  {"multiword", no_argument, NULL, 'm'},
  {"skip-line", required_argument, NULL, 's'},
  {"skip-xml",  no_argument, NULL, 'X'},
  {0, 0, NULL, 0}
};

/////////////
// Methods //
/////////////

/**
 * @brief Clean dymaically allocated memory.
 */
static void make_cleanup(void) {
  if (iline) free(iline);
}

/**
 * @brief Read and process options from command-line.
 */
static void usage(int exit_status) {
  if (exit_status) {
    fprintf (stderr, "Try `%s --help' to see usage.\n",
	     progname);
  } else {
    printf ("NAME:\n%s\n\n",
	    progname);
    printf ("SYNOPSIS:\n%s [options] [FILE...]\n\n",
	    progname);
    printf ("DESCRIPTION:\n\
Read input lines in one word per line format and correct words\n\
which appear to be out-of-vocabulary in case when their transformation\n\
according to rules yields a valid in-vocabulary term.\n\n\
OPTIONS:\n\
-h, --help    print this screen and exit\n\
-e, --encoding ENCODING   encoding of input LINE (either iso-8859-1 or\n\
                          utf-8 which is used by default)\n\
-l, --language LANG   language of input text (either en_US or\n\
                          de_DE, or de_CH which is used by default)\n\
-s, --skip-line LINE   do not process input lines equal to LINE\n\
-X, --skip-xml   do not process lines consisting solely of XML tags\n\
\n\
DIAGNOSTICS:\n\
Exit status is 0 on successful processing and >0 if an error occurred.\n");
  }
  exit(exit_status);
}

/** @brief Read and process options from command-line.
 */
static void process_options(int &argc, char** &argv) {
  int opt;
  progname = argv[0];
  while ((opt = getopt_long(argc, argv, SHORTOPTS, LONGOPTS, NULL)) != -1) {
    switch (opt) {
    case 'h':
      usage(0);
    case 'e':
      ienc_p = optarg;
      break;
    case 'l':
      ilang_p = optarg;
      break;
    case 's':
      skip_line = optarg;
      break;
    case 'X':
      skip_xml = true;
      break;
    case '?':
    default:
      usage(1);
    }
  }
}

//////////
// Main //
//////////
/** @brief actual processing method of file */
int main(int argc, char* argv[]) {
  atexit(make_cleanup);
  // read and process options
  process_options(argc, argv);

  // initialize dictionary used for checking words
  Dictionary idict(ilang_p, ienc_p, true);

  // name of file which is opened or is going to be opened
  char const *fname;
  // pointer to current argument
  char const *arg;
  // automatic variable for input stream associated with file
  FILE *fp;
  // allocate memmory for input line
  iline = (char*) malloc(sizeof(char) * ilinesize);
  // process files specified on command line or stdin if no further
  // arguments were given
  do {
    arg = argv[optind];
    // open file for reading
    if (arg == NULL || ! strcmp(arg, "-")) {
      fname = "/dev/stdin";
      fp = stdin;
    } else {
      fname = arg;
      if ((fp = fopen(arg, "r")) == NULL)
	error(2, 0, "File '%s' could not be opened", fname);
    }

    // read and process file
    while ((ilinelen = getline(&iline, &ilinesize, fp)) != -1) {
      // remove trailing white spaces and newlines
      rstrip(iline, ilinelen);
      // check if skip_line is defined and do not change input line if
      // it is identical to skip line
      if ((! *iline) || (skip_line && strcmp(iline, skip_line) == 0) || \
	  (skip_xml && is_xml_tag(iline)))
	std::cout << iline << std::endl;
      else {
	// process input line
	if (! idict.checkw(iline))
	  std::cout << "!";
	std::cout << iline << std::endl;
      }
    }

    // close file
    if (fp != stdin) {
      if (fclose(fp) != 0)
	error(3, 0, "file '%s' could not be properly closed", fname);
    } else {
      // if we read from stdin, clear EOF marker from it, so
      // that subsequent reads from it would be possible
      clearerr(fp);
    }
  } while (++optind < argc);
  return exit_status;
}
