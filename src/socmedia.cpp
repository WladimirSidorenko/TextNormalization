/** @file socmedia.cpp

    @brief Main pipeline for analyzing social media texts.

    Read input either in text or in JSON format, reconstruct
    discussions (if necessary), and analyze sentiment and discourse
    relations in text.
*/

///////////////
// Libraries //
///////////////
#include "OptionParser.h"

#include <locale.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <string>
#include <error.h>
#include <stdio.h>

///////////////
// Variables //
///////////////

/////////////
// Methods //
/////////////
static void process_txt()
{
}

static void process_tsv()
{
}

static void process_json()
{
}

//////////
// Main //
//////////

/**
 * Do full processing of either raw text or JSON file containing tweets.
 *
 * @param argc - number of arguments specified on command line
 * @param argv - array of comand line arguments
 *
 * @return \c 0 on success, non-\c 0 otherwise
 */
int main(int argc, char *argv[]) {

  // Process options
  Option::Parser opt_parser("Analyze plain text or Twitter discussions.");
  opt_parser.add_option('h', "help", "show this screen and exit");
  opt_parser.add_option('e', "encoding", "encoding of the input text (type `locale -a`\
 to see possible values)", Option::ArgType::CHAR_PTR, (const void *) "");
  opt_parser.add_option('f', "flush", "flush output buffer as soon as possible");
  opt_parser.add_option('j', "json", "accept input in JSON format (reconstruct and output discussions)");
  opt_parser.add_option('s', "skip-line", "line which should be skipped from processing", \
			Option::ArgType::CHAR_PTR, (const void *) "");
  opt_parser.add_option('t', "tsv", "accept input in TSV format (discussions are indented, 0-th field represents the id)");
  opt_parser.add_option(0, "txt", "accept input in plain text format (default)");

  int args_processed = opt_parser.parse(argc, argv);

  if (opt_parser.check("help")) {
    opt_parser.usage(std::cerr);
    return 0;
  }

  // set appropriate locale (invalid value will be ignored)
  setlocale(LC_ALL, *((const char **) opt_parser.get_arg('e')));

  // iterate over files specified as command line arguments
  std::wstring iline;
  std::wifstream ifstream;
  std::wistream *istream;

  for (int i = args_processed; i < argc || i == args_processed; ++i) {
    // open file for reading or use `std::wcin`
    if (argv[i] == nullptr || strncmp("-", argv[i], 2) == 0) {
      istream = &std::wcin;
    } else {
      ifstream.open(argv[i], std::wifstream::in);
      istream = &ifstream;
    }

    // read line
    while (std::getline(*istream, iline).good()) {
      std::wcout << "Line is: '" << iline << '\'' << std::endl;
    }

    // close input stream
    if (istream == &std::wcin)
      istream->clear();
    else
      ifstream.close();
  }
  return 0;
}
