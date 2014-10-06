/** @file socmedia.cpp

    @brief Main pipeline for analyzing social media texts.

    Read input either in text or in JSON format, reconstruct
    discussions (if necessary), and analyze sentiment and discourse
    relations in text.
*/

///////////////
// Libraries //
///////////////
#include "OptParser.h"

#include <iostream>
#include <fstream>
#include <string>
#include <error.h>
#include <stdio.h>

///////////////
// Variables //
///////////////
static std::ifstream istream;
static std::string iline;

/////////////
// Methods //
/////////////

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
  OptParser opt_parser{"Analyze plain text or Twitter discussions."};
  // opt_parser.add_option('h', "help", "show this screen and exit", OptParser::help);
  opt_parser.parse(argc, argv);

  // iterate over command line files
  // for (auto& fname: opt_parser.args["files"]) {
  //   std::cout << "fname = '" << fname << "'" << std::endl;
  //   // // open input stream associated with given file
  //   // istream.open(fname, std::ifstream::in);
  //   // // read line
  //   // while (std::getline(istream, iline).good()) {
  //   //   std::cout << "Line is: '" << iline << '\'' << std::endl;
  //   // }
  //   // // close open input stream
  //   // istream.close();
  // }
  return 0;
}
