#ifndef __MISSPELLING_RESTORER_HGUARD__
# define __MISSPELLING_RESTORER_HGUARD__ 1
///////////////
// Libraries //
///////////////
#include "Dictionary.h"	// Dictionary()

/////////////////////
// Data Structures //
/////////////////////

/// struct comprising a single rule for correcting text
struct MRRule {
 /// pointer to a function which should check whether given rule
 /// should be applied
  bool (*predfuncp)(const char *);
 /// pointer to a function which should modify given input string
 /// according to rule
  char *(*modfuncp)(const char *);
};

///////////
// Class //
///////////

/// class comprising a set of `struct MRRule`s and providing methods
/// for making corrections
class MisspellingRestorer {
 public:
  /// Constructor of class
  MisspellingRestorer(Dictionary *idctp, const MRRule *rulesp[]);
  /// Correct word if rule says to do so
  void correct(const char* &iword);

 private:
};
#endif
