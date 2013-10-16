#ifndef __MISSPELLING_RESTORER_HGUARD__
# define __MISSPELLING_RESTORER_HGUARD__ 1
///////////////
// Libraries //
///////////////
#include "Dictionary.h"	// Dictionary()

/////////////////////
// Data Structures //
/////////////////////
struct MRRule {
  bool (*predfuncp)(const char *);
  char *(*modfuncp)(const char *);
};

///////////
// Class //
///////////
class MisspellingRestorer {
 public:
  MisspellingRestorer(Dictionary *idctp, const MRRule *rulesp[]);

  void correct(char* &irule);

 private:
};
#endif
