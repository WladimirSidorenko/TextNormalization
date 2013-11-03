#ifndef __DICTIONARY_GUARD_H__
# define __DICTIONARY_GUARD_H__ 1
/////////////
// Headers //
/////////////
#include <stddef.h>		 // size_t
#include <hunspell/hunspell.hxx> // Hunspell()

////////////
// Macros //
////////////
#define CHARSIZE 10
#define DEFAULT_DICT "de_CH"
#define DEFAULT_ENC  "utf-8"

////////////////
// Data Types //
////////////////

///////////
// Class //
///////////

/// Proxy class providing interface to Hunspell dictionary.
struct Dictionary {
  /* Data Members */

  /// language of dictionary
  const char *m_lang;
  /// encoding of dictionary and incoming text
  const char *m_encoding;
  /// pointer to Hunspell dictionary
  Hunspell *m_dictp;

  /* Methods */

  /// Class' Constructor
  Dictionary(const char* dictname = DEFAULT_DICT, \
	     const char* encoding = DEFAULT_ENC);
  /// Class' Destructor
  ~Dictionary(void);

  /// check word in dictionary and return true if it was found
  bool checkw(const char *iword) const;
  /// suggest correction variants for word
  char** suggestw(const char *iword) const;

 private:
  /* Methods */
  /// return list of paths in which dictionaries should be searched
  static const char* get_default_dict_path(void);

  /// search for file `fname` + `ext` in system dictionary path
  static char *find_file(const char* srch_path, const char* fname, \
			       const char* ext);
};
#endif