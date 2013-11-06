/** @file Dictionary.cpp

    @brief Implementation of Dictionary class.

    This file actually implements all methods, constructor, and
    destructor of the Dictionary class.
*/

///////////////
// Libraries //
///////////////
#include <assert.h>		// assert()
#include <ctype.h>		// toupper(), tolower()
#include <error.h>		// error()
#include <errno.h>		// ENOMEM
#include <stdlib.h>		// getenv()
#include <string.h>		// strlen()
#include <strings.h>		// strcasecmp()
#include <sys/stat.h>		// stat()
#include <sys/types.h>		// stat() related
#include <unistd.h>		// stat() related

#include "Dictionary.h"		// declarations of members and methods

////////////
// Macros //
////////////
/// extension of an affix file
#define AFFSUFFIX  ".aff"
/// extension of a dictionary file
#define DICTSUFFIX ".dic"
/// maximum allowed length of a path
#define MAXPATHLEN 1000

/// search path for Hunspell dictionaries
#define LIBDIR \
    "/usr/share/hunspell:" \
    "/usr/share/myspell:" \
    "/usr/share/myspell/dicts:" \
    "/Library/Spelling"
/**
 * supplementary search path for Hunspell dictionaries in case they
 * are located somewhere in OpenOffice directory
 */
#define USEROOODIR \
    ".openoffice.org/3/user/wordbook:" \
    ".openoffice.org2/user/wordbook:" \
    ".openoffice.org2.0/user/wordbook:" \
    "Library/Spelling"

/**
 * one more supplementary search path for Hunspell dictionaries which
 * might be shared with OpenOffice
 */
#define OOODIR \
    "/opt/openoffice.org/basis3.0/share/dict/ooo:" \
    "/usr/lib/openoffice.org/basis3.0/share/dict/ooo:" \
    "/opt/openoffice.org2.4/share/dict/ooo:" \
    "/usr/lib/openoffice.org2.4/share/dict/ooo:" \
    "/opt/openoffice.org2.3/share/dict/ooo:" \
    "/usr/lib/openoffice.org2.3/share/dict/ooo:" \
    "/opt/openoffice.org2.2/share/dict/ooo:" \
    "/usr/lib/openoffice.org2.2/share/dict/ooo:" \
    "/opt/openoffice.org2.1/share/dict/ooo:" \
    "/usr/lib/openoffice.org2.1/share/dict/ooo:" \
    "/opt/openoffice.org2.0/share/dict/ooo:" \
    "/usr/lib/openoffice.org2.0/share/dict/ooo"
/// macor for HOME directory path of a user
#define HOME    getenv("HOME")
/// macor for obtaining directory path from `PRJ_DICTIONARY`
/// environmental variable
#define PRJDICT getenv("PRJ_DICTIONARY")
/// separator of directories in a single path
#define DIRSEP "/"
/// separator of single paths in search path for dictionaries
#define PATHSEP ":"

///////////////
// Variables //
///////////////

///////////
// Class //
///////////

/**
 * Constructor of Dictionary class.
 * @param dictname - name of dictionary to be used for checking language
 * @param encoding - encoding of input language
 * @param ignore_case - do not take case into account when checking words
 */
Dictionary::Dictionary(const char* dictname, const char *encoding, \
		       bool ignore_case):
  m_lang(dictname), m_encoding(encoding), m_ignore_case(ignore_case)
{
  if (! dictname)
    dictname = DEFAULT_DICT;
  if (! encoding)
    encoding = DEFAULT_ENC;

  char *affpath, *dictpath;
  if (! (affpath  = find_file(NULL, dictname, AFFSUFFIX)) ||
      ! (dictpath = find_file(NULL, dictname, DICTSUFFIX))) {
    free(affpath);
    error(EINVAL, EINVAL, "Unknown dictionary name '%s'", dictname);
  }
  m_dictp = new Hunspell(affpath, dictpath);
  free(affpath); free(dictpath);
}

/**
 * Destructor of Dictionary class.
 *
 * This function release dynamically allocated memory for Hunspell
 * dictionary.
 */
Dictionary::~Dictionary(void)
{
  delete m_dictp;
}

/**
 * check word in dictionary and return true if it was found
 *
 * @param iword - pointer to word to be checked
 *
 * @return true if word was found in dictionary
 */
bool Dictionary::checkw(const char *iword)
  const
{
  // first check if given word is known to dictionary as is
  if (m_dictp->spell(iword))
    return true;
  // if case should not be ignored during testing, treturn false
  if (! m_ignore_case)
    return false;
  // otherwise, try capitalized version of this word, since German
  // dictionaries are case sensitive
  bool ret = false;
  char *word_copy = strdup(iword);
  if (word_copy == NULL)
    error(ENOMEM, ENOMEM, "Could not allocate memory for string copy.");
  // capitalize copied string and check capitalized version,
  // subsequently freeing the memory
  word_copy[0] = toupper(word_copy[0]);
  for (size_t i = 1; word_copy[i]; ++i) {
    word_copy[i] = tolower(word_copy[i]);
  }
  ret = m_dictp->spell(word_copy);
  free(word_copy);
  return ret;
}

/**
 * Suggest correction variants for word `iword`.
 *
 * @param iword - input word for which suggestions should be generated
 *
 * @return pointer to an array of char pointers each of which points
 * to a suggestions string
 */
char** Dictionary::suggestw(const char *iword)
  const
{
  return NULL;
}

/**
 * Search for file `fname` + `ext` in system dictionary path and
 * return pointer to the newly constructed path or NULL if dict was
 * not found. This function mimics the procedure in `hunspel.cxx`.
 *
 * @param srch_path - colon separated list of directories in which
 *                    dicitionaries should be searched (default search
 *                    path is used if this parameter is set to NULL)
 * @param fname - name of the file to be searched
 * @param ext   - extension of the file to be searched
 *
 * @return pointer to the constructed valid path or NULL if file was
 * not found
 */
char *Dictionary::find_file(const char* srch_path,		\
			    const char* fname,			\
			    const char* ext)
{
  // if no search path was provided, use a default one
  if (! srch_path)
    srch_path = get_default_dict_path();
  // allocate memory for concatenation of `fname` and `ext`
  size_t fnlen   = strlen(fname) + strlen(ext) + 1;
  char *filename = (char *) malloc(sizeof(char) * fnlen);
  if (filename == NULL)
    error(ENOMEM, ENOMEM, "Could not allocate memory for search file name.");
  // construct file name by concatenating `fname` and `ext`
  strcat(strcpy(filename, fname), ext);
  // beware, filepath (aka "found file path") is not freed if file is
  // found, since it's also the return value of this function
  char *filepath = (char *) malloc(sizeof(char) * (MAXPATHLEN + fnlen));
  if (filepath == NULL)
    error(ENOMEM, ENOMEM, "Could not allocate memory for search file path.");
  // pointer to the search path part under consideration
  const char *path_start = srch_path;
  const char *path_end = path_start;
  // length of a single path in `srch_path` list
  size_t path_len;
  // dummy buffer for storing stat() information
  struct stat statbuf;
  while (1) {
    while (!((*path_end == *PATHSEP) || (*path_end == '\0'))) ++path_end;
    path_len = path_end - path_start;
    // if path prefix length + the length of filename fits into
    // allocated memory, construct a new path
    if (path_len < MAXPATHLEN) {
      strncpy(filepath, path_start, path_len);
      strcpy(filepath + path_len++, DIRSEP);
      strcpy(filepath + path_len, filename);
      // if file was found, return it
      if (stat(filepath, &statbuf) == 0) {
	free(filename);
	return filepath;
      }
    }
    // if the list of possible directories is exhausted, then break
    if (*path_end == '\0')
      break;
    // start looking for the next path
    path_start = ++path_end;
  }
  free(filepath); free(filename);
  return NULL;
}

/**
 * Return or generate a list of directories in which dictionaries should be
 * searched by default.
 */
const char* Dictionary::get_default_dict_path(void) {
  static char *default_dict_path;
  if (default_dict_path)
    return default_dict_path;

  // construct a vector of const char pointers for successive elements
  // which should be added to the search path list
  const char *paths[] = {PRJDICT, ".", "", getenv("DICPATH"), LIBDIR, \
			 HOME, USEROOODIR, OOODIR};
  size_t i, j = sizeof(paths) / sizeof(const char *);
  default_dict_path = (char *) malloc(sizeof(char));
  size_t newlen = 1, pslen = strlen(PATHSEP);
  // successively add each element in `paths` to `default_dict_path`
  for (i = 0; i < j; ++i) {
    if (paths[i] == NULL)
      continue;
    newlen += strlen(paths[i]) + pslen;
    assert(realloc(default_dict_path, newlen));
    strcat(default_dict_path, paths[i]);
    strcat(default_dict_path, PATHSEP);
  }
  if (newlen-- > 1)
    --newlen;
  default_dict_path[newlen] = '\0';
  return default_dict_path;
}
