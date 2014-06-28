/** @file StringTools.cpp

    @brief File implementing various string related methods.

    This file actually provides several methods for manipulating
    strings.
*/

///////////////
// Libraries //
///////////////
#include <ctype.h>		// isspace()
#include <stdio.h>		// ssize_t
#include <string.h>		// srcchr()

///////////////
// Variables //
///////////////

////////////////////
// Global Methods //
////////////////////

/**
 * @brief remove characters from the right end of line
 *
 * Remove from the right end of character array `iline` characters which
 * appear in `chars2delete`. If `chars2delete` points to NULL, white spaces
 * and newlines will be deleted by default.  Note, that the actual size of the
 * character array `iline` is not changed, this method simply puts a '0'
 * character instead of the first trailing character from `chars2delete`.
 *
 * @param iline - pointer to string from which characters should be stripped
 * @param ilinelen - length of `iline`
 * @param chars2delete - array of characters which should be stripped off from `iline`
 *
 * @return - pointer to changed `iline`
 */
char *rstrip(char* &iline, ssize_t &ilinelen, const char* chars2delete) {
  int ret;
  ssize_t llen = ilinelen;
  if (chars2delete) {
    while (llen && (ret = (strchr(chars2delete, iline[--llen]) != NULL)));
  } else {
    while (llen && (ret = isspace(iline[--llen])));
  }

  if (llen == 0 && ret)
    ilinelen = 0;
  else
    ilinelen = llen + 1;

  iline[ilinelen] = '\0';
  return iline;
}

/**
 * @brief check if input string consists solely of XML tags
 *
 * Consecutively check if string `istring` consists solely of XML tags and
 * return true if it does. Beware of strings which consist soleley of spaces
 * which are also considered as XML tags due to recursive implementation of
 * this function.
 *
 * @param istring - pointer to input string
 *
 * @return - \c true if given string appears to be an XML tag
 */
bool is_xml_tag(const char *istring) {
  char c;
  const char *stringp = istring;
  while ((c = *(stringp++))) {
    fprintf(stderr, "c = %c\n", c);
    if (isalnum(c)) {
      return false;
    } else if (ispunct(c)) {
      if (c != '<')
	return false;

      while ((stringp = strchr(istring, '>'))) {
	if (*(stringp - 1) == '\\' && ((stringp - istring) < 2 || \
				       *(stringp - 2) != '\\'))
	  continue;

	return is_xml_tag(++stringp);
      }
      return false;
    } else if (isspace(c))
      continue;
    else
      return false;
  }
  return true;
}
