/** @file StringTools.h
 *
 *  @brief Declaration of various functions for handling strings.
 *
 *  @author Uladzimir Sidarenka <sidarenk@uni-potsdam.de>
 */

#ifndef __STRINGTOOLS_H_GUARD__
# define __STRINGTOOLS_H_GUARD__ 1
/////////////
// Headers //
/////////////
#include <string.h>		// ssize_t

/////////////
// Methods //
/////////////

/// Delete characters from the right end of string
char *rstrip(char* &iline, ssize_t &ilinelen, const char* chars2delete = NULL);
/// Check if istring consists solely of XML tags
bool is_xml_tag(const char *istring);
#endif
