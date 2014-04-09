##########
# Set-up #
##########
m4_changequote(`[', `]')
m4_changequote([``], [''])

##########
# Macros #
##########
m4_define(``t_ADJX'', ``
$1	ADJA	$2
$1	ADJD	$2
'')

m4_define(``t_VVINF'', ``
$1	VVINF	$2
$1	VVFIN	$2
'')

m4_define(``t_ANY'', ``...'')
