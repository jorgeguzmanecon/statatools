{smcl}
{* *! version 1.1.8  06mar2015}{...}

{title:Title}

{p2colset 5 19 21 2}{...}
{p2col :{manlink D matrix_ttest} {hline 2}}Perform a t-test amongst many pairs of {p_end}

	Created by: Jorge Guzman (jag2367@gsb.columbia.edu)
	Last Updated: October, 2018
	Requirements: None
	Please visit http://www.jorgeguzman.co/code for updates

{p2colreset}{...}


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{opt matrix_ttest} [if] {it:{help varname}}  {cmd:,} {opt by(varname)}  [{opt save(string)}] [{opt format(string)}] [{opt p}] [{opt df}]




{marker description}{...}
{title:Description}

{pstd}
{cmd:matrix_ttest} creates ttests by groups for all groups, prints the difference in means between the two variables, the standard deviation of the difference, and the two-sided t-test p-value.


{marker options}{...}
{title:Options}

{phang}
{opt varname} the variable with the value

{phang}
{opt by} the variable with the groups to compare

{phang}
{opt save} if given, the resulting matrix (in long form) is stored in the path provided.

{phang}
{opt format} change the format of the output. Default is %9.3f


{phang}
{opt p} also display p values.


{phang}
{opt df} also display degrees of freedom



{marker example}{...}
{title:Examples}

The following example compares the level of entrepreneurial quality by county for all states in the Startup Cartography Project.
example[
	{
	
	clear 
	use https://www.startupmaps.us/s/counties_share_1988_2012.dta
	//allstates	
	gen log_quality = ln(quality)
	matrix_ttest log_quality, by(datastate)

	//All years in one state
	matrix_ttest quality if datastate == "CA", by(incyear)
]
{smcl}