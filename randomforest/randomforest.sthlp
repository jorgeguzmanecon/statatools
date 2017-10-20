{smcl}
{* *! version 1.1.8  06mar2015}{...}

{title:Title}

{p2colset 5 19 21 2}{...}
{p2col :{manlink D randomforest} {hline 2}}Wrapper for Python's Scikit random forest{p_end}

	Created by: Jorge Guzman (jorgeg@mit.edu)
	Last Updated: October, 2017
	Requirements: 
	- Must have python 2.7 and scikit-learn installed.
	- Must have files ~/ado/randomforest.ado and ~/ado/randomforest/randomforest_asdfa.py
	
	Please visit http://www.jorgeguzman.co/code for updates

{p2colreset}{...}


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{opt randomforest} {it:{help varlist}}  {cmd:,} {opt gen(string)}  [{opt logit}  {opt train_index(varname)}  {opt predict_index(varname)}  {opt roconly} {opt store_roc(string)}]




{marker description}{...}
{title:Description}

{pstd}
{cmd:randomforest} is a wrapper for Python's scikit-learn implementation of random forest. Its purpose 
is to allow researchers to easily build random forest through Stata. 



{marker options}{...}
{title:Options}

{phang}
{opt gen} the name of the variable that will hold the predicted value of the random forest

{phang}
{opt logit} if you also want to run a logit, for comparability purposes.

{phang}
{opt train_index} an variable with a value of 1 for the observations that will be used to train the model, and 0 otherwise.

{phang}
{opt predict_index} an variable with a value of 1 for the observations that will be used to predict the model, and 0 otherwise. (only matters for reported ROC Scores)

{phang}
{opt roconly} do not report the index, only the ROC

{phang}
{opt store_roc} store an ROC curve under this path.
