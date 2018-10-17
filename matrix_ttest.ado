
/************************************************************************
 ** Creates a matrix of ttests for a variable by type and reports the mean
 ** difference and significance.
 ** File must be in long format first (i.e. melted)
 ** by variable must be string
**********************************************************************/
 
program define matrix_ttest , rclass
    syntax varname [if] , by(varname) [save(string)] [format(string)] [p] [df] [order(string)]
{
    preserve
    /*** setup some default values **/
   
    quietly { 
        local ttestvar=subinstr("`1'",",","",.)
        replace `by' = "_blank" if `by' == ""

        if "`if'" != "" {
            keep `if'
        }
        
        if"`order'" == "" {
            levelsof `by', local(bylevels) clean
        }
        else {
            local bylevels `order'
        }

        if "`format'" == "" {
            local format format(%9.3f)
        }
        else {
            local format format(`format')
        }
    }
    /** end of defaults setup **/

    quietly{
        tempfile results
        capture postclose diffs
        postfile diffs str30 left str30 right str20 stat value using `results', every(1)

        //Estimate all t-test combinations
        foreach left in `bylevels' {
            foreach right in `bylevels' {
     
                if "`left'" == "`right'" {
                    post diffs ("`left'") ("`right'") ("diff") (0) 
                    continue
                }

                ttest `ttestvar'  if inlist(`by',"`left'","`right'"), by(`by')
                local diff = `r(mu_1)' - `r(mu_2)'

                if "`right'" < "`left'" {
                    //If this was the case, it did it in the opposite order we intended. Invert the value.
                    local diff = `diff' * -1
                }
                
                post diffs ("`left'") ("`right'") ("diff") (`diff')
                post diffs ("`left'") ("`right'") ("se") (`r(se)')
                post diffs ("`left'") ("`right'") ("p") (`r(p)')
                post diffs ("`left'") ("`right'") ("t") (`r(t)')
                post diffs ("`left'") ("`right'") ("df") (`r(df_t)')
            }
        }
        postclose diffs

    }
    use `results', replace
    if "`save'" != "" {
        save `save'
    }

    if "`order'" != "" {
        local i  = 1
        foreach o in `order' {
            replace left = "`i'. " + left if left == "`o'"
            replace right = "`i'. " + right if right == "`o'"
            local i = `i'+1
        }
    }
    
    tabdisp  stat right if inlist(stat,"diff","se","`p'","`df'"), cellvar(value) by(left) `format' missing concise 
    restore

    tabstat `ttestvar', by(`by') statistics(mean sd) columns(statistics) longstub varwidth(24)
}
end
