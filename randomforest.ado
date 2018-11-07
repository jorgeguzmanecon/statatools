capture program drop randomforest

program define randomforest , rclass
  syntax varlist , [n_estimators(integer 20)] gen(string) [logit] [train_index(varname)] [predict_index(varname)] [roconly] [store_roc(string)]

{

    if "`logit'" != "" {
        local logitparam --logit
    }

    if "`train_index'" != "" {
        local train_index_param --train_data `train_index'
    }
    if "`predict_index'" != "" {
        local predict_index_param --predict_data `predict_index'
    }
    if "`roconly'" == "roconly" {
        local roconly_param --roconly
    }

    if "`store_roc'" != "" {
        local store_roc_param --store_roc `store_roc'
    }
    
    
    //store data in a tempfile and call python
    //tempfile forestfile
    local forestfile  "~/temp/forestfile.dta "
    qui: save  `forestfile' , replace
    
    //Call python
    !python ~/ado/randomforest/stata_randomforest_only.py `roconly_param' -n `n_estimators' -g `gen' `logitparam' `train_index_param' `predict_index_param' `store_roc_param' `forestfile' `varlist' 

    local rocfile = trim(subinstr("`forestfile'",".dta","",.))
    local rocfile  "`rocfile'_rocscores.dta"

    use `rocfile'
    foreach stat in roc_full roc_prediction roc_train {
        qui:sum `stat'
        return scalar `stat'=`r(mean)'
    }
    
    //load the new data
    use `forestfile', replace
    
}
end
