primebsn_att_dict = {"Project":"ISTP>International - Solar-Terrestrial Physics" ,
"Source_name":"PRIME>Probailistic Regressor for Input to the Magnetosphere Estimation" ,
"Discipline":"Space Physics>Magnetospheric Science" ,
"Data_type":"K0>Key Parameter" ,
"Descriptor":"BSN>Bow Shock Nose" ,
"Data_version":"1" ,
"Logical_file_id":"prime_bsn_20150902_v01" ,
"PI_name":"C. O'Brien" ,
"PI_affiliation":"JHUAPL" ,
"TEXT":"https://doi.org/10.5281/ZENODO.8065781" ,
"Instrument_type":"Particles and Magnetic Fields (space)" ,
"Mission_group":"Wind" ,
"Logical_source":"PRIME" ,
"Logical_source_description":"PRIME output data at synthetic bow shock nose location"}

epoch_primebsn_att = {"CATDESC":"Output UTC epoch in UNIX time." ,
"FIELDNAM":"UTC Time" ,
"FILLVAL":-1.0E31 ,
"UNITS":"ms" ,
"VALIDMIN":"least valid value" ,
"VALIDMAX":"greatest valid value" ,
"VAR_TYPE":"support_data"}

bgse_primebsn_att = {"CATDESC":"Interplanetary magnetic field distribution peak in GSE coordinates (X/Y/Z)." ,
"DEPEND_0":"Epoch" ,
"DISPLAY_TYPE":"time_series",
"FIELDNAM":"BGSE" ,
"FILLVAL":-1.0E31 ,
"FORMAT":"%lf" ,
"LABL_PTR_1":"bgse_label",
"UNITS_PTR":"bgse_units" ,
"VALIDMIN":"least valid value" ,
"VALIDMAX":"greatest valid value" ,
"VAR_TYPE":"data"}

bgsesig_primebsn_att = {"CATDESC":"Interplanetary magnetic field distribution width in GSE coordinates (X/Y/Z)." ,
"DEPEND_0":"Epoch" ,
"DISPLAY_TYPE":"time_series",
"FIELDNAM":"BGSE Sigma" ,
"FILLVAL":-1.0E31 ,
"FORMAT":"%lf" ,
"LABL_PTR_1":"bgse_label",
"UNITS_PTR":"bgse_units" ,
"VALIDMIN":"least valid value" ,
"VALIDMAX":"greatest valid value" ,
"VAR_TYPE":"data"}

vgse_primebsn_att = {"CATDESC":"Solar wind velocity distribution peak in GSE coordinates (X/Y/Z)." ,
"DEPEND_0":"Epoch" ,
"DISPLAY_TYPE":"time_series",
"FIELDNAM":"VGSE" ,
"FILLVAL":-1.0E31 ,
"FORMAT":"%lf" ,
"LABL_PTR_1":"vgse_label",
"UNITS_PTR":"vgse_units" ,
"VALIDMIN":"least valid value" ,
"VALIDMAX":"greatest valid value" ,
"VAR_TYPE":"data"}

vgsesig_primebsn_att = {"CATDESC":"Solar wind velocity distribution width in GSE coordinates (X/Y/Z)." ,
"DEPEND_0":"Epoch" ,
"DISPLAY_TYPE":"time_series",
"FIELDNAM":"VGSE Sigma" ,
"FILLVAL":-1.0E31 ,
"FORMAT":"%lf" ,
"LABL_PTR_1":"vgsesig_label",
"UNITS_PTR":"vgsesig_units" ,
"VALIDMIN":"least valid value" ,
"VALIDMAX":"greatest valid value" ,
"VAR_TYPE":"data"}

n_primebsn_att = {"CATDESC":"Solar wind electron density distribution peak." ,
"DEPEND_0":"Epoch" ,
"DISPLAY_TYPE":"time_series",
"FIELDNAM":"N" ,
"FILLVAL":-1.0E31 ,
"FORMAT":"%lf" ,
"LABL_PTR_1":"n_label",
"UNITS_PTR":"n_units" ,
"VALIDMIN":"least valid value" ,
"VALIDMAX":"greatest valid value" ,
"VAR_TYPE":"data"}

nsig_primebsn_att = {"CATDESC":"Solar wind electron density distribution width." ,
"DEPEND_0":"Epoch" ,
"DISPLAY_TYPE":"time_series",
"FIELDNAM":"N Sigma" ,
"FILLVAL":-1.0E31 ,
"FORMAT":"%lf" ,
"LABL_PTR_1":"nsig_label",
"UNITS_PTR":"nsig_units" ,
"VALIDMIN":"least valid value" ,
"VALIDMAX":"greatest valid value" ,
"VAR_TYPE":"data"}

flag_primebsn_att = {"CATDESC":"Fraction of input data that is interpolated quality flag." ,
"DEPEND_0":"Epoch" ,
"DISPLAY_TYPE":"no_plot",
"FIELDNAM":"Flag" ,
"FILLVAL":-1.0E31 ,
"FORMAT":"%i" ,
"UNITS":" " ,
"VALIDMIN":"least valid value" ,
"VALIDMAX":"greatest valid value" ,
"VAR_TYPE":"support_data"}

bgse_primebsn_label = {
"CATDESC":"B GSE label.",
"FIELDNAM":"B GSE label.",
"VAR_TYPE":"metadata"
}

bgse_primebsn_units = {
"CATDESC":"B GSE units.",
"FIELDNAM":"B GSE units.",
"VAR_TYPE":"metadata"
}

bgsesig_primebsn_label = {
"CATDESC":"B GSE Sigma label.",
"FIELDNAM":"B GSE Sigma label.",
"VAR_TYPE":"metadata"
}

bgsesig_primebsn_units = {
"CATDESC":"B GSE Sigma units.",
"FIELDNAM":"B GSE Sigma units.",
"VAR_TYPE":"metadata"
}

vgse_primebsn_label = {
"CATDESC":"V GSE label.",
"FIELDNAM":"V GSE label.",
"VAR_TYPE":"metadata"
}

vgse_primebsn_units = {
"CATDESC":"V GSE units.",
"FIELDNAM":"V GSE units.",
"VAR_TYPE":"metadata"
}

vgsesig_primebsn_label = {
"CATDESC":"V GSE Sigma label.",
"FIELDNAM":"V GSE Sigma label.",
"VAR_TYPE":"metadata"
}

vgsesig_primebsn_units = {
"CATDESC":"V GSE Sigma units.",
"FIELDNAM":"V GSE Sigma units.",
"VAR_TYPE":"metadata"
}

n_primebsn_label = {
"CATDESC":"N label.",
"FIELDNAM":"N label.",
"VAR_TYPE":"metadata"
}

n_primebsn_units = {
"CATDESC":"N units.",
"FIELDNAM":"N units.",
"VAR_TYPE":"metadata"
}

nsig_label = {
"CATDESC":"N Sigma label.",
"FIELDNAM":"N Sigma label.",
"VAR_TYPE":"metadata"
}

nsig_primebsn_units = {
"CATDESC":"N Sigma units.",
"FIELDNAM":"N Sigma units.",
"VAR_TYPE":"metadata"
}