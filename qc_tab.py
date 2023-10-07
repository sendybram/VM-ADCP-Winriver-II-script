#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:42:57 2023

@author: sendy
"""

import pandas as pd
import glob
from pathlib import Path
import re

def qctabulasi(file_in, file_out):
    dat = pd.read_csv(file_in, sep='\t', skiprows=33)
    dat2 = dat.drop_duplicates(subset = 'ens')
    dat2.reset_index(inplace=True, drop=True)
    
    dum = list()
    for ens, ens_df in dat2:
        ens_df.loc[:,'name'] = Path(fo).stem
        ens_df.loc[:,'r_max'] = dat2["roll"].max()
        ens_df.loc[:,'r_min'] = dat2["roll"].min()
        ens_df.loc[:,'r_mean'] = ens_df["r_max"] - ens_df["r_min"]
        ens_df.loc[:,'r_range'] = dat2["roll"].max()
        ens_df.loc[:,'r_stdev'] = dat2["roll"].std()
        
        
        dum.append(ens_df)
    tabul = pd.concat(dum, ignore_index=True)
    tabul.to_csv(file_out, index=False)
    
    return tabul

if __name__ == "__main__":
    outpath = Path("Orde_1 - CP_PRODUCT")
    if not outpath.exists():
        outpath.mkdir(parents=True)

    transect_file = glob.glob("Orde_1 - CP_PRODUCT/Transect_file/*.txt")
        
    if not transect_file:
        print('No files to convert.')
        exit()
    else:
        print("Detected ASCII Transect files: \n", "\n".join(transect_file))
        for file_in in transect_file:
            fo = Path(file_in).name
            file_out = re.sub(r'(?i)_ASC.TXT', '.txt', fo)
            qctabulasi(file_in, file_out)