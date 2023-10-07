#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 08 22:29:24 2023
Add command to extract Pitch and Roll Value

@author: sendy

modified from dopler.py

original file from https://github.com/esmoreido/dopler
"""

from pathlib import Path
from datetime import datetime as dt
import glob
import re
from pandas import DataFrame, concat
from numpy import nan
from itertools import islice

def get_chunk(file, n):
    return [x.strip() for x in islice(file, n)]

def head_proc(lines):
    avg = [float(lines[1].split()[8]),
           float(lines[1].split()[9]),
           float(lines[1].split()[10]),
           float(lines[1].split()[11]),
           ]   
    avg = DataFrame({'avg': avg}).replace([0, -32768], nan)
    avg = avg.mean().avg
    
    res = {'ens': int(lines[0].split()[7]),
           'date': dt.strftime(dt(year=2000 + int(lines[0].split()[0]), 
                                  month=int(lines[0].split()[1]),
                                  day=int(lines[0].split()[2]), 
                                  hour=int(lines[0].split()[3]),
                                  minute=int(lines[0].split()[4]), 
                                  second=int(lines[0].split()[5])),
                               format('%d.%m.%Y %H:%M:%S')),
           'h': round(avg, 3),
           'b': round(float(lines[2].split()[4]), 2),
           'lat': float(lines[3].split()[0]) if float(lines[3].split()[0]) != 30000. else nan,
           'lon': float(lines[3].split()[1]) if float(lines[3].split()[1]) != 30000. else nan,
           'nbins': int(lines[5].split()[0]),
           'roll': float(lines[0].split()[10]),
           'pitch': float(lines[0].split()[9]),
           }
   
    return res

def ens_proc(ens, date, ensnum, ensdist, ensh, enslat, enslon, ensroll, enspitch):
    df = DataFrame([x.split() for x in ens], 
                   columns=['hb', 'vel', 'dir', 'u', 'v', 'w', 'errv', 'bs1', 'bs2', 'bs3',
                            'bs4', 'percgood', 'q'], dtype='float')
    
    df['bs'] = df[['bs1', 'bs2', 'bs3', 'bs4']].mean(axis=1).round(3)  # calculate the average scatter
    df = df.replace([-32768, 2147483647, 255], nan)
    df.drop(['percgood', 'q', 'bs1', 'bs2', 'bs3', 'bs4'], inplace=True, axis=1)  # remove unnecessary scatter columns
    df['date'] = date
    df['ens'] = ensnum  # add the ensemble number
    df['dist'] = ensdist  # add distance from the edge
    df['h'] = ensh  # total depth
    df['lat'] = enslat  # latitude
    df['lon'] = enslon  # longitude
    df['roll'] = ensroll
    df['pitch'] = enspitch
    df[['vel', 'u', 'v', 'w', 'errv']] = (df[['vel', 'u', 'v', 'w', 'errv']] * 0.01).round(3)
    df = df.dropna()  # remove missing
    res = df[['date', 'ens', 'dist', 'lat', 'lon', 'roll', 'pitch', 'h', 'hb', 'u', 'v', 'w', 'errv', 'vel', 'dir', 'bs']]
    return res

def file_proc(path_in, path_out, mt):
    with open(path_in, "r") as fn:
        print(f'Process file {fo}, Please Wait!')
        fn.readline()  # skip three empty lines
        fn.readline()
        fn.readline()
        df = DataFrame()  # array for data
        # read the first piece of service information - always 6 lines
        head = get_chunk(fn, 6)
        while head:
            opr = head_proc(head)
            chunk = get_chunk(fn, opr['nbins'])
            ens = ens_proc(chunk, opr['date'], opr['ens'], opr['b'], opr['h'], opr['lat'], opr['lon'], opr['roll'], opr['pitch'])
            df = concat([df, ens], ignore_index=True)
            head = get_chunk(fn, 6)
        df = df.dropna()
        # save output
        df.to_csv(outpath / path_out, sep='\t', index=False, na_rep='-32768')
        print(f'Finished for {path_out}')
        
        # add metadata
        with open(mt) as fp:
            data = fp.read()
        with open(outpath / path_out) as fp:
            data2 = fp.read()
        
        data += data2
  
        with open (outpath / path_out, 'w') as fp:
            fp.write(data)
            print(f'Finished add Metadata to {path_out}\n')
    return

if __name__ == "__main__":
    outpath = Path('Orde_1 - CP_PRODUCT/Transect_file')
    if not outpath.exists():
        outpath.mkdir(parents=True)


    mtdt = glob.glob("Orde_1 - CP_PRODUCT/metadata.txt")
    ff = glob.glob("Orde_0 - CP_RAW/PLAYBACK DATA/*_ASC.TXT")
    
    if not ff:
        print('No files to convert.')
        exit()
    else:
        print("Detected ASCII *_ASC.TXT files: \n", "\n".join(ff))
        for f in ff:
            if not mtdt:
                print(f'Warning: metadata.txt not found for {f}, store it in Orde_1 - CP_PRODUCT folder!')
            else:
                mt = mtdt[0]
                file_in = f
                fo = Path(file_in).name
                file_out = re.sub(r'(?i)_ASC.TXT', '.txt', fo)
                file_proc(file_in, file_out, mt)
            



