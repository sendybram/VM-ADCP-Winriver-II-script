#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  17 22:11:18 2023

@author: sendy

This script is use for process transect file from reformat result,
process the tide data for plotting, calculate sea current DAV,
save the ploting image, save DAV calulating result and save transect line 
to ESRI shapefile
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from math import degrees, atan
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import glob
from pathlib import Path
import re
import dav
import currentprop
import avg_5m
import warnings
import matplotlib.pylab as pylab

params = {
   'axes.labelsize': 8,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [12, 2.5]
   }
pylab.rcParams.update(params)

#suppress warnings
warnings.filterwarnings('ignore')

def process_tide(line, tide):
    # masking tide data match to line
    # Get t_sol and t_eol
    t_sol = line.date.min()
    t_eol = line.date.max()

    # resampling tide data evey 30 minutes and interpolate
    #tide_res = tide.set_index('Time').resample(rule='30T').mean()
    #tide_res_reindexed = tide_res.reindex(pd.date_range(start=tide_res.index.min(), end=tide_res.index.max(),freq='S'))
    #tidenew = tide_res_reindexed.interpolate(method='linear') 
    #tidenew['Height'] = round(tidenew['Height'],3)

    # parse the date from the 'Time' column
    tide['Time'] = pd.to_datetime(tide['Time'], format='%Y-%m-%d %H:%M:%S')

    # set the 'Time' column as the index
    tide.set_index('Time', inplace=True)

    # resample the DataFrame to every second
    tide_res = tide.resample('1S').asfreq()

    # interpolate missing values
    tidenew = tide_res.interpolate().round(3)
        

    # masking tide data between time of line transect
    mask = (tidenew.index >= t_sol) & (tidenew.index <= t_eol)
    masked_tl = tidenew.loc[mask]
    masked_tl.index.name = 'date'

    # locate center of mask data
    xc = masked_tl.index[(int(len(masked_tl)/2))]
    yc = masked_tl['Height'].iloc[len(masked_tl) // 2]
    
    return t_sol, t_eol, masked_tl, tidenew, xc, yc

def dir_vel_zdist(ens_df):
    ens_df['z'] = ens_df['h'] - ens_df['hb'] # calculate height above seabed
    ens_df['z'] = ens_df['z'].replace(ens_df['z'].max(), ens_df['h'].max()) # replace max height with depth
    ens_df['z'] = round(ens_df['z'],3)

    # calculate velocity and direction from vector
    ens_df['vel'] = np.sqrt((ens_df['u']**2)+(ens_df['v']**2)).round(3)
    ens_df['bearing'] = np.degrees(np.arctan(ens_df['u']/ens_df['v'])).round(3)
    ens_df.loc[(ens_df['u']>0) & (ens_df['v']>0), 'dir'] = ens_df['bearing']
    ens_df.loc[((ens_df['u']>0) & (ens_df['v']<0)) | ((ens_df['u']<0) & (ens_df['v']<0)), 'dir'] = (ens_df['bearing']+180) %360
    ens_df.loc[(ens_df['u']<0) & (ens_df['v']>0), 'dir'] = (ens_df['bearing']+360) %360
    ens_df = ens_df.drop(['bearing'], axis=1)
    return ens_df

def vel_cal(df):
    df = df.groupby(['ens'], group_keys=True).apply(dir_vel_zdist)
    return df

def process_line(line, masked_tl):
    # Process transect file for plotting
    # Tide correction
    line = pd.merge_asof(line, masked_tl, on='date', direction = 'backward')
    line['hn'] = line['h'] - line['Height']
    line['hbn'] = line['hb'] - line['Height']
    
    # Recompute velocity, bearing and add Z distance
    line = vel_cal(line)
    
    # remove duplicate data for recompute distance calculation
    df2 = line.drop_duplicates(subset = 'ens')
    df2.reset_index(inplace=True, drop=True)

    # define spatial geometry
    gdf = gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2.lon, df2.lat),
                           crs = 'EPSG:4326').to_crs('EPSG:32750')
    lx = LineString(gdf['geometry'].to_list())
    
    # calculate line bearing orientation need to reveresed or not
    xx =  lx.coords
    p1 = xx[0]
    p2 = xx[-1]

    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    angle = degrees(atan(dx/dy))

    if (dx>0 and dy>0):
        plot_brg = angle
    elif ((dx>0 and dy<0) or (dx<0 and dy<0)):
        plot_brg = (angle+180) %360
    else:
        plot_brg = (angle+360) %360
    
    

    # recalculate distance by projected coordinate
    gdf['dist_prev'] = 0
    gdf['dist_tot'] = 0
    for i in gdf.index[:-1]:
        gdf.loc[i+1, 'dist_prev'] = gdf.loc[i, 'geometry'].distance(gdf.loc[i+1, 'geometry'])
        
    for j in gdf.index[:-1]:
        gdf.loc[j+1, 'dist_tot'] = gdf.loc[j, 'dist_tot'] + gdf.loc[j+1, 'dist_prev']

    # round distance
    gdf.dist_tot = round(gdf.dist_tot,3)

    #replace with new distance
    line['dist'] = line['dist'].map(dict(zip(gdf.dist, gdf.dist_tot)))
    
    # create pivot table
    arr = line.pivot_table(index='hb', columns='dist', values='vel')
    
    # save df
    out= Path('Orde_1 - CP_PRODUCT')
    file_out = re.sub(r'(?i).txt', 'test.csv', fo)
    
    if not out.exists():
        out.mkdir(parents=True)
        
    line.to_csv(out / file_out, encoding='utf-8')
    return line, gdf, arr, plot_brg

def save_shp(file, geodata):
    # save data to SHP
    # define folder for SHP results
    outshp= Path('Orde_1 - CP_PRODUCT/Transect ESRI Shapefile')
    shp_out = re.sub(r'(?i).txt', '.shp', fo)
    
    if not outshp.exists():
        outshp.mkdir(parents=True)
    
    # create line geometry
    line_geom = LineString(geodata['geometry'].to_list())
    line_name = Path(file).stem
    line_num = re.findall(r'\d+', line_name)
    trans_line = gpd.GeoDataFrame({'geometry':[line_geom], 'name':["T_" + "".join(line_num)]}, 
                                  crs='EPSG:32750').to_crs('EPSG:4326')
    trans_line.to_file(outshp / shp_out, encoding='utf-8')
    return print(f'* Finished save to shapefile {shp_out}')

def label_orient(line_heading):
    # SETTING PLOT'S LABEL BASED ON LINE ORIENTATION
    # define label line orientation. plot_brg is line heading / label EOL
    label_dict = {
        (337.5 < line_heading <= 22.5): ('S', 'N'),
        (22.5 < line_heading <= 67.5): ('SW', 'NE'),
        (67.5 < line_heading <= 122.5): ('W', 'E'),
        (122.5 < line_heading <= 157.5): ('NW', 'SE'),
        (157.5 < line_heading <= 202.5): ('N', 'S'),
        (202.5 < line_heading <= 247.5): ('NE', 'SW'),
        (247.5 < line_heading <= 292.5): ('E', 'W'),
        (292.5 < line_heading <= 337.5): ('SE', 'NW'),
    }
    lbl_start, lbl_end = label_dict[True]
    return lbl_start, lbl_end

def transect_proc(file_in, ftide):
    # process line transect
    print(f'Process file {fo}, Please Wait!')
    # Read transect file and tide file
    tide = pd.read_csv(ftide, parse_dates=['Time'], na_values=['*****'])

    line = pd.read_csv(file_in, sep='\t', skiprows=33, parse_dates=['date'])

    # Process tide file
    t_sol, t_eol, masked_tl, tidenew, xc, yc = process_tide(line, tide)
    
    # Process transect file
    df, gdf, arr, plot_brg = process_line(line, masked_tl)

    # Process averaging velocity every 5m depth
    av = avg_5m.cal_avg_speed(line)

    # get midle time of tide every transect
    #sum_dat = {'name':[f'T_{fo[9:11]}'], 'x':[xc], 'y':[yc]}
    #summary = pd.DataFrame(sum_dat)
    #summary.to_csv('overview_tide.txt', mode='a', index=False, header=False)

    # clean direction
    dfc = currentprop.clean_dir(df)
    
    # Calculate shear velocity
    shear_raw = currentprop.shear(dfc, fo)

    # process to DAV calculation
    dav_dat = dav.proc_dav(shear_raw, fo)
   
    # Plot DAV feather plot
    dav.davplot(dav_dat, fo)
    
    # save dav to shp, average value every 75 ensemble for visualize purpose
    dav.dav_to_shp(dav_dat, 75, fo)
    
    # calculate turbulence intensity
    turb = currentprop.turbulence(shear_raw, plot_brg, fo)
                
    # save to shp
    save_shp(file_in, gdf)
    
    # define label orientation
    lbl_start, lbl_end = label_orient(plot_brg)
    
    # plot turbulence intensity
    currentprop.turb_plot(turb, gdf, fo, lbl_start, lbl_end)
    
    """ Plotting"""
    # define image folder
    outpath = Path('Orde_1 - CP_PRODUCT/Transect_Image_Profile')
    file_out = re.sub(r'(?i).txt', '_avg5m.jpg', fo)
    
    if not outpath.exists():
        outpath.mkdir(parents=True)
    
    # define image name
    linename = Path(file_in).stem
    pattern = r'Transect_(\w+)'
    match = re.search(pattern, linename)
    lineid = match.group(1)
    plt_name = f'Transect {lineid}'
    
    # data plotting
    fig, ax = plt.subplots(ncols=2, 
                           gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.15})
    fig.suptitle(plt_name + " - (" + t_sol.strftime('%d %B %Y') + ")")

    # TRANSECT PLOT ver.1 avg 5m
    # plot velocity magnitude and invert Y-axis
    vel = av['vel']
    u = av['u5']
    v = av['v5']

    norm = colors.Normalize(vmin=0.1, vmax=2.0)
    Q = ax[0].quiver(av['dist'], av['ens_h'], u, v, vel, width=0.0008, cmap='jet', norm=norm, scale=1 / 0.005)
    ax[0].quiverkey(Q, 0.15, 0.91, 5, r'East 5 m/s', labelpos='E',coordinates='figure')
    ax[0].invert_yaxis()

    # plot MSL & seabed profile 
    ax[0].axhline(y=0, color='k', lw=1, linestyle='--')
    ax[0].plot(gdf.dist_tot, gdf.Height*-1,'r')
    
    zmax = av.h.max() +5
    ax[0].plot(av.dist, av.h,'k')
    ax[0].fill_between(av.dist, av.h, zmax, color='slategrey')

    # setting label
    bbox = dict(facecolor='none', edgecolor='black')
    ax[0].annotate(lbl_start, xy=(0, 0), xycoords='axes fraction',xytext=(-0.01, -0.25), bbox = bbox)
    ax[0].annotate(lbl_end, xy=(0, 0), xycoords='axes fraction',xytext=(0.99, -0.25),bbox = bbox)
    ax[0].tick_params(axis="x", direction='in', length=8)
    ax[0].set_ylim(zmax, -5.0)
    ax[0].set_xlim(-50, 11500)
    ax[0].set_xlabel('Distance (m)')
    ax[0].set_ylabel('Depth (m)')

    # define colorbar properties
    axins = inset_axes(ax[0], width="80%", height="5%",loc='lower center', borderpad=-5)

    # plot colorbar
    fig.colorbar(Q, cax=axins, orientation="horizontal", label='Velocity m/s')

    # Tide Plot
    # plot tide height
    ax[1].plot(tidenew.index, tidenew.Height, color='deepskyblue')
    ax[1].plot(masked_tl.index, masked_tl.Height, color='red')
    ax[1].plot(xc, yc, marker='o', markersize=5, color='red',label='Survey Time')
    ax[1].axhline(y=0, color='g', lw=0.75)

    ax[1].legend(loc='lower right')

    for label in ax[1].get_xticklabels():
        label.set_rotation(40)
        label.set_horizontalalignment('right')
    
    ax[1].set_xlabel('Time (UTC)')
    ax[1].set_ylabel('Height (m)')
    ax[1].grid(lw=0.3)

    # Save image
    fig.savefig(outpath / file_out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> {file_out} image saved")

    return print(f'Process file {fo}, DONE! \n')

if __name__ == "__main__":

    transect_file = glob.glob("Orde_1 - CP_PRODUCT/Transect_file/*.txt")
    tide_file = glob.glob("Orde_1 - CP_PRODUCT/tide_*.csv")
        
    if not tide_file:
        print('No Tide File.')
        exit()
    else:
        for ftd in tide_file:
            print("Detected Tide File: ", "\n".join(tide_file))
        
    if not transect_file:
        print('No files to convert.')
        exit()
    else:
        print("Detected ASCII Transect files: \n", "\n".join(transect_file))
        for file_in in transect_file:
            fo = Path(file_in).name
            transect_proc(file_in, ftd)