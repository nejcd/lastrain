#!/usr/bin/env python
"""
/***************************************************************************
 

                              -------------------
        begin                : 2016-11-12
        git sha              : $Format:%H$
        copyright            : (C) 2016 by Nejc Dougan
        email                : nejc.dougan@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import numpy as np
import os, glob

def get_list_of_npy(directory):
    os.chdir(directory)
    return glob.glob("*.npy")

path = '/media/nejc/Prostor/AI/data/kelag_32_MSS/'

files = get_list_of_npy(path)
#merged = np.empty((1,2))
#test = np.empty((1,2))
merged = []
number_of_files = len(files)
for file in files:
    if number_of_files == len(files):
        data = np.load(file)
    elif number_of_files != len(files):
        data = np.append(data, np.load(file), axis=0)
    number_of_files -= 1
    print ('{0} files to go'.format(number_of_files))

print('Saving {0}'.format(path + 'merged.npy'))    
np.save(path + 'merged' + '.npy', data)
print('Finished!')
