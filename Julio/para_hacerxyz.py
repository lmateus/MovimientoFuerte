# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:32:50 2020

@author: JULIO CESAR
"""

import pandas as pd

#df = pd.read_csv('dem_all_z18.txt', sep=';', decimal=',', header=0, names=['FID', 'pointid', 'grid_code', 'X', 'Y'])
df = pd.read_csv('UTM_Z18_30m.txt', sep=';', decimal=',', header=0)
#print(df.astype(float))
'''
pd.to_numeric(df['FID']) # Convertir de texto a float la columna
pd.to_numeric(df['pointid'])
pd.to_numeric(df['grid_code'])
pd.to_numeric(df['X'])
pd.to_numeric(df['Y'])
print(df)
print(type(df['X'][2]))

print('PROBANDO QUE SE HAYAN CONVERTIDO LAS COLUMNAS EN ENTEROS')
print((df['X'][2]) + df['Y'][2])
'''
df_yx = df.sort_values(['Y', 'X'], ascending=[True, True])
print(df_yx)

'''

print(df_yx)
print(min(df['Y']), max(df['Y']), min(df['X']), max(df['X']))
z_yx = pd.DataFrame()
z_yx = df_yx['grid_code']
#z_yx['grid_code'] = df_yx['grid_code']
print(z_yx)
z_yx.to_csv("ojala_funcione.dat", decimal='.', header=False, index=False)
'''