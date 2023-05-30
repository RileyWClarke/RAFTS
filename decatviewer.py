import os 
import pandas as pd

dataframe = pd.read_csv('ddf_flares.csv', index_col=0)

cosmos = dataframe[dataframe['field'] == 'COSMOS']

cosmos.sort_values(by='object magnitude', inplace=True)

start = input("Enter starting index: ")

for i,id in enumerate(cosmos['candidate id'].unique()[int(start):]):

    os.system("open https://decat-webap.lbl.gov/decatview.py/cand/{}?rbtype=undefined".format(id))
    print('Showing candidate {0}/{1}'.format(i+1+int(start), len(cosmos['candidate id'].unique())))
    prompt = input("Press c to continue: ")

    if prompt == 'c':
        continue

    else: 
        break
        

