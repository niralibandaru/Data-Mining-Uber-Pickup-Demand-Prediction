"""
UBER PICKUP DEMAND PREDICTION WITH LSTM

"""

import csv
import pandas as pd

#function to read the csv's and append it to the combined csv dataset file
def csv_combine(read_data):
    for row in read_data:
        writer.writerow(row)

#list of datasets available

data_sets = ['uber-raw-data-apr14_mod.csv', 'uber-raw-data-may14_mod.csv', 'uber-raw-data-jun14_mod.csv',
             'uber-raw-data-jul14_mod.csv', 'uber-raw-data-aug14_mod.csv', 'uber-raw-data-sep14_mod.csv']


f = open("combined_data.csv", "w")
writer = csv.writer(f, lineterminator='\n')

for reader in range(0, 6):
    reader = csv.reader(open(data_sets[reader]))
    csv_combine(reader)

f.close()
