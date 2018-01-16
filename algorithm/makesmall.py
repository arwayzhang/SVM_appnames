import csv

dataFile = open('pcamean1000_datawithname.csv', 'r')
dataInfo = csv.reader(dataFile)


n1=3000
n2=4000

# write csv files

with open('smalltestdata.csv', 'w',newline='') as combineFile:
    abcsv = csv.writer(combineFile, dialect='excel')

    k=0

    for i in dataInfo:
        
        if k >= n1 and k <=n2:
        #if k in index:
            abcsv.writerow(i)

        k += 1

