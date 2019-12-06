import csv

mylist = [1,2,3,4,5]

with open('filename', 'wb') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(mylist)