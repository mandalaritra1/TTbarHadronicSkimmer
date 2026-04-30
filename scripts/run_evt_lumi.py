#!/usr/bin/env python3
from __future__ import print_function
import coffea.util as cutil
import csv
import pandas as pd 
from itertools import zip_longest

# Directory containing Coffea files
directory = "outputs/dy/"

# List of input files and corresponding labels
files = {
    "ZPrime2000": directory + "ZPrime2000_10_2018.coffea",
    "ZPrime4000": directory + "ZPrime4000_10_2018.coffea",
    "JetHT_2018A": directory + "JetHT_2018A.coffea",
    "JetHT_2018B": directory + "JetHT_2018B.coffea",
    "JetHT_2018C": directory + "JetHT_2018C.coffea",
    "JetHT_2018D": directory + "JetHT_2018D.coffea",
}
cutflow =False 
#with open('data2018.csv', 'w', newline= '') as f :
for label, filename in files.items():
    print("\n=== Processing %s ===" % label)

    output = cutil.load(filename)
    print("Available keys in the output:", output['cutflow'].keys())

    if "event_list" in output:
     
        event_list = output["event_list"]
        run=event_list["run"]
        lumi=event_list["lumi"]
        evt=event_list["event"]
        '''   
        df = pd.DataFrame({
        'run':  pd.Series(ev['run']),
        'lumi': pd.Series(ev['lumi']),
        'evt':  pd.Series(ev['event'])
        })

        '''

        with open('events'+str(label)+'.csv', 'w', newline= '') as f :
          w = csv.writer(f)
          w.writerow(['run', 'lumi', 'evt'])
          for r, l, e in zip_longest(run, lumi, evt, fillvalue=''):
             w.writerow([r, l, e])

        print("run:lumi:evt --> ", len(run) , ":", len(lumi) , ":" , len(evt), "event_list is ", len(event_list) )

    if cutflow and  "cutflow" in output:
        cutflow = output["cutflow"]
        total_events=cutflow["all events"]
        print("Cutflow entries:")
        print("total events:", total_events)
        for step, count in cutflow.items():
            #print("  %s: %s" % (step, count))
            print("  %s: %.2f" % (step, count * 59740.0/total_events))
            print("  %s: %.2f" % (step, count), "no scaling")
    else:
        print("No cutflow table found in this file.")

