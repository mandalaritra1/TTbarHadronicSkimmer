#!/usr/bin/env python3
from __future__ import print_function
import coffea.util as cutil

# Directory containing Coffea files
directory = "outputs/dy/"

# List of input files and corresponding labels
files = {
    "ZPrime 1000 GeV": directory + "ZPrime1000_10_2018.coffea",



}

for label, filename in files.items():
    print("\n=== Processing %s ===" % label)

    # Load the Coffea file
    output = cutil.load(filename)

    # Optionally print all keys available in the output (for debugging)
    print("Available keys in the output:", output['cutflow'].keys())

    # Check if the 'cutflow' table exists
    if "cutflow" in output:
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

