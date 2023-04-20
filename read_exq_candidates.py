# Reads the logged candidate file of an ExqWorker
# Converts the input into a json array and outputs a json file

import argparse
import json
import numpy as np
from ast import literal_eval

parser = argparse.ArgumentParser()
parser.add_argument('candidates_file', type=str)
parser.add_argument('output_file', type=str)
parser.add_argument('--ids_only', action='store_true', default=False)

args = parser.parse_args();

print("Reading and converting log file")
with open(args.candidates_file, 'r') as f:
    noms = [s.strip() for s in f.readlines()]
    nomslit = [literal_eval(n) for n in noms]
    if args.ids_only:
        out = []
        for n in nomslit:
            out.append([])
            for nn in n:
                out[-1].append(nn[0])
        with open(args.output_file, 'w') as ff:
            json.dump(out, ff)
    else:
        with open(args.output_file, 'w') as ff:
            json.dump(nomslit, ff)
print("Done!")