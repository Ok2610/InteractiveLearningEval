import json
import random
import argparse

TOTAL_ITEMS = 58066
N_RUNS = 50
N_POS = 3
N_NEG = 5

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=N_RUNS)
parser.add_argument('--n_pos', type=int, default=N_POS)
parser.add_argument('--n_neg', type=int, default=N_NEG)

args = parser.parse_args()

with open('../files/actor_items.json','r') as f:
    actor_items = json.load(f)


actors = {}

for i in range(len(actor_items)):
    actors[i] = {}
    actors[i]['pos'] = []
    actors[i]['neg'] = []
    actors[i]['relevant'] = actor_items[i]['pos_items']
    #actors[i]['maybe'] = actor_items[i]['ignore_items']
    for r in range(args.runs):
        actors[i]['pos'].append(random.sample(actor_items[i]['pos_items'],args.n_pos))
        actors[i]['neg'].append([])
        for k in range(args.n_neg):
            while(1):
                x = random.randint(0,TOTAL_ITEMS)
                if x not in actor_items[i]['pos_items']: #and x not in actor_items[i]['ignore_items']:
                    actors[i]['neg'][r].append(x)
                    break

out_f = '../files/actors_' + 'p' + str(args.n_pos) + 'n' + str(args.n_neg) + 'r' + str(args.runs) + '.json'
with open(out_f,'w') as f:
    json.dump(actors,f)

