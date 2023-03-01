import json
import csv
import random

vidinfo = []
# [exqId, clsId, fname, cls]
with open('../files/comm_cls_info.json','r') as f:
    vidinfo = json.load(f)

cats = []
# cats[category] = [classes]
with open('../files/categories.json','r') as f:
    cats = json.load(f)

class_to_items = {}
# get classes and corresponding item array
for c in vidinfo:
    class_to_items[c[1]] = []

for c in vidinfo:
    class_to_items[c[1]].append(c[0])

with open('../files/class_to_items.json','w') as f:
    json.dump(class_to_items,f)

class_to_idx = {}
for c in vidinfo:
    class_to_idx[c[3]] = c[1]

actor_items = []
for c in cats:
    rnd = random.sample(cats[c], 3)
    for r in rnd:
        actor_items.append({})
        actor_items[-1]['class'] = r
        actor_items[-1]['pos_items'] = class_to_items[class_to_idx[r]]

with open('../files/actor_items.json','w') as f:
    json.dump(actor_items,f)
