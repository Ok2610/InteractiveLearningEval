import json

vidinfo = {}
with open('vidinfo_20.json','r') as f:
    vidinfo = json.load(f)

exqmap = {}
with open('exqmap.json','r') as f:
    exqmap = json.load(f)

judgements = []
with open('../avs.qrels.main.tv21','r') as f:
    judgements = [s.strip().split(' ') for s in f.readlines()]

# judgements[.][0]: query
# judgements[.][1]: junk
# judgements[.][2]: shotID
# judgements[.][3]: sampling-stratum(1(top),2,3)
# judgements[.][4]: judgements (-1 not judged, 1 pos, 0 neg)

topics = {}
actor_items = []

act = 0
for i in range(len(judgements)):
    if judgements[i][0] not in topics:
        topics[judgements[i][0]] = act
        actor_items.append({})
        act += 1
        actor_items[topics[judgements[i][0]]]['pos_items'] = []
        # as -1 items were submitted but not judged adding them as negative could be bad
        actor_items[topics[judgements[i][0]]]['ignore_items'] = []
    shot = judgements[i][2]
    vid = shot.split('_')[0][4:]
    shotId = int(shot.split('_')[1])
    if int(judgements[i][4]) == 1:
       for item in exqmap[str(vidinfo[vid]['shots'][shotId-1]['exqId'])]:
            if item not in actor_items[topics[judgements[i][0]]]:
                actor_items[topics[judgements[i][0]]]['pos_items'].append(item)
    if int(judgements[i][4]) == -1:
       for item in exqmap[str(vidinfo[vid]['shots'][shotId-1]['exqId'])]:
            if item not in actor_items[topics[judgements[i][0]]]:
                actor_items[topics[judgements[i][0]]]['ignore_items'].append(item)

with open('actor_items.json', 'w') as f:
    json.dump(actor_items,f)
