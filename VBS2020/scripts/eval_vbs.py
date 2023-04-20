import exq
import os
import json
import numpy as np
import h5py
import argparse
import bisect
from math import sqrt
from time import time, sleep, strftime
from random import seed, random, randint, sample
from scipy.spatial import distance

#######################
# Global variables
#######################
### VBS 2020
DECOMP_MASK = [pow(2, 10) - 1, pow(2, 14) - 1, pow(2, 16) - 1, pow(2, 20) - 1, pow(2, 30) - 1]
MULTIPLIER = [np.uint64(1000), np.uint64(10000),
              np.uint64(1000000), np.uint64(1000000000)]

MAX_N_INT = np.uint64(15)

BIT_SHIFT_RATIO = [np.uint64(4), np.uint64(14), np.uint64(24),
                   np.uint64(34), np.uint64(44), np.uint64(54)]

FEAT_PER_INT_RATIO = 6
MULTIPLIER_RATIO = np.uint64(1000)
PRECISION_RATIO = 3

BIT_SHIFT_INIT_RATIO = np.uint64(54)
MULTIPLIER_INIT = np.uint64(pow(10, 16))
MASK_INIT = np.uint64(18014398509481983)

CATEGORIES = {}
TAGS = {}
FACES = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

GLOBAL_POS = []
GLOBAL_NEG = []
GLOBAL_SUGGS = []
ALL_IDS = set([x for x in range(0,1082657)])


###

GLOBAL_POS = []
GLOBAL_NEG = []
GLOBAL_SUGGS = []

MAX_B = 10827
TOTAL_VIDEOS = 7475
ALL_IDS = set([x for x in range(0,1082657)])

#######################
# Functions
#######################
def ts():
    return "[" + str(strftime("%d %b, %H:%M:%S")) + "]"


# HDF5 Helper Functions
def decompress(compInit, compIds, compFeat):
    decomp = []

    comp_init = np.uint64(compInit)
    comp_ids = np.uint64(compIds)
    comp_feat = np.uint64(compFeat)

    feat_init_id = comp_init >> BIT_SHIFT_INIT_RATIO
    feat_init_score = (comp_init & MASK_INIT) / float(MULTIPLIER_INIT)
    feat_score = feat_init_score
    decomp.append((feat_init_id,feat_init_score))
    for f_pos in range(comp_ids & MAX_N_INT):
        feat_i = (comp_ids >> BIT_SHIFT_RATIO[f_pos]) & np.uint64(DECOMP_MASK[0])
        feat_score *= ((comp_feat >> BIT_SHIFT_RATIO[f_pos]) & np.uint64(DECOMP_MASK[0])) / float(MULTIPLIER_RATIO)
        decomp.append((feat_i, feat_score))
    return decomp


def read_single_item_features(idx, initFile, idsFile, ratiosFile):
    with h5py.File(initFile, 'r') as init:
        f_init = init['data'][idx]
        with h5py.File(idsFile, 'r') as ids:
            f_ids = ids['data'][idx]
            with h5py.File(ratiosFile, 'r') as ratios:
                f_ratios = ratios['data'][idx]
                return decompress(f_init, f_ids, f_ratios)


def read_multiple_item_features(items, initFile, idsFile, ratiosFile):
    items = []
    with h5py.File(initFile, 'r') as init:
        with h5py.File(idsFile, 'r') as ids:
            with h5py.File(ratiosFile, 'r') as ratios:
                items = [decompress(init['data'][i], ids['data'][i], ratios['data'][i]) for i in items]
    return items

# Euclidean distance between two vectors
def calculate_distance_cmb_vector(suggVector, cmbVector):
    dist = 0.0
    ids_avg = {}
    for key in cmbVector:
        ids_avg[int(key)] = cmbVector[key]

    for (idx,val) in suggVector:
        if idx in ids_avg:
            dist += (val - ids_avg[idx]) * (val - ids_avg[idx])
            ids_avg[idx] = 0.0
        else:
            dist += val * val

    for idx in ids_avg:
        dist += ids_avg[idx] * ids_avg[idx]

    return dist


def read_actors(actorsFile):
    actors = {}
    with open(actorsFile, 'r') as f:
        actors = json.load(f)
    return actors


def read_item_properties(filtersFile, vidInfoFile):
    filters = []
    with open(filtersFile,'r') as f:
        filters = json.load(f)
    n_size = len(filters)

    with open(vidInfoFile,'r') as f:
        vidInfo = json.load(f)

    item_metadata = []
    for i in range(n_size):
        item = \
            [0, True, filters[i]['vidId'], \
                [
                ], #std_props
                [
                    [filters[i]['faces']],
                ], #collection_props
                [] #count_props
            ]
        item_metadata.append(item)
    
    f_categories = []
    f_tags = []
    for i in range(TOTAL_VIDEOS):
        vid_id = '{:05d}'.format((i+1))
        f_categories.append(filters[vidInfo[vid_id]['shots'][0]['exqId']]['catIds'])
        f_tags.append(filters[vidInfo[vid_id]['shots'][0]['exqId']]['tagIds'])

    vid_metadata = [[]]
    for i in range(TOTAL_VIDEOS):
        vid = [
            f_categories[i],
            f_tags[i]
        ]
        vid_metadata[0].append(vid)

    return item_metadata, vid_metadata

def initialize_exquisitor(noms, searchExpansion, numWorkers, segments, modInfoFiles,
                          expansionType, statLevel, modWeights, ffs, guaranteedSlots,
                          item_meta, vid_meta):
    mod_info = []
    with open(modInfoFiles,'r') as f:
        mod_info = json.load(f)

    iota = 1
    noms = noms
    num_workers = numWorkers
    segments = segments
    indx_conf_files = [c['indx_path'] for c in mod_info]
    num_modalities = len(mod_info)
    mod_weights = []
    if len(modWeights) != 0:
        mod_weights = modWeights
    else:
        for m in range(num_modalities):
            mod_weights.append(1.0)
    b = searchExpansion
    mod_feature_dimensions = [c['total_feats'] for c in mod_info]
    func_type = 0
    func_objs = []
    for m,c in enumerate(mod_info):
        func_objs.append([
            c['n_feat_int']+1,
            c['bit_shift_t'],
            c['bit_shift_ir'],
            c['bit_shift_ir'],
            c['decomp_mask_t'],
            float(c['multiplier_t']),
            pow(2, c['decomp_mask_ir'])-1,
            pow(2, c['decomp_mask_ir'])-1,
            float(c['multiplier_ir']),
            mod_weights[m]
        ])
    item_metadata = item_meta
    video_metadata = vid_meta
    exq.initialize(iota, noms, num_workers, segments, num_modalities, b, indx_conf_files, mod_feature_dimensions,
                   func_type, func_objs, item_metadata, video_metadata, expansionType, statLevel, ffs, guaranteedSlots)


def classify_suggestions(suggList, cmbVector, relevant, posPolicy, negPolicy, p, n, rd, compFiles):
    # Process the suggestions
    global GLOBAL_POS, GLOBAL_NEG, GLOBAL_SUGGS
    done = False
    pos = []
    neg = []
    distances = []
    for s in suggList:
        if s in relevant:
            done = True
        features = read_single_item_features(s, compFiles[0], compFiles[1], compFiles[2])
        distances.append((calculate_distance_cmb_vector(features, cmbVector),s))

    distances = sorted(distances)
    # print(distances)

    if posPolicy == 0: #Acc-add
        GLOBAL_POS += distances[:p]

    if negPolicy == 0: #Acc-add
        GLOBAL_NEG += distances[-n:]

    temp = sorted(GLOBAL_POS + GLOBAL_NEG + distances)
    if posPolicy == 1: #Acc-replace
        if rd == 1:
            GLOBAL_POS = distances[:p]
        else:
            GLOBAL_POS = [e for e in temp[:(p*rd)]]

    if negPolicy == 1: #Acc-replace
        if rd == 1:
            GLOBAL_NEG = distances[-n:]
        else:
            GLOBAL_NEG = [e for e in temp[-(n*rd):]]

    if posPolicy == 2: #Fixed
        for (d,e) in distances:
            if len(GLOBAL_POS) < p:
                GLOBAL_POS.append((d,e))
                GLOBAL_POS = sorted(GLOBAL_POS)
            elif d < GLOBAL_POS[-1][0]:
                GLOBAL_POS[-1] = (d,e)
                GLOBAL_POS = sorted(GLOBAL_POS)
            else:
                break

    if negPolicy == 2: #Fixed
        for (d,e) in distances[::-1]:
            if len(GLOBAL_NEG) < n:
                GLOBAL_NEG.append((d,e))
                GLOBAL_NEG = sorted(GLOBAL_NEG)
            elif d > GLOBAL_NEG[0][0]:
                GLOBAL_NEG[0] = (d,e)
                GLOBAL_NEG = sorted(GLOBAL_NEG)
            else:
                break

    if negPolicy == 3: #Rand-local
        if rd == 1:
            GLOBAL_NEG = distances[-n:]
        if posPolicy == 0:
            try:
                GLOBAL_NEG += sample(distances[p:],n)
            except:
                print('Failed to get negative samples. Round: %d Len(suggs): %d' % (rd,len(distances)))
        elif posPolicy == 1:
            try:
                GLOBAL_NEG += sample(temp[(p*rd):],n)
            except:
                print('Failed to get negative samples. Round: %d Len(temp): %d' % (rd,len(temp)))
        elif posPolicy == 2:
            try:
                GLOBAL_NEG += sample((set(distances)-set(GLOBAL_POS)),n)
            except:
                print('Failed to get negative samples. Round: %d Len(suggs): %d' % (rd,len(distances)))

    if negPolicy == 4: #Rand-global
        pos = [e[1] for e in GLOBAL_POS]
        GLOBAL_NEG += sample((ALL_IDS - set(pos) - set(GLOBAL_NEG)),n)
        neg = [e for e in GLOBAL_NEG]


    if negPolicy != 4:
        # print(p,n,GLOBAL_POS, GLOBAL_NEG)
        pos = [e[1] for e in GLOBAL_POS]
        neg = [e[1] for e in GLOBAL_NEG]

    exq.reset_model(False, False)

    return (pos,neg,done)


def run_experiment(resultDir, actorId, actor, runs, rounds, numSuggs, numSegments,
                   numPos, numNeg, measurements, maxB, static_w, no_reset_w, compFiles):
    global GLOBAL_POS, GLOBAL_NEG, GLOBAL_SUGGS
    metrics = {}
    metrics['p'] = 0.0
    metrics['r'] = 0.0
    metrics['t'] = 0.0
    pn = {}
    for r in range(runs):
        metrics[r] = {}
        metrics[r]['p'] = []
        metrics[r]['r'] = []
        metrics[r]['t'] = []
        metrics[r]['time-train'] = []
        metrics[r]['time-b'] = []
        metrics[r]['time-train-pipe'] = []
        metrics[r]['time-suggest'] = []
        metrics[r]['time-suggest-overhead'] = []
        metrics[r]['suggs'] = []
        metrics[r]['segments'] = {}
        for s in range(numSegments):
            metrics[r]['segments'][s] = {}
            metrics[r]['segments'][s]['time-score'] = []
            metrics[r]['segments'][s]['total-scored'] = []
        pn[r] = {}
        pn[r]['pos'] = []
        pn[r]['neg'] = []

    seed(1)
    for r in range(runs):
        actual_run = True
        if static_w:
            actual_run = False
        if not no_reset_w:
            exq.reset_model(True,True)
        train = True
        rd = 0
        start_time = 0
        current_session_time = 0
        session_end_time = rounds
        train_data = []
        train_labels = []
        GLOBAL_POS = []
        GLOBAL_NEG = []
        GLOBAL_SUGGS = []
        train_data = actor['start'][str(r)]
        train_labels = [1.0 for x in range(5)] + [-1.0 for x in range(5)]
        seen_list = []

        while(current_session_time < session_end_time):
            train_times = [0.0 for x in range(3)]
            t_start = time()

            # print("Training")
            # print(train_data, train_labels)
            if train:
                if actual_run:
                    train_times = exq.train(train_data, train_labels, False, [], False)
                else:
                    train_times = exq.train(train_data, train_labels, False, [], True)
            # print(train_times)
            # print("Getting suggestions")
            (sugg_list, total, worker_time, sugg_time, sugg_overhead) = exq.suggest(numSuggs, numSegments, seen_list, False, [])
            # print(sugg_list, total, worker_time, sugg_time, sugg_overhead)
            print(sugg_list)
            # print("Got suggestions")

            t_stop = time()
            t = t_stop - t_start

            if measurements:
                if rd == 0:
                    exq.log(actorId,r,rd,1)
                else:
                    exq.log(actorId,r,rd,0)

            suggs = set(sugg_list)
            seen_set = set(seen_list)
            seen_set |= suggs
            seen_list = list(seen_set)

            t_classify_start = time()
            # AccRep
            posPolicy = 1
            negPolicy = 1
            (pos,neg,done) = classify_suggestions(sugg_list, actor['relVector'], actor['relevant'],
                                                  posPolicy, negPolicy, numPos, numNeg, rd+1, compFiles)
            t_classify_stop = time()
            # print("Time to classify: %f" % (t_classify_stop - t_classify_start))
            # print(pos, neg, done)

            if actual_run:
                if len(sugg_list) == 0:
                    metrics[r]['p'].append(0.0)
                else:
                    metrics[r]['p'].append(float(len(pos))/len(sugg_list))

                if done:
                    metrics[r]['r'].append(1.0)
                else:
                    metrics[r]['r'].append(0.0)

                metrics[r]['t'].append(t)
                metrics[r]['time-train'].append(train_times[0])
                metrics[r]['time-b'].append(train_times[1])
                metrics[r]['time-train-pipe'].append(train_times[2])
                metrics[r]['time-suggest'].append(sugg_time)
                metrics[r]['time-suggest-overhead'].append(sugg_overhead)
                metrics[r]['suggs'].append(sugg_list)
                pn[r]['pos'].append(pos)
                pn[r]['neg'].append(neg)
                for s in range(numSegments):
                    metrics[r]['segments'][s]['time-score'].append(worker_time[s])
                    metrics[r]['segments'][s]['total-scored'].append(total[s])


            if len(sugg_list) == 0:
                print('%s Actor %d run %d can not advance further!' % (ts(), actorId, r))
                break
            else:
                train = True
                train_data = pos + neg
                train_labels = [1.0 for x in range(len(pos))] + [-1.0 for x in range(len(neg))]
                train_item_count = len(pos) + len(neg)

            rd += 1
            if static_w and not actual_run and session_end_time == rd:
                rd = 0
                actual_run = True
                GLOBAL_POS = []
                GLOBAL_NEG = []
                GLOBAL_SUGGS = []
                train_data = actor['pos'][r] + actor['neg'][r]
                train_labels = [1.0 for x in range(len(actor['pos'][r]))] + [-1.0 for x in range(len(actor['neg'][r]))]
                seen_list = actor['pos'][r] + actor['neg'][r]

            current_session_time = rd
        print("%s Actor %d run %d done after %d rounds." % (ts(), actorId, r, rd))

    # pn_file = ('a%d_PN.json') % actorId
    # pn_path = os.path.join(resultDir, pn_file)
    # with open(pn_path, 'w') as f:
    #     json.dump(pn,f)


    p_sum_r = 0.0
    t_sum_r = 0.0
    for r in range(runs):
        rds = len(metrics[r]['p'])
        p_sum_rds = 0.0
        t_sum_rds = 0.0
        for rd in range(rds):
            p_sum_rds += metrics[r]['p'][rd]
            metrics['r'] += metrics[r]['r'][rd]
            t_sum_rds += metrics[r]['t'][rd]
        p_sum_r += p_sum_rds/rds
        t_sum_r += t_sum_rds/rds
    metrics['p'] = p_sum_r/runs
    metrics['r'] /= runs
    metrics['t'] = t_sum_r/runs

    return metrics


#######################
# Main
#######################
ACTORS_FILE_HELP = "JSON path containing actor files."
RESULT_FILE_HELP = "File where all the metrics of the experiment will be stored (JSON)."
RESULT_DIR_HELP = "Directory where all the metrics of the experiment will be stored."
INDEX_CONFIG_FILES_HELP = "JSON File containing information about modalities and index file path location."
NUMBER_OF_SUGGESTIONS_HELP = "Set number of suggestions to get per round. Default is 25."
NUMBER_OF_WORKERS_HELP = "Set number of workers to use. Default is 1."
NUMBER_OF_SEGMENTS_HELP = "Set number of segments to use. Default is 16."
NUMBER_OF_FEATURES_HELP = "Set number of features in modality. Default is the 1K imagenet classes."
NUMBER_OF_POSITIVES_HELP = "Number of positives to select. Works with selection policy 2, 3 and 4. Default is 12"
NUMBER_OF_NEGATIVES_HELP = "Number of negatives to select. Works with selection policy 2 and 4. Default is 13"
SEARCH_EXPANSION_FIXED_HELP = "The number of clusters selected from training, also known as b. Default is 64."
FILTER_COUNT_HELP = "Set this option if expansion should be based on LSC active filters."
NUMBER_OF_RUNS_HELP = "Set number of times the experiment runs. Default is 50."
NUMBER_OF_ROUNDS_HELP = "Set number of interaction rounds in each run. Default is 10."
ACTORS_APPEND_HELP = "Which actors to run."
FILTERS_FILE_HELP = "JSON File containing metadata for all items."
VIDINFO_FILE_HELP = "JSON File containing metadata for all videos."

parser = argparse.ArgumentParser(description="")
parser.add_argument('actors_path', type=str, help=ACTORS_FILE_HELP)
parser.add_argument('result_dir', type=str, help=RESULT_DIR_HELP)
parser.add_argument('result_file', type=str, help=RESULT_FILE_HELP)
parser.add_argument('mod_info_files', type=str, help=INDEX_CONFIG_FILES_HELP)
parser.add_argument('h5_init_feat_file', type=str)
parser.add_argument('h5_feat_ids_file', type=str)
parser.add_argument('h5_ratios_file', type=str)
parser.add_argument('filters_file', type=str, help=FILTERS_FILE_HELP)
parser.add_argument('vidinfo_file', type=str, help=VIDINFO_FILE_HELP)
parser.add_argument('categories_file', type=str)
parser.add_argument('tags_file', type=str)
parser.add_argument('--measurements', action='store_true', default=False)
parser.add_argument('--noms', type=int, default=1000)
parser.add_argument('--num_suggestions', type=int, default=25, help=NUMBER_OF_SUGGESTIONS_HELP)
parser.add_argument('--num_workers', type=int, default=1, help=NUMBER_OF_WORKERS_HELP)
parser.add_argument('--num_segments', type=int, default=16, help=NUMBER_OF_SEGMENTS_HELP)
parser.add_argument('--num_features', type=int, default=1000, help=NUMBER_OF_FEATURES_HELP)
parser.add_argument('--num_pos', type=int, default=-1, help=NUMBER_OF_POSITIVES_HELP)
parser.add_argument('--num_neg', type=int, default=-1, help=NUMBER_OF_NEGATIVES_HELP)
parser.add_argument('--search_expansion_b', type=int, default=256, help=SEARCH_EXPANSION_FIXED_HELP)
parser.add_argument('--expansion_type', type=int, default=0, help='ExpansionType: 0=CNT, 1=GRC, 2=FRC, 3=ERC, 4=ARC')
parser.add_argument('--stat_level', type=int, default=1, help='ECP Statistics level. Default = 1')
parser.add_argument('--number_of_runs', type=int, default=50, help=NUMBER_OF_RUNS_HELP)
parser.add_argument('--number_of_rounds', type=int, default=10, help=NUMBER_OF_ROUNDS_HELP)
parser.add_argument('--actors_append', action='append', type=int, default=[], help=ACTORS_APPEND_HELP)
parser.add_argument('--modw_append', action='append', type=float, default=[], help='Weights for each modality. Default 1.0.')
parser.add_argument('--ffs', action='store_true', default=False, help='Run with FFS')
parser.add_argument('--guaranteed_slots', type=int, default=0, help='Number of guaranteed slots in FFS')
parser.add_argument('--static_w', action='store_true', default=False, help='Re-run active run with learned weights')
parser.add_argument('--no_reset_weights', action='store_true', default=False, help='Do not reset modality weights after first run')

args = parser.parse_args()

result_json_file = os.path.join(args.result_dir, args.result_file, args.result_file + '.json')
if os.path.isfile(result_json_file):
    print("RESULT FILE ALREADY EXISTS!")
    exit(0)

result_json_dir = os.path.join(args.result_dir, args.result_file)
if not(os.path.isdir(result_json_dir)):
    os.mkdir(result_json_dir)


actors = read_actors(args.actors_path)

item_metadata, vid_metadata = read_item_properties(args.filters_file, args.vidinfo_file)

comp_files = []
comp_files.append(args.h5_init_feat_file)
comp_files.append(args.h5_feat_ids_file)
comp_files.append(args.h5_ratios_file)
with open(args.categories_file, 'r') as f:
    cats = json.load(f)
    for idx,s in enumerate(cats):
        CATEGORIES[s.lower()] = idx

with open(args.tags_file, 'r') as f:
    tags = json.load(f)
    for idx,s in enumerate(tags):
        TAGS[s.lower()] = idx

initialize_exquisitor(args.noms, args.search_expansion_b, args.num_workers, args.num_segments,
                      args.mod_info_files, args.expansion_type,
                      args.stat_level, args.modw_append, args.ffs, args.guaranteed_slots,
                      item_metadata, vid_metadata)

print("%s Initialized!" % ts())

a_to_run = set()
if (len(args.actors_append) == 0):
    a_to_run = set([x for x in range(0,len(actors))])
else:
    a_to_run = set(args.actors_append)

metrics = {}
for idx, a in enumerate(actors):
    if idx not in a_to_run:
        print('Skipping actor %d' % idx)
        continue
    metrics[idx] = run_experiment(result_json_dir, idx, a, args.number_of_runs, args.number_of_rounds,
                                  args.num_suggestions, args.num_segments, args.num_pos, args.num_neg,
                                  args.measurements, (args.search_expansion_b == MAX_B), args.static_w, args.no_reset_weights,
                                  comp_files)
    print("%s Actor %d done" % (ts(),idx))


with open(result_json_file, 'w') as f:
    json.dump(metrics,f)
