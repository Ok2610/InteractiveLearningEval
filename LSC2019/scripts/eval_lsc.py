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
GLOBAL_POS = []
GLOBAL_NEG = []
GLOBAL_SUGGS = []

MAX_B = 417
TOTAL_VIDEOS = -1 # Not applicable
ALL_IDS = set([x for x in range(0,41666)])

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


def read_item_properties(filtersFile):
    filters = []
    with open(filtersFile,'r') as f:
        filters = json.load(f)
    n_size = len(filters)
    f_locations = [filters[i]['locationId'] for i in range(n_size)]
    f_hours = [filters[i]['hour'] for i in range(n_size)]
    f_days = [filters[i]['day'] for i in range(n_size)]

    item_metadata = []
    for i in range(n_size):
        item = \
            [0, False, 0, \
                [
                    [f_days[i]],
                    [f_hours[i]],
                ], #std_props
                [
                    [f_locations[i]],
                ], #collection_props
                [] #count_props
            ]
        item_metadata.append(item)

    return item_metadata


def initialize_exquisitor(noms, searchExpansion, numWorkers, segments, modInfoFiles,
                          expansionType, statLevel, modWeights, ffs, guaranteedSlots,
                          collFilters):
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
    item_metadata = collFilters
    video_metadata = []
    exq.initialize(iota, noms, num_workers, segments, num_modalities, b, indx_conf_files, mod_feature_dimensions,
                   func_type, func_objs, item_metadata, video_metadata, expansionType, statLevel, ffs, guaranteedSlots)


def classify_suggestions(suggList, compFiles, cmbVector, relevant, posPolicy, negPolicy, threshold, p, n, rd):
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
        pos = [e[1] for e in GLOBAL_POS]
        neg = [e[1] for e in GLOBAL_NEG]

    exq.reset_model(0)

    return (pos,neg,done)


def run_experiment(resultDir, actorId, actor, runs, rounds, numSuggs, numSegments,
                   numPos, numNeg, measurements, maxB, static_w, no_reset_w):
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
        train_item_count = 0
        GLOBAL_POS = []
        GLOBAL_NEG = []
        GLOBAL_SUGGS = []
        relevant = set()
        train_data = actor['pos'][r] + actor['neg'][r]
        train_labels = [1.0 for x in range(len(actor['pos'][r]))] + [-1.0 for x in range(len(actor['neg'][r]))]
        seen_list = actor['pos'][r] + actor['neg'][r]

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
            # print(sugg_list)
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
            seen_list += actor['neg_r'][r][rd]

            t_classify_start = time()
            (pos,neg,done) = classify_suggestions(sugg_list, actor['relevant'],
                                                  numPos, numNeg, rd+1, actor['neg_r'][r][rd])
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

parser = argparse.ArgumentParser(description="")
parser.add_argument('actors_path', type=str, help=ACTORS_FILE_HELP)
parser.add_argument('result_dir', type=str, help=RESULT_DIR_HELP)
parser.add_argument('result_file', type=str, help=RESULT_FILE_HELP)
parser.add_argument('mod_info_files', type=str, help=INDEX_CONFIG_FILES_HELP)
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


actors = read_actors(args.actors_path, args.number_of_runs)

filters = read_item_properties(args.filters_file)

initialize_exquisitor(args.noms, args.search_expansion_b, args.num_workers, args.num_segments,
                      args.mod_info_files, args.expansion_type,
                      args.stat_level, args.modw_append, args.ffs, args.guaranteed_slots,
                      filters)

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
    metrics[idx] = run_experiment(result_json_dir, idx, actors[a], args.number_of_runs, args.number_of_rounds,
                                  args.num_suggestions, args.num_segments, args.num_pos, args.num_neg,
                                  args.measurements, (args.search_expansion_b == MAX_B), args.static_w, args.no_reset_weights)
    print("%s Actor %d done" % (ts(),idx))


with open(result_json_file, 'w') as f:
    json.dump(metrics,f)