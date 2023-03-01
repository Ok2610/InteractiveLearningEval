import exq as exquisitor
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

COLLECTION_SIZE = 1007360
MAX_B = 10074
TOTAL_VIDEOS = 7475

IV = []

#######################
# Functions
#######################
def ts():
    return "[" + str(strftime("%d %b, %H:%M:%S")) + "]"


def read_actors(actorsFile):
    actors = {}
    with open(actorsFile, 'r') as f:
        actors = json.load(f)
    return actors


def initialize_exquisitor(noms, searchExpansion, numWorkers, segments, modInfoFiles,
                          expansionType, statLevel, modWeights, learningRate):
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
            pow(2, c['decomp_mask_t'])-1,
            float(pow(2, c['multiplier_t'])),
            pow(2, c['decomp_mask_ir'])-1,
            pow(2, c['decomp_mask_ir'])-1,
            float(pow(2, c['multiplier_ir'])),
            mod_weights[m]
        ])
        # [5, 48, 16, 16, pow(2, 32)-1, float(pow(2, 32)), pow(2, 16)-1, pow(2, 16)-1, pow(2, 16)],
        # [7, 54, 10, 10, pow(2, 32)-1, float(pow(2, 32)), pow(2, 10)-1, pow(2, 10)-1, pow(2, 10)],
        # [8, 55, 9, 9, pow(2, 32)-1, float(pow(2, 32)), pow(2, 9)-1, pow(2, 9)-1, pow(2, 9)]
    item_metadata = []
    video_metadata = []
    exquisitor.initialize(iota, noms, num_workers, segments, num_modalities, b, indx_conf_files, mod_feature_dimensions,
                   func_type, func_objs, item_metadata, video_metadata, expansionType, statLevel, learningRate)


def classify_suggestions(suggList, relevant, p, n, rd):
    # Process the suggestions
    global GLOBAL_POS, GLOBAL_NEG, GLOBAL_SUGGS
    pos = 0
    neg = 0
    rel = []

    for i in suggList:
        if i in relevant:
            rel.append(i)
            if pos != p:
                GLOBAL_POS.append(i)
                pos += 1
        else:
            GLOBAL_NEG.append(i)
            neg += 1

    exquisitor.reset_model(False)

    return (GLOBAL_POS, GLOBAL_NEG, rel)


def run_experiment(resultDir, actorId, actor, runs, rounds, numSuggs, numSegments,
                   numPos, numNeg, measurements, maxB, static_w):
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
        exquisitor.reset_model(True)
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
            faces_filter = []
            categories_filter = []
            tags_filter = []
            train_times = [0.0 for x in range(3)]
            t_start = time()

            #print("Training")
            #print(train_data, train_labels)
            #print(filters)
            if train:
                if actual_run:
                    train_times = exquisitor.train(train_data, train_labels, False, [], False)
                else:
                    train_times = exquisitor.train(train_data, train_labels, False, [], True)
            #print(train_times)
            #print("Getting suggestions")
            (sugg_list, total, worker_time, sugg_time, sugg_overhead) = exquisitor.suggest(numSuggs, numSegments, seen_list, False, [])
            #print(sugg_list, total, worker_time, sugg_time, sugg_overhead)
            #print(sugg_list)
            #print("Got suggestions")

            t_stop = time()
            t = t_stop - t_start

            if measurements:
                if rd == 0:
                    exquisitor.log(actorId,r,rd,1)
                else:
                    exquisitor.log(actorId,r,rd,0)

            suggs = set(sugg_list)
            seen_set = set(seen_list)
            seen_set |= suggs
            seen_list = list(seen_set)

            t_classify_start = time()
            (pos,neg,rel) = classify_suggestions(sugg_list, actor['relevant'], #actor['maybe'],
                                                  numPos, numNeg, rd+1)
            t_classify_stop = time()
            #print("Time to classify: %f" % (t_classify_stop - t_classify_start))
            #print(pos, neg, done)

            if actual_run:
                metrics[r]['p'].append(float(len(rel))/float(len(sugg_list)))

                relevant |= set(rel)
                rec = float(len(relevant))/len(actor['relevant'])
                metrics[r]['r'].append(rec)

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

            #if len(sugg_list) == 0: #and maxB:
            #    print('%s Actor %d run %d can not advance further!' % (ts(), actorId, r))
            #    break
            #elif len(sugg_list) == 0:
            #    train = False
            #else:
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

    #pn_file = ('a%d_PN.json') % actorId
    #pn_path = os.path.join(resultDir, pn_file)
    #with open(pn_path, 'w') as f:
    #    json.dump(pn,f)

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

    #if selectionPolicy == 1:
    #    m_file = ('a%d_metrics.json') % actorId
    #    m_path = os.path.join(resultDir, m_file)
    #    with open(m_path, 'w') as f:
    #        json.dump(metrics,f)
    #    return {}

    return metrics


#######################
# Main
#######################
ACTORS_FILE_HELP = "JSON file containing LSC actors."
RESULT_FILE_HELP = "File where all the metrics of the experiment will be stored (JSON)."
RESULT_DIR_HELP = "Directory where all the metrics of the experiment will be stored."
INIT_FEAT_FILE_HELP = "HDF5 File containing top feature information (Ratio-i64)"
FEAT_IDS_FILE_HELP = "HDF5 File containing other feature ids information (Ratio-i64)"
RATIOS_FILE_HELP = "HDF5 File containing feature ratio information (Ratio-i64)"
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
PREDICT_HELP = "Set this option if exquisitor is compiled with CCOVERAGE_PROB or PREDICT_COUNT"
CLUSTER_OPT_HELP = "Any cluster containing items below the set id will be marked and only they will be used during selection."
ITEM_FILTER_HELP = "Sets a filter on the set id. Anything below the id passes."
ACTORS_APPEND_HELP = "Which actors to run."

parser = argparse.ArgumentParser(description="")
parser.add_argument('actors_file', type=str, help=ACTORS_FILE_HELP)
parser.add_argument('result_dir', type=str, help=RESULT_DIR_HELP)
parser.add_argument('result_file', type=str, help=RESULT_FILE_HELP)
parser.add_argument('mod_info_files', type=str, help=INDEX_CONFIG_FILES_HELP)
parser.add_argument('--measurements', action='store_true', default=False)
parser.add_argument('--noms', type=int, default=1000)
parser.add_argument('--num_suggestions', type=int, default=25, help=NUMBER_OF_SUGGESTIONS_HELP)
parser.add_argument('--num_workers', type=int, default=1, help=NUMBER_OF_WORKERS_HELP)
parser.add_argument('--num_segments', type=int, default=16, help=NUMBER_OF_SEGMENTS_HELP)
parser.add_argument('--num_features', type=int, default=1000, help=NUMBER_OF_FEATURES_HELP)
parser.add_argument('--num_pos', type=int, default=5, help=NUMBER_OF_POSITIVES_HELP)
parser.add_argument('--num_neg', type=int, default=15, help=NUMBER_OF_NEGATIVES_HELP)
parser.add_argument('--search_expansion_b', type=int, default=256, help=SEARCH_EXPANSION_FIXED_HELP)
parser.add_argument('--expansion_type', type=int, default=0, help='ExpansionType: 0=CNT, 1=GRC, 2=FRC, 3=ERC, 4=ARC')
parser.add_argument('--stat_level', type=int, default=1, help='ECP Statistics level. Default = 1')
parser.add_argument('--number_of_runs', type=int, default=50, help=NUMBER_OF_RUNS_HELP)
parser.add_argument('--number_of_rounds', type=int, default=10, help=NUMBER_OF_ROUNDS_HELP)
parser.add_argument('--cluster_opt', type=int, default=0, help=CLUSTER_OPT_HELP)
parser.add_argument('--actors_append', action='append', type=int, default=[], help=ACTORS_APPEND_HELP)
parser.add_argument('--modw_append', action='append', type=float, default=[], help='Weights for each modality. Default 1.0.')
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--static_w', action='store_true', default=False, help='Re-run active run with learned weights')

args = parser.parse_args()

result_json_file = os.path.join(args.result_dir, args.result_file, args.result_file + '.json')
if os.path.isfile(result_json_file):
    print("RESULT FILE ALREADY EXISTS!")
    exit(0)

result_json_dir = os.path.join(args.result_dir, args.result_file)
if not(os.path.isdir(result_json_dir)):
    os.mkdir(result_json_dir)


actors = read_actors(args.actors_file)

initialize_exquisitor(args.noms, args.search_expansion_b, args.num_workers, args.num_segments,
                      args.mod_info_files, args.expansion_type,
                      args.stat_level, args.modw_append, args.learning_rate)
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
                                  args.measurements, (args.search_expansion_b == MAX_B), args.static_w)
    print("%s Actor %d done" % (ts(),idx))


with open(result_json_file, 'w') as f:
    json.dump(metrics,f)
