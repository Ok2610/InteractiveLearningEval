import exqjournal as exquisitor
import os
import json
import numpy as np
import h5py
import argparse
import bisect
from math import sqrt
from time import time, clock, sleep, strftime
from random import seed, random, randint, sample
from scipy.spatial import distance

#######################
# Global variables
#######################
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

DAYS = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

LOCATIONS = {}

GLOBAL_POS = []
GLOBAL_NEG = []
GLOBAL_SUGGS = []
ALL_IDS = set([x for x in range(0,41666)])

MAX_B = 417

IV = []

#######################
# Functions
#######################
def ts():
    return "[" + str(strftime("%d %b, %H:%M:%S")) + "]"


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


def read_total_items_count(initFile):
    size = 0
    with h5py.File(initFile, 'r') as init:
        size = len(init['data'])
    return size

def read_single_item_features_normalized(idx, initFile, idsFile, ratiosFile):
    with h5py.File(initFile, 'r') as init:
        f_init = init['data'][idx]
        with h5py.File(idsFile, 'r') as ids:
            f_ids = ids['data'][idx]
            with h5py.File(ratiosFile, 'r') as ratios:
                f_ratios = ratios['data'][idx]
                features = decompress(f_init, f_ids, f_ratios)
                l2norm = 0.0
                for (idx,val) in features:
                    l2norm += pow(val,2)
                try:
                    l2norm = 1.0/sqrt(l2norm)
                except ZeroDivisionError:
                    l2norm = 1.0
                for idx,(f_idx,val) in enumerate(features):
                    features[idx] = (f_idx, val*l2norm)
                return features


def read_single_item_features(idx, initFile, idsFile, ratiosFile):
    with h5py.File(initFile, 'r') as init:
        f_init = init['data'][idx]
        with h5py.File(idsFile, 'r') as ids:
            f_ids = ids['data'][idx]
            with h5py.File(ratiosFile, 'r') as ratios:
                f_ratios = ratios['data'][idx]
                return decompress(f_init, f_ids, f_ratios)


def read_many_item_features(items, compFiles):
    items = []
    with h5py.File(compFiles[0], 'r') as init:
        with h5py.File(compFiles[1], 'r') as ids:
            with h5py.File(compFiles[2], 'r') as ratios:
                items = [decompress(init['data'][i], ids['data'][i], ratios['data'][i]) for i in items]
    return items

def calculate_distance_items(suggVector, relVector):
    dist = 0.0
    feature_vector = [0.0 for x in range(1000)]
    for (idx,val) in relVector:
        feature_vector[idx] += val

    for (idx,val) in suggVector:
        if idx in feature_vector:
            dist += (val - feature_vector[idx]) * (val - feature_vector[idx])
            feature_vector[idx] = 0.0
        else:
            dist += val * val

    for (idx,val) in relVector:
        dist += feature_vector[idx] * feature_vector[idx]

    return dist

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


def calculate_distance_cmb_vector_normalized(suggVector, cmbVector):
    dist = 0.0
    ids_avg = {}
    for key in cmbVector:
        ids_avg[int(key)] = cmbVector[key]

    l2norm = 0.0
    for idx in ids_avg:
        l2norm += pow(ids_avg[idx],2)
    try:
        l2norm = 1.0/sqrt(l2norm)
    except ZeroDivisionError:
        l2norm = 1.0
    for idx in ids_avg:
        ids_avg[idx] *= l2norm

    for (idx,val) in suggVector:
        if idx in ids_avg:
            dist += (val - ids_avg[idx]) * (val - ids_avg[idx])
            ids_avg[idx] = 0.0
        else:
            dist += val * val

    for idx in ids_avg:
        dist += ids_avg[idx] * ids_avg[idx]

    return dist

def calculate_distance_mahal_vec(suggVector, relevantVector, iv):
    dist = 0.0
    sugg_vector = np.array([0.0 for x in range(1000)]) #Make range a parameter
    rel_vector = np.array([0.0 for x in range(1000)])
    for (idx,val) in suggVector:
        sugg_vector[idx] = val
    for (idx,val) in relevantVector:
        rel_vector[idx] = val
    dist = distance.mahalanobis(sugg_vector, rel_vector, iv)
    return dist


def calculate_distance_mahal_cmb(suggVector, cmbVector, iv):
    dist = 0.0
    avg_vector = np.array([0.0 for x in range(1000)])
    for key in cmbVector:
        avg_vector[int(key)] = cmbVector[key]
    sugg_vector = np.array([0.0 for x in range(1000)]) #Make range a parameter
    for (idx,val) in suggVector:
        sugg_vector[idx] = val
    dist = distance.mahalanobis(sugg_vector, avg_vector, iv)
    return dist


def read_actors(actorsFile):
    actors = {}
    with open(actorsFile, 'r') as f:
        actors = json.load(f)
    return actors


def initialize_exquisitor(noms, searchExpansion, numWorkers, segments, indexConfigFile, compFiles,
                          modality, numFeat, filtersFile, clusterOpt, itemFilter, expansion_type):
    filters = []
    with open(filtersFile,'r') as f:
        filters = json.load(f)

    mods = ['vis','txt','mm']
    #Arguments
    iota = 1
    n_noms = noms
    n_workers = numWorkers
    n_mod = mods.index(modality)
    n_feat_mod_vis = numFeat
    n_feat_mod_txt = 0
    vis_ids = compFiles[1]
    vis_ratios = compFiles[2]
    vis_init = compFiles[0]
    txt_ids = ''
    txt_ratios = ''
    txt_init = ''
    indx_conf_txt = ''
    indx_conf_vis = indexConfigFile
    b = searchExpansion
    cluster_opt = clusterOpt
    item_filter = itemFilter

    n_size = len(filters)
    f_locations = [filters[i]['locationId'] for i in range(n_size)]
    f_hours = [filters[i]['hour'] for i in range(n_size)]
    f_days = [filters[i]['day'] for i in range(n_size)]

    exquisitor.initialize(iota, n_noms, n_workers, n_mod, n_feat_mod_vis, n_feat_mod_txt, vis_ids, vis_ratios,
                          vis_init, txt_ids, txt_ratios, txt_init, indx_conf_txt, indx_conf_vis, b, expansion_type, cluster_opt,
                          item_filter, f_locations, f_hours, f_days)

def classify_suggestions(suggList, compFiles, cmbVector, relevant, posPolicy, negPolicy, threshold, p, n, rd,
                         distanceCalculation):
    # Process the suggestions
    global GLOBAL_POS, GLOBAL_NEG, GLOBAL_SUGGS
    done = False
    pos = []
    neg = []
    distances = []
    if distanceCalculation == 2:
        dist_map = {}
        for s in suggList:
            dist_map[s] = 1000.0

        for s in suggList:
            sugg_vec = read_single_item_features(s, compFiles[0], compFiles[1], compFiles[2])
            for r in relevant:
                rel_vec = read_single_item_features(s, compFiles[0], compFiles[1], compFiles[2])
                dist = calculate_distance_items(sugg_vec, rel_vec)
                if dist_map[s] > dist:
                    dist_map[s] = dist

        for s in dist_map:
            distances.append((dist_map[s],s))
    else:
        for s in suggList:
            if s in relevant:
                done = True
            features = read_single_item_features(s, compFiles[0], compFiles[1], compFiles[2])
            if distanceCalculation == 0:
                distances.append((calculate_distance_cmb_vector(features, cmbVector),s))
            elif distanceCalculation == 1:
                distances.append((calculate_distance_mahal_avg(features, cmbVector, IV),s))

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

    exquisitor.reset_model(0)

    return (pos,neg,done)


def run_experiment(resultDir, actorId, actor, runs, roundsOrSessionTime, ttt, timed, numSuggs, numSegments,
                   compFiles, posSelPolicy, negSelPolicy, threshold, numPos, numNeg, spike, filterScenario, measurements,
                   expansionType, expansionThreshold, maxB, distanceCalculation):
    global GLOBAL_POS, GLOBAL_NEG, GLOBAL_SUGGS
    total_collection_size = read_total_items_count(compFiles[0])
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
        metrics[r]['done'] = []
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
        seen_list = []
        exquisitor.reset_model(1)
        train = True
        rd = 0
        start_time = 0
        current_session_time = 0
        session_end_time = roundsOrSessionTime
        train_data = []
        train_labels = []
        train_item_count = 0
        GLOBAL_POS = []
        GLOBAL_NEG = []
        GLOBAL_SUGGS = []
        if spike == 0:
            #Random spike
            #train_data = [int(random() * total_collection_size) for x in range(10)]
            train_data = actor['start'][str(r)]
            train_labels = [1.0 for x in range(5)] + [-1.0 for x in range(5)]
            train_item_count = 10
        elif spike == 1:
            #Spike based on the similar items to the average vector
            train_data = actor['spikeItems']
            train_labels = [1.0 for x in range(len(actor['spikeItems']))]
            train_item_count = len(actor['spikeItems'])
        elif spike == 2:
            #Spike based on relevant items
            train_data = actor['relevant']
            train_labels = [1.0 for x in range(len(actor['relevant']))]
            train_item_count = len(actor['relevant'])

        if timed:
            current_session_time = time()
            start_time = time()
            session_end_time = time() + roundsOrSessionTime
        while(current_session_time < session_end_time):
            locations_filter = []
            hours_filter = []
            days_filter = []
            if filterScenario == 0:
                #No filters
                locations_filter = []
                hours_filter = []
                days_filter = []
            elif filterScenario == 1 and timed:
                #Apply filters based on time spent
                for f in actor['time_filters']:
                    if (current_session_time - start_time) >= float(f):
                        locations_filter = list(set([LOCATIONS[location.lower()]
                                                for location in actor['time_filters'][f]['locations']])|
                                                set(locations_filter))
                        hours_filter = list(set(actor['time_filters'][f]['hours'])|
                                            set(hours_filter))
                        days_filter = list(set([DAYS[day] for day in actor['time_filters'][f]['days']])|
                                           set(days_filter))
            elif filterScenario == 1 and not timed:
                #Apply filters based on rounds
                for idx,f in enumerate(actor['time_filters']):
                    if idx < rd:
                        locations_filter = list(set([LOCATIONS[location.lower()]
                                                for location in actor['time_filters'][f]['locations']])|
                                                set(locations_filter))
                        hours_filter = list(set(actor['time_filters'][f]['hours'])|
                                            set(hours_filter))
                        days_filter = list(set([DAYS[day] for day in actor['time_filters'][f]['days']])|
                                           set(days_filter))
            elif filterScenario == 2 and timed:
                #Apply user filters based on time spent
                for f in actor['advanced_filters']:
                    if (current_session_time - start_time) >= float(f):
                        locations_filter = list(set([LOCATIONS[location.lower()]
                                                for location in actor['advanced_filters'][f]['locations']])|
                                                set(locations_filter))
                        hours_filter = list(set(actor['advanced_filters'][f]['hours']) | set(hours_filter))
                        days_filter = list(set([DAYS[day] for day in actor['advanced_filters'][f]['days']])|
                                           set(days_filter))
            elif filterScenario == 2 and not timed:
                #Apply user filters based on rounds
                for idx,f in enumerate(actor['advanced_filters']):
                    if idx < rd:
                        locations_filter = list(set([LOCATIONS[location.lower()]
                                                for location in actor['advanced_filters'][f]['locations']])|
                                                set(locations_filter))
                        hours_filter = list(set(actor['advanced_filters'][f]['hours']) | set(hours_filter))
                        days_filter = list(set([DAYS[day] for day in actor['advanced_filters'][f]['days']])|
                                           set(days_filter))
            elif filterScenario == 3 and timed:
                #Apply lifelogger filters based on time spent
                for f in actor['advanced_filters']:
                    if (current_session_time - start_time) >= float(f):
                        locations_filter = list(set([LOCATIONS[location.lower()]
                                                for location in actor['advanced_filters'][f]['locations']]) | 
                                                set([LOCATIONS[location.lower()]
                                                for location in actor['advanced_filters'][f]['adv_l']]) |
                                                set(locations_filter))
                        hours_filter = list(set(actor['advanced_filters'][f]['hours']) |
                                            set(actor['advanced_filters'][f]['adv_h']) |
                                            set(hours_filter))
                        days_filter = list(set([DAYS[day] for day in actor['advanced_filters'][f]['days']]) | 
                                           set([DAYS[day] for day in actor['advanced_filters'][f]['adv_d']])|
                                           set(days_filter))
            elif filterScenario == 3 and not timed:
                #Apply lifelogger filters based on rounds
                for idx,f in enumerate(actor['advanced_filters']):
                    if idx < rd:
                        locations_filter = list(set([LOCATIONS[location.lower()]
                                                for location in actor['advanced_filters'][f]['locations']]) | 
                                                set([LOCATIONS[location.lower()]
                                                for location in actor['advanced_filters'][f]['adv_l']]) |
                                                set(locations_filter))
                        hours_filter = list(set(actor['advanced_filters'][f]['hours']) |
                                            set(actor['advanced_filters'][f]['adv_h']) |
                                            set(hours_filter))
                        days_filter = list(set([DAYS[day] for day in actor['advanced_filters'][f]['days']]) | 
                                           set([DAYS[day] for day in actor['advanced_filters'][f]['adv_d']])|
                                           set(days_filter))
            elif filterScenario == 4:
                #Apply metadata filters
                locations_filter = [LOCATIONS[location.lower()] for location in actor['filters']['locations']]
                hours_filter = actor['filters']['hours']
                days_filter = [DAYS[day] for day in actor['filters']['days']]

            train_times = [0.0 for x in range(3)]
            t_start = time()
            #print("Training")
            #print(train_data, train_labels, train_item_count)
            if train:
                train_times = exquisitor.train(train_data, train_labels, train_item_count, -1, expansionType,
                                               expansionThreshold, numSuggs, locations_filter, hours_filter,
                                               days_filter)
            #print(train_times)
            #print("Getting suggestions")
            (sugg_list, total, worker_time, sugg_time, sugg_overhead) = exquisitor.suggest(numSuggs, numSegments,
                                                                                           seen_list, locations_filter,
                                                                                           hours_filter, days_filter, -1)
            #print(sugg_list, total, worker_time, sugg_time, sugg_overhead)
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
            (pos,neg,done) = classify_suggestions(sugg_list, compFiles, actor['avgVector'], actor['relevant'],
                                                  posSelPolicy, negSelPolicy, threshold, numPos, numNeg, rd+1,
                                                  distanceCalculation)
            t_classify_stop = time()
            #print("Time to classify: %f" % (t_classify_stop - t_classify_start))
            #print(pos, neg, done)

            if len(sugg_list) == 0:
                metrics[r]['p'].append(0.0)
            else:
                metrics[r]['p'].append(float(len(pos))/len(sugg_list))

            if done:
                metrics[r]['r'].append(1.0)
            else:
                metrics[r]['r'].append(0.0)

            metrics[r]['t'].append(t)
            metrics[r]['done'].append(done)
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

            if done:
                print('%s Actor %d run %d completed task!' % (ts(), actorId, r))
                break
            if len(sugg_list) == 0 and maxB:
                print('%s Actor %d run %d can not advance further!' % (ts(), actorId, r))
                break
            elif len(sugg_list) == 0:
                train = False
            else:
                train = True
                train_data = pos + neg
                train_labels = [1.0 for x in range(len(pos))] + [-1.0 for x in range(len(neg))]
                train_item_count = len(pos) + len(neg)

            rd += 1
            if timed:
                if ttt > 0:
                    current_session_time += ttt
                    #sleep(ttt)
                else:
                    current_session_time += t
            else:
                current_session_time = rd
        print("%s Actor %d run %d done after %d rounds." % (ts(), actorId, r, rd))

    pn_file = ('a%d_PN.json') % actorId
    pn_path = os.path.join(resultDir, pn_file)
    with open(pn_path, 'w') as f:
        json.dump(pn,f)

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
INDEX_CONFIG_FILE_HELP = "Config File containing ecp index information."
FILTERS_FILE_HELP = "JSON File containing filters for all items."
NUMBER_OF_SUGGESTIONS_HELP = "Set number of suggestions to get per round. Default is 25."
NUMBER_OF_WORKERS_HELP = "Set number of workers to use. Default is 1."
NUMBER_OF_SEGMENTS_HELP = "Set number of segments to use. Default is 16."
NUMBER_OF_FEATURES_HELP = "Set number of features in modality. Default is the 1K imagenet classes."
POS_SELECTION_POLICY_HELP = "Select the policy for positive selection. Acc-add (0). Acc-rep (1). Fixed (2)."
NEG_SELECTION_POLICY_HELP = "Select the policy for negative selection. Acc-add (0). Acc-rep (1). Fixed (2). \
                             Rand-local (3). Rand-global (4)."
DISTANCE_THRESHOLD_HELP = "Threshold for selecting suggestion as positive (0.0-2.0). Default is 1.5"
NUMBER_OF_POSITIVES_HELP = "Number of positives to select. Works with selection policy 2, 3 and 4. Default is 12"
NUMBER_OF_NEGATIVES_HELP = "Number of negatives to select. Works with selection policy 2 and 4. Default is 13"
SEARCH_EXPANSION_FIXED_HELP = "The number of clusters selected from training, also known as b. Default is 64."
SEARCH_EXPANSION_AUTO_HELP = "Set this option if the training should automatically select more clusters if \
                              the number of items from the current scope is lower than the expansion \
                              threshold(default=100)."
EXPANSION_THRESHOLD_HELP = "Minimum number of items the training should return. Default is 1000."
FILTER_COUNT_HELP = "Set this option if expansion should be based on LSC active filters."
NUMBER_OF_RUNS_HELP = "Set number of times the experiment runs. Default is 50."
NUMBER_OF_ROUNDS_HELP = "Set number of interaction rounds in each run. Default is 10."
RUN_WITH_SESSION_HELP = "Set this option if the experiments should use session time instead of number of \
                         interaction rounds."
SESSION_TIME_HELP = "Set a custom session time for each run. Default is 300 seconds."
RUN_WITH_TTT_HELP = "Set this option if the experiments should use Time to Think between interaction rounds. \
                     Note: this option is ignored if running with interaction rounds."
TIME_TO_THINK_HELP = "Set a custom time to think per interaction round. Default is 5 seconds."
FILTER_SCENARIO_HELP = "Run with filters or not. 0 = no filters, 1 = computer extracted filters, 2 = user filters, \
                        3 = lifelogger filters, 4 = beyond human filters"
SPIKE_HELP = "Run experiment with spike or not. 0 = random spike, 1 = high scoring non relevant items as spike,\
              1 = relevant items as spike"
PREDICT_HELP = "Set this option if exquisitor is compiled with CCOVERAGE_PROB or PREDICT_COUNT"
CLUSTER_OPT_HELP = "Any cluster containing items below the set id will be marked and only they will be used during selection."
ITEM_FILTER_HELP = "Sets a filter on the set id. Anything below the id passes."
ACTORS_APPEND_HELP = "Which actors to run."
DISTANCE_CALCULATION_HELP = "Which distance calculation to use for feedback selection. 0 = Euclidean, 1 = Mahalanobis"

parser = argparse.ArgumentParser(description="")
parser.add_argument('actors_file', type=str, help=ACTORS_FILE_HELP)
parser.add_argument('result_dir', type=str, help=RESULT_DIR_HELP)
parser.add_argument('result_file', type=str, help=RESULT_FILE_HELP)
parser.add_argument('h5_init_feat_file', type=str, help=INIT_FEAT_FILE_HELP)
parser.add_argument('h5_feat_ids_file', type=str, help=FEAT_IDS_FILE_HELP)
parser.add_argument('h5_ratios_file', type=str, help=RATIOS_FILE_HELP)
parser.add_argument('indx_conf_file', type=str, help=INDEX_CONFIG_FILE_HELP)
parser.add_argument('filters_file', type=str, help=FILTERS_FILE_HELP)
parser.add_argument('locations_file', type=str, help=FILTERS_FILE_HELP)
parser.add_argument('--spike', type=int, default=0, help=SPIKE_HELP)
parser.add_argument('--filter_scenario', type=int, default=0, help=FILTER_SCENARIO_HELP)
parser.add_argument('--measurements', action='store_true', default=False, help=FILTER_SCENARIO_HELP)
parser.add_argument('--noms', type=int, default=1000)
parser.add_argument('--num_suggestions', type=int, default=25, help=NUMBER_OF_SUGGESTIONS_HELP)
parser.add_argument('--num_workers', type=int, default=1, help=NUMBER_OF_WORKERS_HELP)
parser.add_argument('--num_segments', type=int, default=16, help=NUMBER_OF_SEGMENTS_HELP)
parser.add_argument('--num_features', type=int, default=1000, help=NUMBER_OF_FEATURES_HELP)
parser.add_argument('--pos_sel_policy', type=int, default=0, help=POS_SELECTION_POLICY_HELP)
parser.add_argument('--neg_sel_policy', type=int, default=0, help=NEG_SELECTION_POLICY_HELP)
parser.add_argument('--distance_threshold', type=float, default=1.5, help=DISTANCE_THRESHOLD_HELP)
parser.add_argument('--num_pos', type=int, default=12, help=NUMBER_OF_POSITIVES_HELP)
parser.add_argument('--num_neg', type=int, default=13, help=NUMBER_OF_NEGATIVES_HELP)
parser.add_argument('--search_expansion_b', type=int, default=417, help=SEARCH_EXPANSION_FIXED_HELP)
parser.add_argument('--with_search_expansion', action='store_true', default=False, help=SEARCH_EXPANSION_AUTO_HELP)
parser.add_argument('--expansion_threshold', type=int, default=5000, help=EXPANSION_THRESHOLD_HELP)
parser.add_argument('--with_filter_cache', action='store_true', default=False, help=FILTER_COUNT_HELP)
parser.add_argument('--with_stats', action='store_true', default=False, help=PREDICT_HELP)
parser.add_argument('--with_exact', action='store_true', default=False, help=PREDICT_HELP)
parser.add_argument('--with_relax_thresh', action='store_true', default=False, help=PREDICT_HELP)
parser.add_argument('--number_of_runs', type=int, default=50, help=NUMBER_OF_RUNS_HELP)
parser.add_argument('--number_of_rounds', type=int, default=10, help=NUMBER_OF_ROUNDS_HELP)
parser.add_argument('--run_with_session', action='store_true', default=False, help=RUN_WITH_SESSION_HELP)
parser.add_argument('--session_time', type=int, default=300, help=SESSION_TIME_HELP)
parser.add_argument('--run_with_ttt', action='store_true', default=False, help=RUN_WITH_TTT_HELP)
parser.add_argument('--time_to_think', type=int, default=5, help=TIME_TO_THINK_HELP)
parser.add_argument('--cluster_opt', type=int, default=0, help=CLUSTER_OPT_HELP)
parser.add_argument('--item_id_filter', type=int, default=0, help=ITEM_FILTER_HELP)
parser.add_argument('--actors_append', action='append', type=int, default=[], help=ACTORS_APPEND_HELP)
parser.add_argument('--distance_calculation', type=int, default=0, help=DISTANCE_CALCULATION_HELP)
parser.add_argument('--mahal_iv_path', type=str, default="")

args = parser.parse_args()

result_json_file = os.path.join(args.result_dir, args.result_file, args.result_file + '.json')
if os.path.isfile(result_json_file):
    print("RESULT FILE ALREADY EXISTS!")
    exit(0)

result_json_dir = os.path.join(args.result_dir, args.result_file)
if not(os.path.isdir(result_json_dir)):
    os.mkdir(result_json_dir)

distance_calculation = args.distance_calculation
if distance_calculation > 2:
    print("Wrong distance_calculation specified. " + DISTANCE_CALCULATION_HELP)
    exit(0)

if args.distance_calculation == 1:
    if not os.path.isfile(args.mahal_iv_path):
        print("Mahalanobis distance requires a file path to the inverse covariance matrix")
        exit(0)
    else:
        with open(args.mahal_iv_path, 'r') as f:
            IV = np.asarray(json.load(f))

actors = read_actors(args.actors_file)
comp_files = []
comp_files.append(args.h5_init_feat_file)
comp_files.append(args.h5_feat_ids_file)
comp_files.append(args.h5_ratios_file)
with open(args.locations_file, 'r') as f:
    loc = json.load(f)
    for idx,s in enumerate(loc):
        LOCATIONS[s.lower()] = idx

expansion_type = -1
if args.with_search_expansion:
    if args.with_filter_cache and args.with_stats:
        if args.with_relax_thresh:
            expansion_type = 8
        else:
            expansion_type = 4
    elif args.with_stats:
        if args.with_relax_thresh:
            expansion_type = 7
        else:
            expansion_type = 3
    elif args.with_filter_cache:
        if args.with_relax_thresh:
            expansion_type = 6
        else:
            expansion_type = 2
    elif args.with_exact:
        if args.with_relax_thresh:
            expansion_type = 5
        else:
            expansion_type = 1
    else:
        expansion_type = 0
print("Expansion type: %d: " % expansion_type)
    #if args.with_filter_count:
    #    if args.with_predict:
    #        expansion_type = 4  
    #    elif args.with_exact:
    #        expansion_type = 3
    #    else:
    #        expansion_type = 2
    #else:
    #    if args.with_exact:
    #        expansion_type = 1
    #    elif args.with_predict:
    #        expansion_type = 5
    #    else:
    #        expansion_type = 0

initialize_exquisitor(args.noms, args.search_expansion_b, args.num_workers, args.num_segments,
                      args.indx_conf_file, comp_files, 'vis', args.num_features, args.filters_file, args.cluster_opt,
                      args.item_id_filter, expansion_type)
print("%s Initialized!" % ts())

a_to_run = set()
if (len(args.actors_append) == 0):
    a_to_run = set([x for x in range(0,len(actors))])
else:
    a_to_run = set(args.actors_append)

if args.filter_scenario > 0 and args.filter_scenario < 3 and len(args.actors_append) == 0: #Skip actors where no filters are applied
    a_to_run = set([x for x in range(0,len(actors))])
    a_to_run.remove(16)
    a_to_run.remove(17)
    a_to_run.remove(18)
    a_to_run.remove(19)
    a_to_run.remove(20)

for idx, a in enumerate(actors):
    if idx not in a_to_run:
        print('Skipping actor %d' % idx)
        continue
    metrics = {}
    if args.run_with_session:
        if args.run_with_ttt:
            metrics = run_experiment(result_json_dir, idx, a, args.number_of_runs, args.session_time,
                                          args.time_to_think, True, args.num_suggestions, args.num_segments, comp_files,
                                          args.pos_sel_policy, args.neg_sel_policy, args.distance_threshold, args.num_pos,
                                          args.num_neg, args.spike, args.filter_scenario, args.measurements, expansion_type,
                                          args.expansion_threshold, (args.search_expansion_b == MAX_B), distance_calculation)
        else:
            metrics = run_experiment(result_json_dir, idx, a, args.number_of_runs, args.session_time,
                                          0, True, args.num_suggestions, args.num_segments, comp_files, args.pos_sel_policy,
                                          args.neg_sel_policy, args.distance_threshold, args.num_pos, args.num_neg, 
                                          args.spike, args.filter_scenario, args.measurements, expansion_type, 
                                          args.expansion_threshold, (args.search_expansion_b == MAX_B), distance_calculation)
    else:
        metrics = run_experiment(result_json_dir, idx, a, args.number_of_runs, args.number_of_rounds,
                                      0, False, args.num_suggestions, args.num_segments, comp_files, args.pos_sel_policy,
                                      args.neg_sel_policy, args.distance_threshold, args.num_pos, args.num_neg,
                                      args.spike, args.filter_scenario, args.measurements, expasion_type,
                                      args.expansion_threshold, (args.search_expansion_b == MAX_B), distance_calculation)
    results = {}
    if os.path.isfile(result_json_file):
        with open(result_json_file,'r') as f:
            results = json.load(f)
    results[idx] = metrics
    with open(result_json_file,'w') as f:
        json.dump(results,f)


    print("%s Actor %d done" % (ts(),idx))


#with open(result_json_file, 'w') as f:
#    json.dump(metrics,f)
