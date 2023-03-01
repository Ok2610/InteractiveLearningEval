#!/usr/bin/env python2
'''
experiment.py

The experimental script for Exquisitor

Author: Omar Khan, ITU, July 2019
'''

#import exquisitor
import exqjournal as exquisitor
import argparse
import json
import os
from time import time, clock, sleep, strftime

N_SUGGS = 25

N_ACTORS = 50
N_RUNS = 50
N_ROUNDS = 10

SESSION_TIME_LIMIT = 50

MIN = 0
MAX = 1

USER_T_PER_ITEM = 0.1

N_PLACING_TEST = 1500000
N_HAMR = 1500000
N_PLACING = 6500000
N_YFCC100M = 99206564

N_FEAT = {"imagenet": 1000,
          "places": 365,
          "trecvid": 346,
          "yfcc100m": 100,
          "wiki": 100,
          "imagenet_pool5": 1024,
          "places_pool5": 1024,
          None: -1}

N_WORKERS = 1

OPT_NAMES = ['opt_none', 'opt_corr', 'opt_scale', 'opt_all']
OPT_NONE = 1
OPT_DENSE_SVM = 2
OPT_DENSE_RANK = 3

VIS = 0
TXT = 1
MM = 2
MOD_NAMES = ['vis', 'txt', 'mm']

NORM_NONE = 0
NORM_L2 = 1
NORM_NAMES = ['nonorm', 'l2norm']

FEAT_SEL = ['plain', 'tfidf', 'threshold']

REPRE_BT = 0
REPRE_PQ = 1

def ts():
    return "[" + str(strftime("%d %b, %H:%M:%S")) + "]"


def experiment(algorithm,
               dataset,
               iota,
               modality,
               feat_id_vis,
               feat_id_txt,
               layer,
               n_noms,
               opt_mode,
               radius,
               norm,
               feat_sel,
               a_runs_path,
               res_path,
               suffix,
               n_suggs=N_SUGGS,
               actor_i=[0, N_ACTORS],
               timed=False,
               consider_user_t=True,
               t_sess=SESSION_TIME_LIMIT,
               n_rounds=N_ROUNDS,
               force_execution=False,
               n_sq=12,
               k=1024,
               b=256,
               expansion_type=-1,
               threshold=25600):

    # Construct the modality identifier
    mod_id = "%s-" % modality

    if feat_id_vis is not None:
        mod_id += feat_id_vis

        if layer == 'pool5':
            mod_id += "_pool5"

            if algorithm != 'pq':
                feat_id_vis += "_pool5"

    if modality == 'mm':
        mod_id += "_"

    if feat_id_txt is not None:
        mod_id += feat_id_txt

    # Construct the optimization identifier
    opt_id = "opt"

    if opt_mode % OPT_DENSE_SVM == 0:
        opt_id += "_densesvm"

    if opt_mode % OPT_DENSE_RANK == 0:
        opt_id += "_denserank"

    if opt_mode == OPT_NONE:
        opt_id = 'opt_none'

    # Determine the modality activity code
    mod_activity_code = MOD_NAMES.index(modality)

    # Construct the experiment identifier
    if algorithm == 'exquisitor':
        exp_id = "%s-%s-%siota-%snom-%s-%s-%s-%sr" % (algorithm, mod_id,
                                                      iota,
                                                      n_noms,
                                                      feat_sel,
                                                      opt_id,
                                                      norm,
                                                      radius)
    else:
        raise ValueError("Invalid algorithm name: %s" % algorithm)

    if timed:
        exp_id += "_timed"

    if consider_user_t:
        exp_id += "_user"

    # Determine the result path
    if actor_i[0] != 0 or actor_i[1] != N_ACTORS:
        result_dir = "results/%s/%s" % (dataset, exp_id)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            result_path = os.path.join(result_dir, '%s_%s.json' % (actor_i[MIN], actor_i[MAX]))
    elif res_path != "":
        result_path = "%s/%s-%druns-%dactors-%db-%dworkers-%s.json" % (res_path, exp_id, N_RUNS, N_ACTORS, b, N_WORKERS, suffix)
    else:
        result_path = "/var/scratch/bjonsson/results/%s/%s-%druns-%dactors-%db-%dworkers.json" % (dataset, exp_id, N_RUNS, N_ACTORS, b, N_WORKERS)

    # If the experiment has been done already, skip
    if not force_execution and os.path.exists(result_path):
        print("Experiment %s on the %s dataset already completed, skipping." % (exp_id, dataset))
        return

    print("%s <%s, %s> +++ EXPERIMENT STARTED +++" % (ts(), dataset, exp_id))

    # Collect the algorithm parameters and initialize
    if algorithm == 'exquisitor':
        data_dir = "/var/scratch/bjonsson/jzahalka/yfcc100m/subsets/yfcc100m-%s/iota_i64/%s/%s_%s" % (dataset, iota, feat_sel, norm)
        config_file_txt = "txt_index_%s.cnfg" % (dataset)
        config_file_vis = "vis_index_%s.cnfg" % (dataset)
        alg = exquisitor
        arg0 = iota
        arg1 = n_noms
        arg2 = N_WORKERS
        arg3 = mod_activity_code
        arg4 = N_FEAT[feat_id_vis]
        arg5 = N_FEAT[feat_id_txt]
        arg6 = os.path.join(data_dir, "%s_ids.h5" % feat_id_vis)
        arg7 = os.path.join(data_dir, "%s_ratios.h5" % feat_id_vis)
        arg8 = os.path.join(data_dir, "%s_init_feat.h5" % feat_id_vis)
        arg9 = os.path.join(data_dir, "%s_ids.h5" % feat_id_txt)
        arg10 = os.path.join(data_dir, "%s_ratios.h5" % feat_id_txt)
        arg11 = os.path.join(data_dir, "%s_init_feat.h5" % feat_id_txt)
        arg12 = config_file_txt
        arg13 = config_file_vis
        arg14 = b
        arg15 = expansion_type
        arg16 = 0
        arg17 = 0
    else:
        return

    print("Experiment: Calling initialize for alg: %s\n" % alg)
    alg.initialize(arg0, arg1, arg2, arg3, arg4, arg5,
                   arg6, arg7, arg8, arg9, arg10, arg11,
                   arg12, arg13, arg14, arg15, arg16, arg17)

    print("%s <%s, %s> Intelligence module initialized." % (ts(), dataset, exp_id))

    segments = 16
    if b < 16:
        segments = b

    # Prepare the structure that will hold the results
    results = dict()
    results['actors'] = dict()
    results['p'] = [0.0 for x in range(n_rounds)]
    results['r'] = [0.0 for x in range(n_rounds)]
    results['t'] = [0.0 for x in range(n_rounds)]
    results['time-train'] = [0.0 for x in range(n_rounds)]
    results['time-b'] = [0.0 for x in range(n_rounds)]
    results['time-train-pipe'] = [0.0 for x in range(n_rounds)]
    results['time-suggest'] = [0.0 for x in range(n_rounds)]
    results['time-suggest-overhead'] = [0.0 for x in range(n_rounds)]
    results['workers'] = dict()
    for w in range(segments):
        results['workers'][w] = dict()
        results['workers'][w]['time-score'] = [0.0 for x in range(n_rounds)]
        results['workers'][w]['total-scored'] = [0.0 for x in range(n_rounds)]

    if timed:
        results['t_sess'] = t_sess

    for a in range(actor_i[0], actor_i[1]):
        # Load the actor's runs
        if a_runs_path == "":
            actor_runs_path = "/var/scratch/bjonsson/jzahalka/yfcc100m/subsets/yfcc100m-%s/actor_runs/a%s_%s.json" % (dataset, a, radius)
        else:
            actor_runs_path = "%s/a%s_%s.json" % (a_runs_path, a, radius)

        with open(actor_runs_path, 'r') as f:
            actor_runs = json.loads(f.read())

        # Load the actor's items
        actor_items_path = "/var/scratch/bjonsson/jzahalka/yfcc100m/subsets/yfcc100m-%s/actor_item_i/a%s_%s.json" % (dataset, a, radius)

        with open(actor_items_path, 'r') as f:
            actor_items = json.loads(f.read())

        # Prepare the structures holding the results
        results['actors'][a] = dict()
        results['actors'][a]['runs'] = dict()
        results['actors'][a]['p'] = [0.0 for i in range(n_rounds)]
        results['actors'][a]['r'] = [0.0 for i in range(n_rounds)]
        results['actors'][a]['t'] = [0.0 for i in range(n_rounds)]
        results['actors'][a]['time-train'] = [0.0 for i in range(n_rounds)]
        results['actors'][a]['time-b'] = [0.0 for i in range(n_rounds)]
        results['actors'][a]['time-train-pipe'] = [0.0 for i in range(n_rounds)]
        results['actors'][a]['time-suggest'] = [0.0 for i in range(n_rounds)]
        results['actors'][a]['time-suggest-overhead'] = [0.0 for i in range(n_rounds)]
        results['actors'][a]['workers'] = dict()
        for w in range(segments):
            results['actors'][a]['workers'][w] = dict()
            results['actors'][a]['workers'][w]['time-score'] = [0.0 for x in range(n_rounds)]
            results['actors'][a]['workers'][w]['total-scored'] = [0.0 for x in range(n_rounds)]

        all_rel = set([long(x) for x in actor_items])

        # Execute all runs
        counter = 0
        for run in range(N_RUNS):
            counter = counter + 1
            #print("going to inner round: %d\n" % counter)
            results['actors'][a]['runs'][run] = dict()
            results['actors'][a]['runs'][run]['p'] = []
            results['actors'][a]['runs'][run]['r'] = []
            results['actors'][a]['runs'][run]['t'] = []
            results['actors'][a]['runs'][run]['time-train'] = []
            results['actors'][a]['runs'][run]['time-b'] = []
            results['actors'][a]['runs'][run]['time-train-pipe'] = []
            results['actors'][a]['runs'][run]['time-suggest'] = []
            results['actors'][a]['runs'][run]['time-suggest-overhead'] = []
            results['actors'][a]['runs'][run]['suggs'] = []
            results['actors'][a]['runs'][run]['workers'] = dict()
            for w in range(segments):
                results['actors'][a]['runs'][run]['workers'][w] = dict()
                results['actors'][a]['runs'][run]['workers'][w]['total-scored'] = []
                results['actors'][a]['runs'][run]['workers'][w]['time-score'] = []

            pos = [long(x) for x in actor_runs[str(run)]['pos']]
            neg = [long(x) for x in actor_runs[str(run)]['neg_init']]

            if algorithm == 'exquisitor':
                arg1 = pos + neg
                arg2 = [1.0 for x in range(len(pos))] + [-1.0 for x in range(len(neg))]
                arg3 = len(pos) + len(neg)

            seen = [] 
            seen_rel = set()
            unseen_rel = set([long(x) for x in actor_items])
            alg.reset_model(1)
            # Execute all interaction rounds
            if timed:
                print("timed\n")
                session_start = time()

            session_in_progress = True
            rd = 0

            while session_in_progress:
                # Perform the interaction round
                t_test = clock()
                t = time()
                #print("training...")
                train_times = alg.train(arg1, arg2, arg3, -1, expansion_type, threshold, n_suggs)
                #print(train_times)
                #print("suggesting...")
                (sugg_list, total, worker_time, sugg_time, sugg_overhead) = alg.suggest(n_suggs, segments, seen, -1)
                #print("got suggestions")
                #print(sugg_list)
                t = time() - t
                t = clock() - t_test

                t_overhead = time()
                # Process the suggestions
                suggs = set(sugg_list)
                suggs_rel = unseen_rel & suggs
                seen_rel |= suggs_rel
                unseen_rel -= suggs_rel
                seen_set = set(seen)
                seen_set |= suggs
                seen = list(seen_set)

                # Compute precision and recall
                p = float(len(suggs_rel)) / len(suggs)
                r = float(len(seen_rel)) / len(all_rel)

                # Set the training parameters for the next round
                pos = [long(x) for x in suggs_rel]
                neg = [long(x) for x in actor_runs[str(run)]['neg'][rd]]

                alg.reset_model(0)

                if algorithm == 'exquisitor':
                    arg1 = pos + neg
                    arg2 = [1.0 for x in range(len(pos))] + [-1.0 for x in range(len(neg))]
                    arg3 = len(pos) + len(neg)

                for w in range(segments):
                    results['actors'][a]['runs'][run]['workers'][w]['total-scored'].append(total[w])
                    results['actors'][a]['runs'][run]['workers'][w]['time-score'].append(worker_time[w])

                # Record the round results
                results['actors'][a]['runs'][run]['p'].append(p)
                results['actors'][a]['runs'][run]['r'].append(r)
                results['actors'][a]['runs'][run]['time-train'].append(train_times[0])
                results['actors'][a]['runs'][run]['time-b'].append(train_times[1])
                results['actors'][a]['runs'][run]['time-train-pipe'].append(train_times[2])
                results['actors'][a]['runs'][run]['time-suggest'].append(sugg_time)
                results['actors'][a]['runs'][run]['time-suggest-overhead'].append(sugg_overhead)
                results['actors'][a]['runs'][run]['suggs'].append(list(suggs))
                if not timed:
                    results['actors'][a]['runs'][run]['t'].append(t)

                rd += 1
                t_overhead = time() - t_overhead

                # Decide if to keep going
                if timed:
                    if consider_user_t:
                        sleep(USER_T_PER_ITEM * n_suggs - t_overhead)

                    tick = time() - session_start
                    results['actors'][a]['runs'][run]['t'].append(tick)
                    if tick > t_sess:
                        session_in_progress = False

                    if rd >= n_rounds:
                        rd = 0
                else:
                    if rd >= n_rounds:
                        session_in_progress = False

            print("%s <%s, %s> Actor %s, run %s done." %
                  (ts(), dataset, exp_id, a + 1, run + 1))

        # Compute the averages per actor
        if timed:
            results['actors'][a]['r'] = 0.0

            for run in results['actors'][a]['runs']:
                results['actors'][a]['r'] += results['actors'][a]['runs'][run]['r'][-1]

            results['actors'][a]['r'] /= len(results['actors'][a]['runs'])
        else:
            for m in ['p', 'r', 't','time-train','time-b', 'time-suggest', 'time-train-pipe','time-suggest-overhead']:
                for rd in range(n_rounds):
                    for run in results['actors'][a]['runs']:
                        results['actors'][a][m][rd] += results['actors'][a]['runs'][run][m][rd]

                    results['actors'][a][m][rd] /= len(results['actors'][a]['runs'])
            for w in range(segments):
                for rd in range(n_rounds):
                    for run in results['actors'][a]['runs']:
                        results['actors'][a]['workers'][w]['total-scored'][rd] +=\
                            results['actors'][a]['runs'][run]['workers'][w]['total-scored'][rd]
                        results['actors'][a]['workers'][w]['time-score'][rd] +=\
                            results['actors'][a]['runs'][run]['workers'][w]['time-score'][rd]

                    results['actors'][a]['workers'][w]['total-scored'][rd] /= len(results['actors'][a]['runs'])
                    results['actors'][a]['workers'][w]['time-score'][rd] /= len(results['actors'][a]['runs'])

    alg.terminate()

    # Compute the averages for the algorithm
    if actor_i[0] == 0 and actor_i[1] == N_ACTORS:
        if timed:
            print("compute average")
            results['r'] = 0.0

            for a in range(actor_i[0], actor_i[1]):
                results['r'] += results['actors'][a]['r']

            results['r'] /= N_ACTORS
        else:
            for m in ['p', 'r', 't','time-train','time-b', 'time-suggest', 'time-train-pipe', 'time-suggest-overhead']:
                for rd in range(n_rounds):
                    for a in range(actor_i[0], actor_i[1]):
                        results[m][rd] += results['actors'][a][m][rd]

                    results[m][rd] /= N_ACTORS

            for w in range(segments):
                for rd in range(n_rounds):
                    for a in range(actor_i[0], actor_i[1]):
                        results['workers'][w]['total-scored'][rd] += results['actors'][a]['workers'][w]['total-scored'][rd]
                        results['workers'][w]['time-score'][rd] += results['actors'][a]['workers'][w]['time-score'][rd]
                    results['workers'][w]['total-scored'][rd] /= N_ACTORS
                    results['workers'][w]['time-score'][rd] /= N_ACTORS


    print("%s <%s, %s> Writing the results..." %
          (ts(), dataset, exp_id))


    with open(result_path, 'w') as f:
        f.write(json.dumps(results))

    print("%s <%s, %s> +++ EXPERIMENT FINISHED +++" %
          (ts(), dataset, exp_id))

parser = argparse.ArgumentParser()
parser.add_argument('algorithm', choices=['baseline', 'exquisitor', 'pq'])
parser.add_argument('dataset', choices=['placing_test', 'full', 'hamr'])
parser.add_argument('modality', choices=MOD_NAMES)
parser.add_argument('iota', type=int, choices=[1, 5, 10, 15])
parser.add_argument('opt_mode', type=int)
parser.add_argument('norm', choices=["nonorm", "l2norm"])
parser.add_argument('feat_sel', choices=["plain", "tfidf", "threshold"])
parser.add_argument('radius', type=int, choices=[1, 10, 100, 1000])
parser.add_argument('--vfi')
parser.add_argument('--tfi')
parser.add_argument('--b', type=int,default=10)
parser.add_argument('--layer')
parser.add_argument('--timed', action='store_true', default=False)
parser.add_argument('--user', action='store_true', default=False)
parser.add_argument('--force', action='store_true', default=False)
parser.add_argument('--noms', type=int)
parser.add_argument('--suggs', type=int, default=25)
parser.add_argument('--min_a', type=int)
parser.add_argument('--max_a', type=int)
parser.add_argument('--n_rounds', type=int, default=N_ROUNDS)
parser.add_argument('--n_neg', type=int, default=100)
parser.add_argument('--n_repre', type=int, default=1)
parser.add_argument('--t_sess', type=int, default=SESSION_TIME_LIMIT)
parser.add_argument('--n_sq', type=int, default=0)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--a_runs_path', type=str, default="")
parser.add_argument('--res_path', type=str, default="")
parser.add_argument('--suffix', type=str, default="")
parser.add_argument('--expansion_type', type=int, default=-1)
parser.add_argument('--expansion_threshold', type=int, default=25600)

args = parser.parse_args()

if args.min_a is not None and args.max_a is not None:
    actors = [args.min_a, args.max_a]
else:
    actors = [0, N_ACTORS]

experiment(args.algorithm,
           args.dataset,
           args.iota,
           args.modality,
           args.vfi,
           args.tfi,
           args.layer,
           args.noms,
           args.opt_mode,
           args.radius,
           args.norm,
           args.feat_sel,
           args.a_runs_path,
           args.res_path,
           args.suffix,
           args.suggs,
           actors,
           args.timed,
           args.user,
           args.t_sess,
           args.n_rounds,
           args.force,
           args.n_sq,
           args.k,
           args.b,
           args.expansion_type,
           args.expansion_threshold)


