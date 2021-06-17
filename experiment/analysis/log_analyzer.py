import glob
from shutil import copy2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pprint
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

prompt = lambda q : input("{} (y/n): ".format(q)).lower().strip()[:1] == "y"

def parse_log(filename, params, eval_k, cl_to_plot_id, target_measure, print_params, start_line=None, end_line=None):
    res_map={}
    errors = {}
    losses = {}
    MRRs = {}
    MAPs = {}
    AUCs = {}
    prec = {}
    rec = {}
    f1 = {}
    prec_at_k = {}
    rec_at_k = {}
    f1_at_k = {}
    prec_cl = {}
    rec_cl = {}
    f1_cl = {}
    prec_at_k_cl = {}
    rec_at_k_cl = {}
    f1_at_k_cl = {}
    best_measure = {}
    best_epoch = {}
    target_metric_best = {}

    last_test_ep={}
    last_test_ep['precision'] =  '-'
    last_test_ep['recall'] = '-'
    last_test_ep['F1'] = '-'
    last_test_ep['AVG-precision'] = '-'
    last_test_ep['AVG-recall'] = '-'
    last_test_ep['AVG-F1'] = '-'
    last_test_ep['precision@'+str(eval_k)] =  '-'
    last_test_ep['recall@'+str(eval_k)] = '-'
    last_test_ep['F1@'+str(eval_k)] = '-'
    last_test_ep['AVG-precision@'+str(eval_k)] =  '-'
    last_test_ep['AVG-recall@'+str(eval_k)] = '-'
    last_test_ep['AVG-F1@'+str(eval_k)] = '-'
    last_test_ep['mrr'] = '-'
    last_test_ep['map'] = '-'
    last_test_ep['auc'] = '-'
    last_test_ep['best_epoch'] = -1

    set_names = ['TRAIN', 'VALID', 'TEST']
    finished = False
    epoch = 0

    metrics_names = ["error" ,
                     "loss" ,
                     "mrr" ,
                     "map" ,
                     "auc" ,
                     "gmauc" ,
                     "lp_map" ,
                     "lp_auc",
                     "1000_auc",
                     "1000_map",
                     "100_auc",
                     "100_map",
                     "10_auc",
                     "10_map",
                     "1_auc",
                     "1_map",
    ]
    metrics = {metric: {} for metric in metrics_names}

    for s in set_names:
        for metric in metrics:
            metrics[metric][s] = {}
        prec[s] = {}
        rec[s] = {}
        f1[s] = {}
        prec_at_k[s] = {}
        rec_at_k[s] = {}
        f1_at_k[s] = {}
        prec_cl[s] = {}
        rec_cl[s] = {}
        f1_cl[s] = {}
        prec_at_k_cl[s] = {}
        rec_at_k_cl[s] = {}
        f1_at_k_cl[s] = {}

        best_measure[s] = 0
        best_epoch[s] = -1

    str_comments=''
    str_comments1=''

    exp_params={}

    #print ("Start parsing: ",filename, 'starting at', start_line)
    with open(filename) as f:
        params_line=True
        readlr=False
        line_nr = 0
        for line in f:
            line_nr += 1
            if start_line != None and start_line > line_nr:
                continue
            if end_line != None and end_line <= line_nr:
                break

            line=line.replace('INFO:root:','').replace('\n','')
            if params_line: #print parameters
                if "'learning_rate':" in line:
                    readlr=True
                if not readlr:
                    str_comments+=line+'\n'
                else:
                    str_comments1+=line+'\n'
                if params_line: #print parameters
                    for p in params:
                        str_p='\''+p+'\': '
                        if str_p in line:
                            exp_params[p]=line.split(str_p)[1].split(',')[0]
                if line=='':
                    params_line=False

            if 'TRAIN epoch' in line or 'VALID epoch' in line or 'TEST epoch' in line:
                set_name = line.split(' ')[1]
                previous_epoch = epoch
                epoch = int(line.split(' ')[3])
                if set_name=='TEST':
                    last_test_ep['best_epoch'] = epoch
                if epoch>=50000:
                    break
                if previous_epoch > epoch and epoch == 1:
                    epoch = previous_epoch #used to distinguish between downstream and frozen decoder
                    break #A new training has started, e.g frozen encoder or downstream
                #print('set_name', set_name, 'epoch', epoch)
            if 'Number of parameters' in line:
                res_map['num_gcn_params'] = int(line.split('GCN: ')[1].split(',')[0])
                res_map['num_cls_params'] = int(line.split('Classifier: ')[1].split(',')[0])
                res_map['num_total_params'] = int(line.split('Total: ')[1].split(',')[0])

                assert(res_map['num_gcn_params'] + res_map['num_cls_params'] == res_map['num_total_params'])

            if "mean" in line:
                for metric in metrics:
                    if "mean {} ".format(metric) in line:
                        v=float(line.split('mean {} '.format(metric))[1].split(' ')[0])
                        metrics[metric][set_name][epoch]=v
                        if target_measure==metric:
                            if target_measure == 'loss':
                                is_better = v<best_measure[set_name]
                            else:
                                is_better = v>best_measure[set_name]
                            if is_better:
                                best_measure[set_name]=v
                                best_epoch[set_name]=epoch
                        if set_name=='TEST':
                            last_test_ep[metric] = v

            if 'measures microavg' in line:
                prec[set_name][epoch]=float(line.split('precision ')[1].split(' ')[0])
                rec[set_name][epoch]=float(line.split('recall ')[1].split(' ')[0])
                f1[set_name][epoch]=float(line.split('f1 ')[1].split(' ')[0])
                if (target_measure=='avg_p' or target_measure=='avg_r' or target_measure=='avg_f1'):
                    if target_measure=='avg_p':
                        v=prec[set_name][epoch]
                    elif target_measure=='avg_r':
                        v=rec[set_name][epoch]
                    else: #F1
                        v=f1[set_name][epoch]
                    if v>best_measure[set_name]:
                        best_measure[set_name]=v
                        best_epoch[set_name]=epoch
                if set_name=='TEST':
                    last_test_ep['AVG-precision'] = prec[set_name][epoch]
                    last_test_ep['AVG-recall'] = rec[set_name][epoch]
                    last_test_ep['AVG-F1'] = f1[set_name][epoch]

            elif 'measures@'+str(eval_k)+' microavg' in line:
                prec_at_k[set_name][epoch]=float(line.split('precision ')[1].split(' ')[0])
                rec_at_k[set_name][epoch]=float(line.split('recall ')[1].split(' ')[0])
                f1_at_k[set_name][epoch]=float(line.split('f1 ')[1].split(' ')[0])
                if set_name=='TEST':
                    last_test_ep['AVG-precision@'+str(eval_k)] =  prec_at_k[set_name][epoch]
                    last_test_ep['AVG-recall@'+str(eval_k)] = rec_at_k[set_name][epoch]
                    last_test_ep['AVG-F1@'+str(eval_k)] = f1_at_k[set_name][epoch]
            elif 'measures for class ' in line:
                cl=int(line.split('class ')[1].split(' ')[0])
                if cl not in prec_cl[set_name]:
                    prec_cl[set_name][cl] = {}
                    rec_cl[set_name][cl] = {}
                    f1_cl[set_name][cl] = {}
                prec_cl[set_name][cl][epoch]=float(line.split('precision ')[1].split(' ')[0])
                rec_cl[set_name][cl][epoch]=float(line.split('recall ')[1].split(' ')[0])
                f1_cl[set_name][cl][epoch]=float(line.split('f1 ')[1].split(' ')[0])
                if (target_measure=='p' or target_measure=='r' or target_measure=='f1') and cl==cl_to_plot_id:
                    if target_measure=='p':
                        v=prec_cl[set_name][cl][epoch]
                    elif target_measure=='r':
                        v=rec_cl[set_name][cl][epoch]
                    else: #F1
                        v=f1_cl[set_name][cl][epoch]
                    if v>best_measure[set_name]:
                        best_measure[set_name]=v
                        best_epoch[set_name]=epoch
                if set_name=='TEST':
                    last_test_ep['precision'] = prec_cl[set_name][cl][epoch]
                    last_test_ep['recall'] = rec_cl[set_name][cl][epoch]
                    last_test_ep['F1'] = f1_cl[set_name][cl][epoch]
            elif 'measures@'+str(eval_k)+' for class ' in line:
                cl=int(line.split('class ')[1].split(' ')[0])
                if cl not in prec_at_k_cl[set_name]:
                    prec_at_k_cl[set_name][cl] = {}
                    rec_at_k_cl[set_name][cl] = {}
                    f1_at_k_cl[set_name][cl] = {}
                prec_at_k_cl[set_name][cl][epoch]=float(line.split('precision ')[1].split(' ')[0])
                rec_at_k_cl[set_name][cl][epoch]=float(line.split('recall ')[1].split(' ')[0])
                f1_at_k_cl[set_name][cl][epoch]=float(line.split('f1 ')[1].split(' ')[0])
                if (target_measure=='p@k' or target_measure=='r@k' or target_measure=='f1@k') and cl==cl_to_plot_id:
                    if target_measure=='p@k':
                        v=prec_at_k_cl[set_name][cl][epoch]
                    elif target_measure=='r@k':
                        v=rec_at_k_cl[set_name][cl][epoch]
                    else:
                        v=f1_at_k_cl[set_name][cl][epoch]
                    if v>best_measure[set_name]:
                        best_measure[set_name]=v
                        best_epoch[set_name]=epoch
                if set_name=='TEST':
                    last_test_ep['precision@'+str(eval_k)] = prec_at_k_cl[set_name][cl][epoch]
                    last_test_ep['recall@'+str(eval_k)] = rec_at_k_cl[set_name][cl][epoch]
                    last_test_ep['F1@'+str(eval_k)] = f1_at_k_cl[set_name][cl][epoch]
            if 'FINISHED' in line:
                finished = True

    if  best_epoch['TEST']<0 and  best_epoch['VALID']<0 or last_test_ep['best_epoch']<1:
        # Nothing learned, it is useless, abort
        print ('best_epoch<0: -> skip')
        target_best = {}
        target_best['TEST'] = 0
        str_legend = 'useless'
        str_results = 0
        return res_map, exp_params, metrics, str_legend, str_results, target_best, finished, line_nr, epoch

    if start_line == None:
        # Will fail for frozen encoder and downstream runs, so only do this for the first parse
        res_map['model'] = exp_params['model'].replace("'","")
        str_params=(pprint.pformat(exp_params))

        if print_params:
            print ('str_params:\n', str_params)

    if best_epoch['VALID']>=0:
        best_ep = best_epoch['VALID']
        #print ('Highest %s values among all epochs: TRAIN  %0.4f\tVALID  %0.4f\tTEST %0.4f' % (target_measure, best_measure['TRAIN'], best_measure['VALID'], best_measure['TEST']))
    else:
        best_ep = best_epoch['TEST']
        #print ('Highest %s values among all epochs:\tTRAIN F1 %0.4f\tTEST %0.4f' % (target_measure, best_measure['TRAIN'], best_measure['TEST']))

    use_latest_ep = True
    try:
        #print ('Values at best Valid Epoch (%d) for target class: TEST Precision %0.4f - Recall %0.4f - F1 %0.4f' % (best_ep, prec_cl['TEST'][cl_to_plot_id][best_ep],rec_cl['TEST'][cl_to_plot_id][best_ep],f1_cl['TEST'][cl_to_plot_id][best_ep]))
        #print ('Values at best Valid Epoch (%d) micro-AVG: TEST Precision %0.4f - Recall %0.4f - F1 %0.4f' % (best_ep, prec['TEST'][best_ep],rec['TEST'][best_ep],f1['TEST'][best_ep]))
        res_map['precision'] =  prec_cl['TEST'][cl_to_plot_id][best_ep]
        res_map['recall'] = rec_cl['TEST'][cl_to_plot_id][best_ep]
        res_map['F1'] = f1_cl['TEST'][cl_to_plot_id][best_ep]
        res_map['AVG-precision'] = prec['TEST'][best_ep]
        res_map['AVG-recall'] = rec['TEST'][best_ep]
        res_map['AVG-F1'] = f1['TEST'][best_ep]
    except:
        res_map['precision'] =  last_test_ep['precision']
        res_map['recall'] = last_test_ep['recall']
        res_map['F1'] = last_test_ep['F1']
        res_map['AVG-precision'] = last_test_ep['AVG-precision']
        res_map['AVG-recall'] = last_test_ep['AVG-F1']
        res_map['AVG-F1'] = last_test_ep['AVG-F1']
        use_latest_ep = False
        #print ('WARNING: last epoch not finished, use the previous one.')

    try:
        #print ('Values at best Valid Epoch (%d) for target class@%d: TEST Precision %0.4f - Recall %0.4f - F1 %0.4f' % (best_ep, eval_k, prec_at_k_cl['TEST'][cl_to_plot_id][best_ep],rec_at_k_cl['TEST'][cl_to_plot_id][best_ep],f1_at_k_cl['TEST'][cl_to_plot_id][best_ep]))
        res_map['precision@'+str(eval_k)] =  prec_at_k_cl['TEST'][cl_to_plot_id][best_ep]
        res_map['recall@'+str(eval_k)] = rec_at_k_cl['TEST'][cl_to_plot_id][best_ep]
        res_map['F1@'+str(eval_k)] = f1_at_k_cl['TEST'][cl_to_plot_id][best_ep]

        #print ('Values at best Valid Epoch (%d) micro-AVG@%d: TEST Precision %0.4f - Recall %0.4f - F1 %0.4f' % (best_ep, eval_k, prec_at_k['TEST'][best_ep],rec_at_k['TEST'][best_ep],f1_at_k['TEST'][best_ep]))
        res_map['AVG-precision@'+str(eval_k)] =  prec_at_k['TEST'][best_ep]
        res_map['AVG-recall@'+str(eval_k)] = rec_at_k['TEST'][best_ep]
        res_map['AVG-F1@'+str(eval_k)] = f1_at_k['TEST'][best_ep]

    except:
        res_map['precision@'+str(eval_k)] =  last_test_ep['precision@'+str(eval_k)]
        res_map['recall@'+str(eval_k)] = last_test_ep['recall@'+str(eval_k)]
        res_map['F1@'+str(eval_k)] = last_test_ep['F1@'+str(eval_k)]
        res_map['AVG-precision@'+str(eval_k)] = last_test_ep['AVG-precision@'+str(eval_k)]
        res_map['AVG-recall@'+str(eval_k)] = last_test_ep['AVG-recall@'+str(eval_k)]
        res_map['AVG-F1@'+str(eval_k)] = last_test_ep['AVG-F1@'+str(eval_k)]

    for metric in metrics:
        if len(metrics[metric]['TEST']) <= 0:
            continue
        try:
            if metric == target_measure:
                target_metric_best['TRAIN'] = metrics[metric]['TRAIN'][best_ep]
                target_metric_best['VALID'] = metrics[metric]['VALID'][best_ep]
                target_metric_best['TEST'] = metrics[metric]['TEST'][best_ep]

            #print('Values at best Valid Epoch ({}) {}: TRAIN  {} - VALID {} - TEST {}'.format(
            #    best_ep,
            #    metric,
            #    metrics[metric]['TRAIN'][best_ep],
            #    metrics[metric]['VALID'][best_ep],
            #    metrics[metric]['TEST'][best_ep]))
            res_map[metric] = metrics[metric]['TEST'][best_ep]
        except:
            res_map[metric] = last_test_ep[metric]
            #print ('WARNING: last epoch not finished, use the previous one.')

    if use_latest_ep:
        res_map['best_epoch'] = best_ep
    else:
        #print ('WARNING: last epoch not finished, use the previous one.')
        res_map['best_epoch'] = last_test_ep['best_epoch']

    str_results = ''
    str_legend = ''
    for k, v in res_map.items():
        str_results+=str(v)+','
        str_legend+=str(k)+','
    for k, v in exp_params.items():
        str_results+=str(v)+','
        str_legend+=str(k)+','
    log_file = filename.split('/')[-1].split('.log')[0]
    res_map['log_file'] = log_file
    grid_cell = log_file.split('grid_')[1]
    res_map['grid_cell'] = grid_cell
    str_results+='{},{}'.format(log_file, grid_cell)
    str_legend+='log_file,grid_cell'
    #print ('\n\nCSV-like output:')
    #print (str_legend)
    #print (str_results)
    return res_map, exp_params, metrics, str_legend, str_results, target_metric_best, finished, line_nr, epoch

def parse_all_logs_in_folder(log_folder, return_continuous_encoder_logs=False):
    cl_to_plot_id = 1 # Target class, typically the low frequent one

    # We don't do edge classification here
    #if 'reddit' in log_folder or ('bitcoin' in log_folder and 'edge' in log_folder):
    #    cl_to_plot_id = 0 # 0 for reddit dataset_name or bitcoin edge cls

    simulate_early_stop = 0 # Early stop patience
    eval_k = 1000 # to compute metrics @K (for instance precision@1000)
    print_params = False # Print the parameters of each simulation
    ##### End parameters ######

    #if 'elliptic' in log_folder or 'reddit' in log_folder or 'enron' in log_folder or ('bitcoin' in log_folder and 'edge' in log_folder):
    #	target_measure='f1' # map mrr auc f1 p r loss avg_p avg_r avg_f1
    #else:
    #    target_measure='map' # map mrr auc f1 p r loss avg_p avg_r avg_f1
    target_measure='map' # map mrr auc f1 p r loss avg_p avg_r avg_f1


    # Hyper parameters to analyze
    params = []
    params.append('learning_rate')
    params.append('num_hist_steps')
    params.append('layer_1_feats')
    params.append('lstm_l1_feats')
    params.append('class_weights')
    params.append('adj_mat_time_window')
    params.append('cls_feats')
    params.append('model')
    params.append('val_sampling')

    logs = {}
    continuous_encoder_logs = {}
    csv = []
    csv_continuous_encoder = []
    header = None
    log_folderfiles = glob.glob(log_folder+'*')
    printstr = ''
    best_log_file = ''
    best_log_file_continuous_encoder = ''
    best_target_metric = 0
    best_target_metric_continuous_encoder = 0

    for log_file in log_folderfiles:
        if log_file.endswith(".log") and not 'best_' in log_file:
            # First check whether it is downstream learning or not, then check if it is only pre-training (encoder only for continuous)
            if 'decoder' in log_file and 'learning_rate' in log_file:
                logs[log_file] = {}
                num_lines = sum(1 for line in open(log_file))
                (res_map, exp_params, metrics, str_legend, str_results,
                 target_metric_best, finished, end_line, end_epoch) = parse_log(log_file, params, eval_k, cl_to_plot_id,
                                                                                target_measure, print_params)
                if end_line < num_lines:
                    if end_epoch == 0:
                        # Downstream
                        downstream_results = parse_log(log_file, params, eval_k, cl_to_plot_id,
                                                       target_measure, print_params)
                        logs[log_file]['downstream'] = {}
                        logs[log_file]['downstream']['res_map'] = downstream_results[0]
                        logs[log_file]['downstream']['metrics'] = downstream_results[2]
                    else:
                        # Using a frozen decoder, store continuous training before filling log file results with frozen decoder training
                        logs[log_file]['continuous_training'] = {}
                        logs[log_file]['continuous_training']['res_map'] = res_map
                        logs[log_file]['continuous_training']['exp_params'] = exp_params
                        logs[log_file]['continuous_training']['metrics'] = metrics
                        logs[log_file]['continuous_training']['str_legend'] = str_legend
                        logs[log_file]['continuous_training']['str_results'] = str_results
                        logs[log_file]['continuous_training']['target_metric_best'] = target_metric_best
                        logs[log_file]['continuous_training']['finished'] = finished

                        (res_map, _, metrics, str_legend, str_results,
                        target_metric_best, finished, end_line, end_epoch) = parse_log(log_file, params, eval_k, cl_to_plot_id,
                                                                                       target_measure, print_params, start_line=end_line)


                # Comment out if old log without finish signal
                #if not finished:
                #    printstr+='Log not finished. Skipping {}\n'.format(log_file)
                #    continue
                logs[log_file]['res_map'] = res_map
                logs[log_file]['exp_params'] = exp_params
                logs[log_file]['metrics'] = metrics
                logs[log_file]['str_legend'] = str_legend
                logs[log_file]['str_results'] = str_results
                logs[log_file]['target_metric_best'] = target_metric_best
                logs[log_file]['finished'] = finished

                try:
                    cell_best = target_metric_best['TEST']
                except KeyError:
                    #print("No test epoch to use")
                    continue
                if best_target_metric < cell_best:
                    best_target_metric = cell_best
                    best_log_file = log_file

                if len(csv) <= 0:
                    header = str_legend
                    csv = [str_legend, str_results]
                else:
                    if(str_legend == header):
                        csv.append(str_results)
                    else:
                        print('Warning header was not correct didn\'t did a file return badly?')
            elif 'decoder' in log_file: #training encoder for continuous
                continuous_encoder_logs[log_file] = {}
                num_lines = sum(1 for line in open(log_file))
                (res_map, exp_params, metrics, str_legend, str_results,
                 target_metric_best, finished, end_line, end_epoch) = parse_log(log_file, params, eval_k, cl_to_plot_id,
                                                                                target_measure, print_params)
                continuous_encoder_logs[log_file]['res_map'] = res_map
                continuous_encoder_logs[log_file]['exp_params'] = exp_params
                continuous_encoder_logs[log_file]['metrics'] = metrics
                continuous_encoder_logs[log_file]['str_legend'] = str_legend
                continuous_encoder_logs[log_file]['str_results'] = str_results
                continuous_encoder_logs[log_file]['target_metric_best'] = target_metric_best
                continuous_encoder_logs[log_file]['finished'] = finished

                cell_best = target_metric_best['TEST']
                if best_target_metric_continuous_encoder < cell_best:
                    best_target_metric_continuous_encoder = cell_best
                    best_log_file_continuous_encoder = log_file

                if len(csv) <= 0:
                    header = str_legend
                    csv_continuous_encoder = [str_legend, str_results]
                else:
                    if(str_legend == header):
                        csv_continuous_encoder.append(str_results)
                    else:
                        print('Warning header was not correct didn\'t did a file return badly?')

            else: #Downstream learning
                # Skipping downstream learning for now.
                pass
    #print(printstr)
    if not return_continuous_encoder_logs:
        return logs, csv, best_log_file
    else:
        return logs, csv, best_log_file, continuous_encoder_logs, csv_continuous_encoder, best_log_file_continuous_encoder

def save_best_log(best_log_file):
    # Add 'best_' in front of filename of the best log.
    best_exist = False
    existing_logs = []
    log_folderfiles = glob.glob(log_folder+'*')
    for log_file in log_folderfiles:
        if 'best_' in log_file:
            existing_logs.append(log_file)
            best_exist = True
    if best_exist and not prompt('Best log already exists, existing logs:\n{}\n write anyway?'
                                 .format('\n'.join(existing_logs))):
        print("Skipping saving best log")
        pass
    else:
        print('Saving best log')
        split_log_file = best_log_file.split('/')
        split_log_file[-1] = "best_"+split_log_file[-1]
        best_log_new_name = "/".join(split_log_file)
        copy2(best_log_file, best_log_new_name)


def write_csv(csv, log_folder):
    csv_file = log_folder+'results.csv'
    log_folderfiles = glob.glob(log_folder+'*')
    if csv_file in log_folderfiles and not prompt('CSV already calculated, overwrite?'):
        print("Skipping writing to csv")
        pass
    else:
        print('Writing to csv')
        with open(csv_file, 'w') as f:
            for line in csv:
                f.write(line+'\n')


# Plot one metric
def plot_metric(metric, metric_name, plot_train):
    df = pd.DataFrame.from_dict(metric)
    if not plot_train:
        df = df.drop(['TRAIN'], axis=1)
    else:
        # Rearrange columns so Train is at the end, thus they keep the same colours
        cols = df.columns.to_list()
        cols.append(cols.pop(0))
        df = df[cols]
    df['epochs'] = df.index
    dfm = pd.melt(df, id_vars=['epochs'], var_name='set', value_name=metric_name)
    fig = px.scatter(dfm, x='epochs', y=metric_name, color='set', trendline='ols')
    return fig

# Plot many metrics
def plot_metrics(metrics, metric_names, plot_train_metrics=['loss']):
    fig = make_subplots(rows=len(metric_names), cols=1, shared_xaxes=False)

    for i, metric_name in enumerate(metric_names):
        metric = metrics[metric_name]

        df = pd.DataFrame.from_dict(metric)
        if metric_name in plot_train_metrics:
            df = df.drop(['TRAIN'], axis=1)
        else:
            # Rearrange columns so Train is at the end, thus they keep the same colours
            cols = df.columns.to_list()
            cols.append(cols.pop(0))
            df = df[cols]
        df['index'] = df.index-1
        dfm = pd.melt(df, id_vars=['index'], var_name='set', value_name=metric_name)
        mfig = px.scatter(dfm, x='index', y=metric_name, color='set', trendline='ols')
        fig.add_trace(mfig['data'][0], row=i+1, col=1)

    # Something isn't right here. Validation is the only thing that shows up.
    # I suspect it might be something with the mfig['data'][0] line since validation is the first and only data.

    fig.show()

if __name__ == '__main__':
    ##### Parameters ######
    log_folder = sys.argv[-1] # log log_folder
    log_folder = log_folder + '/'

    logs, csv, best_log_file = parse_all_logs_in_folder(log_folder)

    for log in logs:
        print(logs[log].keys())
    #save_best_log(best_log_file)
    #write_csv(csv, log_folder)
    print(best_log_file)

    #plot_metrics(logs[best_log_file]['metrics'], ['loss', 'map'])

    #for metric in ['loss', 'map']:
    #    plot_train = metric == 'loss'
    #    fig = plot_metric(logs[best_log_file]['metrics'][metric], metric, plot_train)
    #    fig.show()
