#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import sklearn
import pickle
import torch.optim as optim
from cnn import ResNet9
import collections
from tqdm import tqdm
from batch_retrain import *
import pretrain
from sklearn import metrics
import os

import matplotlib.pyplot as plt

def models_parameter_norm(model):
    for k, v in model.named_parameters():
        print(k , torch.linalg.norm(v).detach().numpy())

def draw_histogram(inp, title_, xlabel_, ylabel_, save_path, err_bar = False):
    plt.figure(figsize=(1800/300,1200/300),dpi=300)
    inp_mean, inp_std = np.mean(inp, axis=0), np.std(inp, axis=0)
    # print(inp_mean)
    plt.figure()
    if err_bar:
        plt.bar(range(inp_mean.shape[0]), inp_mean, yerr = inp_std)
    else:
        plt.bar(range(inp_mean.shape[0]), inp_mean)
    plt.xticks(ticks = [2-1, 5-1, 8-1, inp.shape[1]-1 ], labels=[2, 5, 8, inp.shape[1] ])
    plt.title(title_, fontsize=20)
    plt.xlabel(xlabel_, fontsize=18)
    plt.ylabel(ylabel_, fontsize=18)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':
    RANDOM_SEED = 24
    torch.random.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    const_type = 'early_stop'
    data_name = 'cifar10'
    adapt_mode = '_adapt'   # '_adapt'
    eval_method = 'exp'
    topk = 100
    atk_method = 'topk'
    
    on_gpu = torch.cuda.is_available()
    if on_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with open('./data/' + data_name + '_portion_data', 'rb') as f:
        dataset = pickle.load(f)
    train_loader = dataset['train']
    valid_loader = dataset['val']
    test_loader = dataset['test']
    exp_loader = dataset['exp']

    
    topk_pair = [(i, topk-1) for i in range(topk-1)] + [(topk-1, topk+i) for i in range(topk)]
    atk_epoch = 100
    atk_epsilon = 5e-3
    lr_lambda = 1e-4
    lr_eta=1e-4
    MODES = ['pre']
    cls_flag = True

    pretrain_model_PATH = './model/' + data_name + '_pretrain_model.pth'
    if on_gpu:
        pretrain_model = torch.load(pretrain_model_PATH)
        pretrain_model.to(device)
    else:
        pretrain_model = torch.load(pretrain_model_PATH, map_location=torch.device('cpu'))
    pretrain_model.eval()
    results = collections.defaultdict(list)
    

    %%
    for mode in MODES:
        mix_weight = str(weights[mode]) if len(mode.split(','))==1 else str([weights[m] for m in mode.split(',')])
        # print(mode)
        # for epoch in range(10):
        for epoch in range(1): # for models in /model/opt
            # print(mode, epoch)
            if mode == 'pre':
                model_PATH = './model/' + data_name + act_prefix + '_pretrain_model.pth'
            else:
                model_PATH = './model/opt/' + data_name + act_prefix + start_from + '_model_' + mode + '_' + str(cls_flag) +\
                              '_top' + str(topk) + adapt_mode + '_w_' + mix_weight + '.pth'
                              
            model = torch.load(model_PATH)
            model.eval()
        
            results[mode + str(epoch) + '_acc'] = pretrain.test(model, test_loader)
            
            flip, overlap, norm_diff, pcc, pre_overlap, adv_prob = [], [], [], [], [], []
            
            # for inp_i in tqdm(range(X_eval.shape[0])):
            for inp_i in range(X_eval.shape[0]):
            # for inp_i in [0]:
                inp, inp_y = X_eval[inp_i], Y_eval[inp_i]
                # inp, inp_y = X_train[inp_i], Y_train[inp_i]
                inp_v = Variable(inp.data, requires_grad=True)
                grads = get_gradient(pretrain_model, inp_v)
                sorted_grads, indices = rank_gradient(grads)
                new_grads = get_gradient(model, inp_v)
                _, new_indices = rank_gradient(new_grads)
                
                cnt_topk = topk_overlap(indices, new_indices, topk)
                cnt_norm, cnt_eps, cnt_Tx, cnt_eTx, cnt_deno = 0, 0, 0, 0, 0
                
            #    'stat eval'
                for idx1, idx2 in topk_pair:
                    
                    norm_H, eps, Tx, eTx, deno = compute_Hessian_epsilon(model, inp, indices[idx1], indices[idx2])
                    cnt_norm += norm_H
                    cnt_eps += eps
                    cnt_Tx += Tx
                    cnt_eTx += eTx
                    cnt_deno += deno
                    
                    results[mode + str(epoch) + '_Hnorm'].append(cnt_norm)
                    results[mode + str(epoch) + '_eps'].append(cnt_eps)
                    results[mode + str(epoch) + '_tx'].append(cnt_Tx)
                    results[mode + str(epoch) + '_etx'].append(cnt_eTx)
                    results[mode + str(epoch) + '_deno'].append(cnt_deno)
                    results[mode + str(epoch) + '_topk'].append(cnt_topk)
                
            print(results[mode + str(epoch) + '_AUC'],
                  np.mean(results[mode + str(epoch) + '_topk']))
                    
            
    # %%
    # # # PGD
    model = ResNet9()
    for mode in MODES:
        for weight in [0.001, 0.01, 0.1, 1, 10, 100]:
            norm_weight = weight
            weights = {}
            for m in MODES:
                weights.update({k: weight for k in m.split(',')})
            weights.update({'pre': 1, 'none': 1, 'norm':norm_weight, 'est_norm':norm_weight})
            
            mix_weight = str(weights[mode]) if len(mode.split(','))==1 else str([weights[m] for m in mode.split(',')])
            # print(data_name, topk, mode, mix_weight, eval_method, adapt_mode, atk_method)
            if mode == 'pre':
                model_PATH = './model/' + data_name + '_pretrain_model.pth'
            elif mode in ['at', 'est_norm'] or mode[:2] in ['wd', 'sp']:
                model_PATH = './model/' + data_name + '_model_' + mode + '_' + str(cls_flag) + '_w_1.pth'
            else:
                model_PATH = './model/' + data_name + '_model_' + mode + '_' + str(cls_flag) +\
                        '_top' + str(topk) + adapt_mode + '_w_' + mix_weight + '.pth'
            
            if os.path.exists(model_PATH):
                if on_gpu:
                    # model.load_state_dict(torch.load(model_PATH))
                    model = torch.load(model_PATH)
                    # model.to(device)
                else:
                    model = torch.load(model_PATH, map_location=torch.device('cpu'))
                model.to(device)
            else:
                continue
    
            model.eval()
        
            flip, overlap, norm_diff, pcc, pre_overlap, adv_prob = [], [], [], [], [], []
            atk_inps, atk_grads, atk_retrain_gradients = [], [], []
            for x, y in tqdm(exp_loader):
                y = torch.tensor(y)
                if on_gpu:
                    x, y = x.cuda(), y.cuda()
                inp, inp_y = x.unsqueeze(dim=0), y.unsqueeze(dim=0)
    
                pred_flip, topk_eval, norm_eval, pcc_eval, pre_topk_eval, adv_pred, _, atk_grad, atk_inp, re_grads = PGD_m(model, inp, inp_y, atk_epoch, atk_method,
                                                                                                                           topk_pair, topk, device, const_type, epsilon=atk_epsilon,
                                                                                                                           lr_lambda = lr_lambda, lr_eta=lr_eta)
                if const_type == 'early_stop':
                    flip.append(pred_flip[-1])
                    pre_overlap.append(pre_topk_eval[-1])
                else:
                    flip.append(pred_flip)
                    # overlap.append(topk_eval)
                    norm_diff.append(norm_eval)
                    pcc.append(pcc_eval)
                    pre_overlap.append(pre_topk_eval)
                    adv_prob.append(adv_pred)
                    atk_grads.append(atk_grad)
                    atk_inps.append(atk_inp)
                    atk_retrain_gradients.append(re_grads)
                
                

            flip_aver = np.mean(np.array(flip), axis=0)
            # overlap_aver = np.mean(np.array(overlap), axis=0)
            pre_overlap_aver = np.mean(np.array(pre_overlap), axis=0)
            norm_aver = np.mean(np.array(norm_diff), axis=0)
            pcc_aver = np.mean(np.array(pcc), axis=0)
            
            if const_type == 'early_stop':
                print(topk, mix_weight, mode, const_type, adapt_mode, atk_method, pre_overlap_aver, flip_aver)
            else:
                print(topk, mix_weight, mode, const_type, str(lr_lambda), str(lr_eta), adapt_mode, atk_method, pre_overlap_aver[-1], flip_aver[-1])
            