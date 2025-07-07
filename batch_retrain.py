import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pickle
import torch.optim as optim
from cnn import ResNet9
import collections
from tqdm import tqdm
from scipy.stats import pearsonr
import sklearn
import pretrain
import torch.nn.functional as F

def replace_relu_to_sp(model, beta=0.5):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Softplus(beta=beta))
        else:
            replace_relu_to_sp(child, beta)

def get_gradient(model, inp_v, flag=False):
    output = model(inp_v)
    grad = torch.autograd.grad(output[:, output.argmax()], inp_v, create_graph=flag, retain_graph=flag)[0]
    return grad
    

def get_batch_gradient(model, inp_v, inp_y, flag=False):
    loss = nn.CrossEntropyLoss()(model(inp_v), inp_y)
    grad = torch.autograd.grad(loss, inp_v, create_graph=flag, retain_graph=flag)[0]
    return grad

def rank_gradient(grads, descend = True):
    flatten_grad = torch.flatten(torch.sum(torch.abs(grads).squeeze(), dim=1))
    sorted_grad, indice = torch.sort(flatten_grad, descending=descend)
    return sorted_grad, indice

def rank_batch_gradient(grads, descend=True):
    flatten_grad = torch.flatten(torch.sum(torch.abs(grads), dim=1), start_dim=1, end_dim=2)
    sorted_grad, indice = torch.sort(flatten_grad, descending=descend)
    return sorted_grad, indice


def topk_overlap(tens1, tens2, topk):
    set1 = set(tens1.cpu().numpy()[:topk])
    set2 = set(tens2.cpu().numpy()[:topk])
    inter = set1.intersection(set2)
    return len(inter) / topk

def topk_batch_overlap(tens1, tens2, topk):
    topk1 = tens1.cpu().numpy()[:, :topk]
    topk2 = tens2.cpu().numpy()[:, :topk]
    inter = [len(set(topk1[idx]).intersection(set(topk2[idx]))) for idx in range(tens1.shape[0])]
    return np.mean(inter) / topk

def compute_T(grad1, grad2):
    T = torch.abs(grad1) - torch.abs(grad2)
    return T

def compute_denominator(Hessian, grad1, grad2, idx1, idx2):
    sign1 = torch.sign(grad1.detach())
    sign2 = torch.sign(grad2.detach())
    denominator = torch.norm(Hessian[idx1] - Hessian[idx2] * sign1 * sign2)
    return denominator

def estimate_one_axis(model, inp, idx, h):
    inp_v = Variable(inp.data, requires_grad=True)
    p = torch.ones(inp_v.shape) * h
    inp_l = inp_v - p
    inp_r = inp_v + p
    grads_l = get_gradient(model, inp_l, True)
    grads_r = get_gradient(model, inp_r, True)
    est = (grads_r - grads_l) / (2* torch.linalg.norm(p) )
    return est

def estimate_Hessian(model, inp, inp_y, h):
    inp_v = Variable(inp.data, requires_grad=True)
    grad_v = get_batch_gradient(model, inp_v, inp_y, True)
    g_sign = torch.sign(grad_v)
    inp_p = inp_v + h * g_sign / 3072
    grad_p = get_batch_gradient(model, inp_p, inp_y, True)
    est = (grad_p - grad_v) / h
    return torch.linalg.norm(est)
    
def estimate_denominator(model, inp, idx1, idx2, grad1, grad2, h):
    est1 = estimate_one_axis(model, inp, idx1, h)
    est2 = estimate_one_axis(model, inp, idx2, h)
    sign1 = torch.sign(grad1.detach())
    sign2 = torch.sign(grad2.detach())
    return torch.linalg.norm(est1 - est2 * sign1 * sign2 + h)
    

def PGD_m(model, inp, inp_y, epochs, atk_method, topk_pair, topk, device, const_type='non', epsilon=1e-3, lr_lambda = 1e-4, lr_eta=1e-4):
    def p(x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost


    atk_inp = torch.clone(inp)
    model.eval()
    
    inp_v = Variable(atk_inp.data, requires_grad=True)
    gt_grad = get_batch_gradient(model, inp_v, inp_y).detach()
    _, gt_indices = rank_batch_gradient(gt_grad)
    before_atk_pred = model(inp).detach()
    
    hinge = nn.MarginRankingLoss(reduction='none')
    kl_div = nn.KLDivLoss(reduction='batchmean')
    
    if const_type == 'early_stop':
        flip, overlap, normeval, pcceval, pre_overlap, adv_prob = [1], [1], [], [], [], []
    else:
        flip, overlap, normeval, pcceval, pre_overlap, adv_prob = [], [], [], [], [], []
    atk_grads, atk_inps, atk_retrain_gradients = [], [], []
    
    lambdas = torch.Tensor([1, 1]).to(device)
    
    eta = torch.Tensor([lr_eta]).to(device)
    for it in range(epochs):
        # loss = 0
        inp_v = Variable(atk_inp.data, requires_grad=True)
        re_grads = get_batch_gradient(model, inp_v, inp_y, True)
        flatten_re_grad = torch.flatten(torch.sum(torch.abs(re_grads).squeeze(), dim=0))
        
        if atk_method == 'topk':
            top_idx = [i[0] for i in topk_pair]
            bottom_idx = [i[1] for i in topk_pair]
            g0 = torch.sum(torch.abs(flatten_re_grad[gt_indices[:, top_idx]])) - torch.sum(torch.abs(flatten_re_grad[gt_indices[:, bottom_idx]]))
        elif atk_method == 'norm':
            g0 = - torch.linalg.norm(re_grads - gt_grad + 1e-2)
        
        if const_type != 'non' and const_type != 'early_stop':
            atk_pred = model(atk_inp)
            # symmetric KL
            # g1 = kl_div(F.log_softmax(atk_pred, dim=1), F.softmax(before_atk_pred, dim=1)) + kl_div(F.log_softmax(before_atk_pred, dim=1), F.softmax(atk_pred, dim=1)).view(1)
            g1 = torch.norm(F.softmax(before_atk_pred, dim=1) - F.softmax(atk_pred, dim=1)).view(1)
            g1 = hinge(torch.zeros(1).to(device), g1.to(device), torch.ones(1).to(device))
            g_final = torch.cat((g0.view(1), g1))
            
            if const_type == 'GDA':
                lambdas = lambdas + torch.tensor([1, lr_lambda]).to(device) * g_final
                lambdas = lambdas / torch.sum(lambdas)
                loss = torch.dot(lambdas.detach(), g_final)
            elif const_type == 'hedge':
                lambdas = lambdas / torch.sum(lambdas)
                loss = torch.dot(lambdas.detach(), g_final)
                lambdas = lambdas * torch.exp(eta * g_final)
        else:
            loss = g0
            
        atk_grad = torch.autograd.grad(loss, inp_v, create_graph=True, retain_graph=True)[0]
        atk_inp -= epsilon * atk_grad / (torch.linalg.norm(atk_grad) + 1e-10)
        # atk_inp = torch.clamp(atk_inp, inp-epsilon*epochs, inp+epsilon*epochs)
        atk_inp = torch.min(torch.max(atk_inp, inp-epsilon*epochs), inp+epsilon*epochs)
        # if it >= 980:
        #     print(atk_inp-inp)
        
        # calculate atk performance
        inp_v = Variable(atk_inp.data, requires_grad=True)
        re_grads = get_batch_gradient(model, inp_v, inp_y)
        _, re_indices = rank_batch_gradient(re_grads)
        # atk_topk = topk_overlap(re_indices, indices, topk)
        pre_topk = topk_batch_overlap(re_indices, gt_indices, topk)
        # if it >= 980:
        #     # print(re_indices, gt_indices, pre_topk)
        #     print(re_grads, pre_topk)
        atk_pred = model(atk_inp)
        norm_diff = torch.linalg.norm(re_grads - gt_grad)
        pcc = p(gt_grad.detach(), re_grads.detach())
        if const_type == 'early_stop' and int(torch.argmax(atk_pred)==torch.argmax(before_atk_pred)) != 1:
            break
        flip.append(int(torch.argmax(atk_pred) == torch.argmax(before_atk_pred)))
        # overlap.append(atk_topk)
        pre_overlap.append(pre_topk)
        normeval.append(norm_diff.detach().cpu())
        pcceval.append(pcc.detach().cpu())
        adv_prob.append(atk_pred.detach().cpu().numpy())
        atk_grads.append(atk_grad.detach().cpu().numpy())
        atk_inps.append(atk_inp.clone().data.cpu().numpy())
        atk_retrain_gradients.append(re_grads.detach().cpu().numpy())


    return flip, overlap, normeval, pcceval, pre_overlap, adv_prob, atk_inp.detach(), atk_grads, atk_inps, atk_retrain_gradients


def retrain_regularizer(model, inp, inp_y, topk_pair, penalty_mode, indices, topk):
    if penalty_mode == 'none' or penalty_mode[:2] == 'sp' or penalty_mode[:2] == 'wd' or penalty_mode == 'at':
        return 0
    inp_v = Variable(inp.data, requires_grad=True)
    grads = get_batch_gradient(model, inp_v, inp_y, True)
    if penalty_mode == 'est_norm':
        penalty = estimate_Hessian(model, inp, inp_y, h=1e-3)
    elif penalty_mode[:5] == 'tx_mm':
        n = int(penalty_mode[5:])
        top_idx = indices[:, :topk]
        bottom_idx = indices[:, topk:]
        ranked_grads, _ = rank_batch_gradient(grads)
        top_sort_grad = torch.gather(ranked_grads, dim=1, index=top_idx)
        bottom_sort_grad = torch.gather(ranked_grads, dim=1, index=bottom_idx)
        penalty = torch.sum(bottom_sort_grad[:n]) - torch.sum(top_sort_grad[-n:])
    elif penalty_mode == 'etxmm':
        top_idx = indices[:, :topk]
        bottom_idx = indices[:, topk:]
        ranked_grads, _ = rank_batch_gradient(grads)
        top_sort_grad = torch.gather(ranked_grads, dim=1, index=top_idx)
        bottom_sort_grad = torch.gather(ranked_grads, dim=1, index=bottom_idx)
        penalty = torch.sum(torch.exp(bottom_sort_grad[:topk])) - torch.sum(torch.exp(top_sort_grad[-topk:]))
    else:
        if penalty_mode[:2] == 'tx' and penalty_mode != 'tx':
            n = int(penalty_mode[2:])
            topk_pair = [(i, topk-1) for i in range(max(0, topk-n-1), topk-1)] + [(topk-1, topk+i) for i in range(n)]
        elif penalty_mode[:3] == 'etx' and penalty_mode != 'etx':
            n = int(penalty_mode[3:])
            topk_pair = [(i, topk-1) for i in range(max(0, topk-n-1), topk-1)] + [(topk-1, topk+i) for i in range(n)]
        
        p_top_idx = torch.tensor([[idx[0] for idx in topk_pair] for _ in range(indices.shape[0]) ]).to(device)
        p_bottom_idx = torch.tensor([[idx[1] for idx in topk_pair] for _ in range(indices.shape[0]) ]).to(device)
        top_idx = torch.gather(indices, dim=1, index=p_top_idx)
        bottom_idx = torch.gather(indices, dim=1, index=p_bottom_idx)
        flatten_grad = torch.flatten(torch.sum(torch.abs(grads), dim=1), start_dim=1, end_dim=2)
        top_sort_grad = torch.gather(flatten_grad, dim=1, index=top_idx)
        bottom_sort_grad = torch.gather(flatten_grad, dim=1, index=bottom_idx)
        
        if penalty_mode[:2] == 'tx':
            penalty = torch.sum(bottom_sort_grad - top_sort_grad)
        elif penalty_mode[:3] == 'etx':
            penalty = torch.sum(torch.exp(bottom_sort_grad - top_sort_grad))
            
    return penalty



def retrain(pretrain_model_PATH, epochs, penalty_mode, weights, cls_flag, topk, topk_pair, adapt_mode):
    loss_f = nn.CrossEntropyLoss()
    # loss_f = nn.NLLLoss()
    
    pretrain_model = ResNet9()
    model = ResNet9()
    
    if on_gpu:
        pretrain_model = torch.load(pretrain_model_PATH)
        model = torch.load(pretrain_model_PATH)
        pretrain_model.to(device)
        model.to(device)
        pretrain_model.eval()
    else:
        pretrain_model = torch.load(pretrain_model_PATH, map_location=torch.device('cpu'))
        model = torch.load(pretrain_model_PATH, map_location=torch.device('cpu'))
        pretrain_model.eval()
    if penalty_mode == 'pre':
        return pretrain_model

    if penalty_mode[:2] == 'wd':
        opt = optim.Adam(model.parameters(), lr=1e-5, weight_decay=float(penalty_mode[2:]))
    else:
        opt = optim.Adam(model.parameters(), lr=1e-5)
    
    if penalty_mode[:2] == 'sp':
        beta = float(penalty_mode.split(',')[0][2:])
        replace_relu_to_sp(model, beta)
    
    for epoch in tqdm(range(epochs)):
        model.train()
        for inps, inp_ys in train_loader:
            loss = 0
            if on_gpu:
                inps, inp_ys = inps.cuda(), inp_ys.cuda()
                
            inp_v = Variable(inps.data, requires_grad=True)
            
            grads = get_batch_gradient(model, inp_v, inp_ys, False)
            sorted_grads, indices = rank_batch_gradient(grads)
            regularizer = 0
            
            if penalty_mode == 'at':
                cls_loss = 0
                for x, y in zip(inps, inp_ys):
                    inp, inp_y = x.unsqueeze(dim=0), y.unsqueeze(dim=0)
                    _, _, _, _, _, _, perturb_inp, _, _, _ = PGD_m(model, inp, inp_y, 1, 'topk', topk_pair, topk, device, 'non', epsilon=1)
                    ypred = model(perturb_inp.detach())
                    cls_loss += loss_f(ypred, inp_y)
            else:
                ypred = model(inps.detach())
                cls_loss = loss_f(ypred, inp_ys)
            
            for mode in penalty_mode.split(','):
                if mode[:2] == 'sp':
                    continue
                
                retrain_reg = retrain_regularizer(model, inps, inp_ys, topk_pair, mode, indices, topk)
                if adapt_mode == '_adapt' and mode[:2] != 'wd' and mode != 'at' and mode[:2] != 'sp':
                    weight = torch.abs(weights[mode] * (cls_loss.detach()+1) / retrain_reg.detach())
                else:
                    weight = weights[mode]
                regularizer += weight * retrain_reg
            
            if cls_flag:
                loss += (cls_loss + regularizer)
            else:
                loss += regularizer
                
            opt.zero_grad()
            loss.backward()
            opt.step()
        mix_weight = str(weights[penalty_mode]) if len(penalty_mode.split(','))==1 else str([weights[mode] for mode in penalty_mode.split(',')])
        torch.save(model, './model/' + data_name + '_model_' + penalty_mode + '_' + str(cls_flag) +\
                   '_top' + str(topk) + adapt_mode + '_w_' + mix_weight + '.pth')
        
        
        # eval
        model.eval()
        val_acc = pretrain.test(model, val_loader, on_gpu)
        if val_acc < THRESHOLD:
            continue
        
        eval_topk = []
        for inps, inp_ys in val_loader:
            if on_gpu:
                inps, inp_ys = inps.cuda(), inp_ys.cuda()
            inp_v = Variable(inps.data, requires_grad=True)
            grads = get_batch_gradient(pretrain_model, inp_v, inp_ys, True)
            sorted_grads, indices = rank_batch_gradient(grads)
            new_grads = get_batch_gradient(model, inp_v, inp_ys, True)
            _, new_indices = rank_batch_gradient(new_grads)
        
            eval_topk.append(topk_batch_overlap(indices, new_indices, topk))
        # print(val_acc, np.mean(eval_topk))
        print(epoch, val_acc, np.mean(eval_topk))
        # if val_acc >= THRESHOLD and np.mean(eval_topk) >= 0.9:
        if val_acc >= THRESHOLD:
            if penalty_mode in ['at', 'est_norm'] or penalty_mode[:2] in ['wd', 'sp']:
                torch.save(model, './model/' + data_name + '_model_' + penalty_mode + '_' + str(cls_flag) +\
                           adapt_mode + '_w_' + mix_weight + '.pth')
            else:
                torch.save(model, './model/' + data_name + '_model_' + penalty_mode + '_' + str(cls_flag) +\
                           '_top' + str(topk) + adapt_mode + '_w_' + mix_weight + '.pth')
        # torch.save(model, './model/' + data_name + '_model_' + penalty_mode + '_' + str(cls_flag) +\
        #             '_top' + str(topk) + adapt_mode + '_w_' + mix_weight + '_' + str(epoch) + '.pth')
                
    return model


if __name__ == '__main__':
    RANDOM_SEED = 24
    torch.random.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    THRESHOLD = 86
    retrain_epoch = 10    # 10 for pretrain, 300 for rand
    on_gpu = torch.cuda.is_available()
    if on_gpu:
        device = torch.device("cuda")

    
    data_name = 'cifar10'
    with open('./data/' + data_name + '_portion_data', 'rb') as f:
        dataset = pickle.load(f)
    train_loader = dataset['train']
    val_loader = dataset['val']
    # X_test, Y_test = dataset['X_test'], dataset['Y_test']
    
    "initialize the model"
    pretrain_model_PATH = './model/' + data_name + '_pretrain_model.pth'
    
    adapt_mode = '_adapt' #'_adapt'
    
    topk = 100
    MODES = ['sp0.5,est_norm,tx']
     # ['norm', 'est_norm', 'tx', 'etx', 'tx', 'etxmm']

    for weight in [0.001, 0.01, 0.1, 1, 10, 100]:
        norm_weight = weight
        cls_loss = True
        weights = {}
        for mode in MODES:
            weights.update({k: weight for k in mode.split(',')})
        weights.update({'pre': 1, 'none': 1, 'norm':norm_weight, 'est_norm':norm_weight})
        topk_pair = [(i, topk-1) for i in range(topk-1)] + [(topk-1, topk+i) for i in range(topk)]
        
        for mode in MODES:
            # if weight ==0.001 and mode in ['tx1','tx2']:
            #     continue
            print(data_name, topk, weight, mode, adapt_mode)
            retrain_model = retrain(pretrain_model_PATH, retrain_epoch, mode, weights, cls_loss, topk, topk_pair, adapt_mode)
            
