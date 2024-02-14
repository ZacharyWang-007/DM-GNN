from argparse import Namespace
import os

from sksurv.metrics import concordance_index_censored

import torch

from utils.utils import *
from torch.optim import lr_scheduler
from datasets.dataset_generic import save_splits
from models.model_set_mil import *
from collections import defaultdict
from sklearn.metrics import roc_auc_score


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name: str = 'checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def train(datasets: tuple, cur: int, args: Namespace):
    print('\nTraining Fold {}!'.format(cur))

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError

    reg_fn = None

    print('\nInit Model...', end=' ')
    model_dict = {'n_classes': args.n_classes, 'affinity_threshold':args.affinity_threshold}
    model = MIL_Attention_FC_surv(**model_dict).cuda()

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split,
                                    training=True,
                                    testing=args.testing,
                                    weighted=args.weighted_sample,
                                    mode=args.mode,
                                    batch_size=args.batch_size)

    val_loader = get_split_loader(val_split,
                                  testing=args.testing,
                                  mode=args.mode,
                                  batch_size=args.batch_size)

    train_loss_min = 10
    c_index_small_loss, auc_small_loss = 0.0, 0.0

    for epoch in range(args.max_epochs):
        train_loss_survival = train_loop_survival(args, epoch, model, train_loader, optimizer, scheduler, loss_fn, reg_fn, args.lambda_reg, args.gc)
        c_index, auc = validate_survival(args, epoch, model, val_loader, loss_fn, reg_fn, args.lambda_reg)

        if train_loss_min > train_loss_survival:
            train_loss_min = train_loss_survival
            c_index_small_loss = c_index
            auc_small_loss = auc

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    print('Val c_index_small_loss: {:.4f} c_index_final {:.4f} auc_small_loss {:.4f} auc_final {:.4f}'.format(c_index_small_loss, c_index, auc_small_loss, auc))
    return c_index_small_loss, c_index, auc_small_loss, auc

def generate_context_map(coords):
    coords = coords[:, None, :] - coords[None, :, :]
    coords = torch.sum(torch.abs(coords), dim=-1)
    coords = (coords<4) * 1
    return coords


def train_loop_survival(args, epoch, model, loader, optimizer, scheduler, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, coords, label, event_time, c, path) in enumerate(loader):
        length = data_WSI.size(0)

        if torch.equal(data_WSI, torch.ones(1)):
            continue

        if length > args.num_instances_maximum:
            indice = random.sample(range(0, length-1), int(args.num_instances_maximum))
            data_WSI = data_WSI[indice]
            coords = coords[indice]
        
        maps = generate_context_map(coords.cuda())

        data_WSI = data_WSI.to(device)
        label = label.to(device)
        c = c.to(device)
    
        hazards_gl, S_gl, Y_hat_gl = model(epoch=epoch, x_path=data_WSI, maps=maps)
        S = S_gl

        loss = loss_fn(hazards=hazards_gl, S=S_gl, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()
    optimizer.step()
    optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    scheduler.step()

    
    c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                         all_event_times,
                                         all_risk_scores,
                                         tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch,
                                                                                                 train_loss_surv,
                                                                                                 train_loss,
                                                                                                 c_index))
    return train_loss_surv


def validate_survival(args, epoch, model, loader, loss_fn=None, reg_fn=None, lambda_reg=0.):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.eval()

    y_true, y_score = [], []
    
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    with torch.no_grad():
        for batch_idx, (data_WSI, coords, label, event_time, c, path) in enumerate(loader):
            length = data_WSI.size(0)

            # for sample selection
            if torch.equal(data_WSI, torch.ones(1)):
                continue
            
            if length > args.num_instances_maximum:
                indice = random.sample(range(0, length-1), args.num_instances_maximum)
                data_WSI = data_WSI[indice]  
                coords = coords[indice]
            
            maps = generate_context_map(coords.cuda())
            data_WSI = data_WSI.to(device)
            label = label.to(device)
            c = c.to(device)

            hazards, S, Y_hat = model(x_path=data_WSI,  maps=maps)  # return hazards, S, Y_hat, A_raw, results_dict
        
            hazards = hazards.mean(dim=0, keepdim=True)
            S = S.mean(dim=0, keepdim=True)

            if c<1:
                y_true.append((label>1) * 1)
                y_score.append(S[:, 2])
    
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
            loss_value = loss.item()

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg

            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time

            val_loss_surv += loss_value
            val_loss += loss_value + loss_reg

        val_loss_surv /= len(loader)
        val_loss /= len(loader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                            all_event_times,
                                            all_risk_scores,
                                            tied_tol=1e-08)[0]

        y_true, y_score = torch.cat(y_true).cpu().detach().numpy(), torch.cat(y_score, dim=0).cpu().detach().numpy()
        auc = roc_auc_score(y_true, y_score)

        print('val/loss_surv, {}, {}'.format(val_loss_surv, epoch))
        print('val/auc: {}, {}'.format(auc, epoch))
        print('val/c-index: {}, {}'.format(c_index, epoch))

        return c_index, auc
