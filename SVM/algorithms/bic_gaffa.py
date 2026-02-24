import cvxpy as cp
import numpy as np
import time
import copy

import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.nn import functional as F

import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from utils import load_diabetes, train_val_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def compute_accuracy(loader, model, model_theta):

    number_right = 0
    loss = 0
    loss_theta = 0
    number_right_theta = 0
    for batch_idx, (images, labels) in enumerate(loader): #val_loader
        images, labels = images.to(device), labels.to(device)
        log_probs = model(images)
        log_probs_theta = model_theta(images)
        for i in range(len(labels)):
            q=log_probs[i]*labels[i]
            if q>0:
                number_right=number_right+1
            q_theta = log_probs_theta[i]*labels[i]
            if q_theta>0:
                number_right_theta += 1
        loss += model.loss_upper(log_probs, labels)
        loss_theta += model_theta.loss_upper(log_probs, labels)
    acc=number_right/len(loader.dataset)
    acc_theta=number_right_theta/len(loader.dataset)
    loss /= len(loader.dataset)
    loss_theta /= len(loader.dataset)

    return loss, loss_theta, acc, acc_theta

class LinearSVM(nn.Module):
    def __init__(self, input_size, n_classes, n_sample):
        super(LinearSVM, self).__init__()
        # self.w = nn.Parameter(torch.ones(n_classes, input_size))
        self.w = nn.Parameter(torch.ones(input_size))  # shape (d,)
        self.b = nn.Parameter(torch.tensor(0.))
        self.C = nn.Parameter(torch.empty(n_sample))
        self.C.data.uniform_(1.,5.)
    
    def forward(self, x):
        return F.linear(x, self.w.view(1,-1), self.b)

    def loss_upper(self, y_pred, y_val):
        y_val_tensor = torch.Tensor(y_val)
        x = torch.reshape(y_val_tensor, (y_val_tensor.shape[0],1)) * y_pred / torch.linalg.norm(self.w)
        # relu = nn.LeakyReLU()
        # loss = torch.sum(relu(2*torch.sigmoid(-5.0*x)-1.0))
        loss= torch.sum(torch.exp(1-x))
        # loss += 0.5*torch.linalg.norm(self.C)**2
        return loss

    def loss_lower(self):
        w2 = 0.5*torch.linalg.norm(self.w)**2 + 0.5*torch.linalg.norm(self.b)**2
        loss =  w2# + xi_term
        loss += 0.5*torch.linalg.norm(self.C)**2
        return loss

    def constrain_values(self, srt_id, y_pred, y_train):
        xi_sidx = srt_id
        xi_eidx = srt_id+len(y_pred)
        # xi_batch = self.xi[xi_sidx:xi_eidx]
        # return 1-xi_batch-y_train.view(-1)*y_pred.view(-1)
        return 1-y_train.view(-1)*y_pred.view(-1)

    # def second_constraint_val(self):
    #     return self.xi - self.C

def bic_gaffa(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs, compute_opt=False, early_stopping_th = False, verbose=True):
    # Dataset to tensor
    y_train = torch.tensor(y_train).float()
    x_train = torch.tensor(x_train).float()
    y_val = torch.tensor(y_val).float()
    x_val = torch.tensor(x_val).float()
    y_test = torch.tensor(y_test).float()
    x_test = torch.tensor(x_test).float()
    
    batch_size = 256
    data_train = TensorDataset(
        torch.tensor(x_train), 
        torch.tensor(y_train))
    train_loader = DataLoader(
        dataset=data_train,
        batch_size=batch_size,
        shuffle=True)
    data_val = TensorDataset(
        torch.tensor(x_val), 
        torch.tensor(y_val))
    val_loader = DataLoader(
        dataset=data_val,
        batch_size=batch_size,
        shuffle=True)
    data_test = TensorDataset(
        torch.tensor(x_test), 
        torch.tensor(y_test))
    test_loader = DataLoader(
        dataset=data_test,
        batch_size=batch_size,
        shuffle=True)

    ############ Setting SVM ###########
    feature = 8
    feature=x_train.shape[1]
    N_sample = y_train.shape[0]
    
    model = LinearSVM(feature, 1, N_sample).to(device)
    # model.C.data.copy_(torch.Tensor(x_train.shape[0]).uniform_(1.0,5.0))    ####### Setting C on training data
    
    model_theta = copy.deepcopy(model)
    model_theta.C = model.C
    # model_theta.C.requires_grad = False

    ######### SVM variables
    lamda = torch.ones(N_sample) #+ 1./N_sample
    z = torch.ones(N_sample) #+ 1./N_sample

    params = [p for n, p in model.named_parameters()]
    params_theta = [p for n, p in model_theta.named_parameters() if n != 'C']

    ############ Decode BiC-GAFFA parameters
    alpha  = hparams['alpha']    # step-size for (x, y, z) updates
    yita   = hparams['yita']     # step-size for θ-update (inner projected gradient)
    gama1  = hparams['gama1']    # proximal term for θ - y
    gama2  = hparams['gama2']    # proximal term for λ - z

    
    #epochs = 80
    algorithm_start_time=time.time()

    variables = []
    metrics = []
    
    ############ algorithm ###########    
    for epoch in range(epochs):
        #### Storage of variables and metrics
        vars_to_append = {
            # Upper-level
            'C': model.C.data.clone(),
        
            # Lower-level theta variables (BiC-GAFFA updates)
            'theta_w': model_theta.w.data.clone(),
            'theta_b': model_theta.b.data.clone(),
        
            # Lower-level primal variables (optional, for monitoring)
            'w': model.w.data.clone(),
            'b': model.b.data.clone(),
        
            # Dual / auxiliary variables
            'lambda': lamda.clone(),
            'z': z.clone()
        }

        variables.append(vars_to_append)

        with torch.no_grad():

            train_loss, train_loss_theta, train_acc, train_acc_theta = compute_accuracy(train_loader, model, model_theta)
            val_loss, val_loss_theta, val_acc, val_acc_theta = compute_accuracy(val_loader, model, model_theta)
            test_loss, test_loss_theta, test_acc, test_acc_theta = compute_accuracy(test_loader, model, model_theta)

            # recompute val/test loss with SVM exponential form
            def svm_exp_loss(X, y, w, b):
                logits = F.linear(X, w.view(1,-1), b)
                return torch.sum(torch.exp(1 - y.view(-1,1) * logits))
            val_loss = svm_exp_loss(torch.Tensor(x_val), torch.Tensor(y_val), model.w.data, model.b.data)
            test_loss = svm_exp_loss(torch.Tensor(x_test), torch.Tensor(y_test), model.w.data, model.b.data)


        metrics.append({
            'train_loss': train_loss,
            'train_loss_theta': train_loss_theta.detach().numpy(),
            'train_acc': train_acc,
            'train_acc_theta': train_acc_theta,
            'val_loss': val_loss,
            'val_loss_theta': val_loss_theta.detach().numpy(),
            'val_acc': val_acc,
            'val_acc_theta': val_acc_theta,
            'test_loss': test_loss,
            'test_loss_theta': test_loss_theta.detach().numpy(),
            'test_acc': test_acc,
            'test_acc_theta': test_acc_theta,
            'loss_lower': (0.5*torch.linalg.norm(model.w.detach())**2).numpy(),
            'time_computation': time.time()-algorithm_start_time
        })

        ### Start of the algorithm
        ck = 1.0/((epoch+1.)**0.5)

        ################## Lower-Level Updates
        model_theta.zero_grad()
        loss_theta = model_theta.loss_lower()
        margins = 1-y_train * (x_train @ model_theta.w + model_theta.b)-model_theta.C  # (N,)
        loss_theta+=torch.dot(z, margins)   # scalar
        loss_theta += 1/gama1*0.5*(torch.norm(model_theta.w-model.w.detach())**2+(model_theta.b-model.b.detach())**2)
        loss_theta.backward()  # gradient w.r.t theta
        
        with torch.no_grad():  # prevent tracking in autograd
            ######## theta_k+1
            for param in params_theta:
                if param.requires_grad:
                    param -= yita * param.grad

            ######## lambdak+1
            margins = 1-y_train * (x_train @ model.w + model.b)-model.C 
            lamda+=gama2*margins
            lamda = torch.clamp(lamda, min=0.0)

        ################## Upper-Level Updates (x_k+1, y_k+1, z_k+1)
        model.zero_grad()
        loss = model.loss_lower() 
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            batch_loss = model.loss_upper(logits, labels) # LL not involve x
            loss += 1/ck * batch_loss
        margins = 1-y_train * (x_train @ model.w + model.b)-model.C 
        loss +=torch.dot(lamda, margins)  
        loss-=1/gama1*0.5*(torch.norm(model_theta.w.detach()-model.w)**2+(model_theta.b.detach()-model.b)**2)
        loss.backward()  # gradient w.r.t theta
        with torch.no_grad():  # prevent tracking in autograd
            ######## z_k+1
            margins = 1-y_train * (x_train @ model_theta.w + model_theta.b)-model_theta.C  # (N,)
            z -= alpha*(-(z-lamda)/gama2-margins)
            z = torch.clamp(z, min=0.0)
            
            ######## x_k+1
            model.C-= alpha*z
            ######## y_k+1
            for param in params:
                if param.requires_grad:
                    param -= alpha * param.grad
        
        # if epoch%20==0 and verbose:
        #     print("val acc: {:.2f}".format(val_acc),
        #       "val loss: {:.2f}".format(val_loss),
        #       "test acc: {:.2f}".format(test_acc),
        #       "test loss: {:.2f}".format(test_loss),
        #       "round: {}".format(epoch))
            
        # if torch.linalg.norm(d_C) < early_stopping_th:
        #     break

    return metrics, variables


if __name__ == "__main__":
    ############ Load data code ###########

    data = load_diabetes()

    n_train = 500
    n_val = 150

    metrics = []
    variables = []

    hparams = {
        'alpha': 0.01,
        'gama1': 0.1,
        'gama2': 0.1,
        'yita': 0.001
    }

    epochs = 80
    plot_results = True

    for seed in range(10):

        x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(data, seed, n_train, n_val)

        metrics_seed, variables_seed = bic_gaffa(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs)
        metrics.append(metrics_seed)
        variables_seed.append(variables_seed)

    train_acc = np.array([[x['train_acc'] for x in metric] for metric in metrics])
    val_acc = np.array([[x['val_acc'] for x in metric] for metric in metrics])
    test_acc = np.array([[x['test_acc'] for x in metric] for metric in metrics])

    val_loss = np.array([[x['val_loss'] for x in metric] for metric in metrics])
    test_loss = np.array([[x['test_loss'] for x in metric] for metric in metrics])

    time_computation = np.array([[x['time_computation'] for x in metric] for metric in metrics])

    if plot_results:
        val_loss_mean=np.mean(val_loss,axis=0)
        val_loss_sd=np.std(val_loss,axis=0)/2.0
        test_loss_mean=np.mean(test_loss,axis=0)
        test_loss_sd=np.std(test_loss,axis=0)/2.0

        val_acc_mean=np.mean(val_acc,axis=0)
        val_acc_sd=np.std(val_acc,axis=0)/2.0
        test_acc_mean=np.mean(test_acc,axis=0)
        test_acc_sd=np.std(test_acc,axis=0)/2.0

        axis = np.mean(time_computation,axis=0)

        plt.rcParams.update({'font.size': 18})
        plt.rcParams['font.sans-serif']=['Arial']#如果要显示中文字体，则在此处设为：SimHei
        plt.rcParams['axes.unicode_minus']=False #显示负号
        axis=time_computation.mean(0)
        plt.figure(figsize=(8,6))
        #plt.grid(linestyle = "--") #设置背景网格线为虚线
        ax = plt.gca()
        plt.plot(axis,val_loss_mean,'-',label="Training loss")
        ax.fill_between(axis,val_loss_mean-val_loss_sd,val_loss_mean+val_loss_sd,alpha=0.2)
        plt.plot(axis,test_loss_mean,'--',label="Test loss")
        ax.fill_between(axis,test_loss_mean-test_loss_sd,test_loss_mean+test_loss_sd,alpha=0.2)
        #plt.xticks(np.arange(0,iterations,40))
        plt.title('Kernelized SVM')
        plt.xlabel('Running time /s')
        #plt.legend(loc=4)
        plt.ylabel("Loss")
        #plt.xlim(-0.5,3.5)
        #plt.ylim(0.5,1.0)
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        #plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
        plt.savefig('ho_svm_kernel_1.pdf') 
        #plt.show()

        plt.figure(figsize=(8,6))
        ax = plt.gca()
        plt.plot(axis,val_acc_mean,'-',label="Training accuracy")
        ax.fill_between(axis,val_acc_mean-val_acc_sd,val_acc_mean+val_acc_sd,alpha=0.2)
        plt.plot(axis,test_acc_mean,'--',label="Test accuracy")
        ax.fill_between(axis,test_acc_mean-test_acc_sd,test_acc_mean+test_acc_sd,alpha=0.2) 
        #plt.xticks(np.arange(0,iterations,40))
        plt.title('Kernelized SVM')
        plt.xlabel('Running time /s')
        plt.ylabel("Accuracy")
        # plt.ylim(0.64,0.8)
        #plt.legend(loc=4)
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        #plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
        plt.savefig('ho_svm_kernel_2.pdf') 
        plt.show()
