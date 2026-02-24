import math
import os
from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
# from torch.utils.data import DistributedSampler
from bipost.utils.distributed_sampler import DistributedSampler
from tqdm import tqdm
from bipost.models import Actor
from bipost.utils.deepspeed_utils import freeze_model

from bipost.models import DPOLoss, GPTLMLoss, KDLoss
import time
import math
import datetime
# for recording gpu memory used
import subprocess as sp
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
from torch.nn import functional as F
from flash_attn.utils.distributed import all_gather


class BiLevelTrainer(ABC):
    """
        Trainer for optimizing two objectives in an alternating manner
    """

    def __init__(
        self,
        model,
        strategy,
        tokenizer,
        optim: Optimizer, # Optimizer 1
        # optim2: Optimizer, # Optimizer 2 
        train_dataloader_1,
        train_dataloader_2,
        train_dataloader_2_2,
        eval_dataloader_1,
        eval_dataloader_2,
        scheduler,
        # scheduler2,
        model_param1 = ['lora'],
        model_param2 = ["lm_head","embed_out"],
        lambd=0.5,
        ref_pareto_front=[[0.0, 0.0]],
        ref_model=None,
        ref_model_2=None,
        gam = 10, # penalty constant
        max_training_time = 7200,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = strategy.args.max_epochs
        self.max_norm = strategy.args.max_norm
        self.model = model
        self.model_param1=model_param1 # name list of UL paramter
        self.model_param2=model_param2 # name list of LL paramter
        self.ref_model = ref_model
        self.ref_model_2 = ref_model_2
        self.train_dataloader_1 = train_dataloader_1
        self.train_dataloader_2 = train_dataloader_2
        self.train_dataloader_2_2 = train_dataloader_2_2 # LL SFT other training dataset
        self.eval_dataloader_1 = eval_dataloader_1
        self.eval_dataloader_2 = eval_dataloader_2
        self.scheduler = scheduler # UL optimizer schedualer
        # self.scheduler2 = scheduler2 # LL optimizer schedualer
        self.optimizer = optim # UL optimizer
        # self.optimizer2 = optim2 # LL optimizer
        self.tokenizer = tokenizer
        self.args = strategy.args

        self.gam = gam # penalty constant

        self.train_loss1_records = [[] for _ in range(strategy.args.max_epochs)]
        self.train_loss2_records = [[] for _ in range(strategy.args.max_epochs)]

        self.eval_loss1_records = [[] for _ in range(strategy.args.max_epochs)]
        self.eval_loss2_records = [[] for _ in range(strategy.args.max_epochs)]

        self.time_elapsed = [[] for _ in range(strategy.args.max_epochs)]
        
        # regulate the max training time 
        self.max_training_time = max_training_time
        # define loss functions
        self.loss_fn_1 = self.get_loss_fn(obj_index=1)
        self.loss_fn_2 = self.get_loss_fn(obj_index=2)

        self.lambd=lambd # this is for bi objective NOT BI LEVEL
        self.ref_pareto_front = ref_pareto_front

        # Mixtral 8*7b
        self.aux_loss_1 = self.args.aux_loss_coef_1 > 1e-8
        self.aux_loss_2 = self.args.aux_loss_coef_2 > 1e-8

        # eval functions
        self.evaluate = {"SFT":self.sft_evaluate, "DPO":self.dpo_evaluate, "KD":self.kd_evaluate}

        # fitting algorithm
        self.fit_function ={
            "VaFF": self.fit_VaFF, 
            "VaFF2": self.fit_VaFF_trail, 
            "F2SA": self.fit_F2SA, 
            "ALRIGHT": self.fit_biobjective,
            "BOME": self.fit_BOME,
            "SEQ": self.fit_sequantial,
            "obj1": self.fit_obj1, 
            }
        
        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        self._tensorboard = None
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self,args):
        self.fit_function[args.algorithm](args)

    def fit_VaFF(self, args):
        print("--------------- training via VaFF ---------------")
        # # get the maximum train_loader length,
        # if self.train_dataloader_1.__len__() >= self.train_dataloader_2.__len__():
        #     train_loader_len = self.train_dataloader_1.__len__()
        # else:
        #     train_loader_len = self.train_dataloader_2.__len__()
        
        # take train_dataloader_1 for UL
        train_loader_len = self.train_dataloader_1.__len__()
        
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = train_loader_len  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        # # For blending objectives
        # prob_mass = [self.lambd, 1-self.lambd]

        # flag to check whether sop learning criteria had met
        OPTIM_ACHIEVED = False

        # Setup iterable train_dataloader
        if isinstance(self.train_dataloader_1.sampler, DistributedSampler):
            sample_shuffler_1 = 0
            self.train_dataloader_1.sampler.set_epoch(sample_shuffler_1)
            sample_shuffler_1 += 1
        iter_train_dataloader_1 = iter(self.train_dataloader_1)

        if isinstance(self.train_dataloader_2.sampler, DistributedSampler):
            sample_shuffler_2 = 0
            self.train_dataloader_2.sampler.set_epoch(sample_shuffler_2)
            sample_shuffler_2 += 1
        iter_train_dataloader_2 = iter(self.train_dataloader_2)

        # # setup objective sampling flag
        # sample_index = True
        
        start_time = time.time()
        for epoch in range(self.epochs):

            step_bar = tqdm(
                range(train_loader_len),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model:
                self.ref_model.eval()
            if self.ref_model_2:
                self.ref_model_2.eval()
            acc_mean = 0
            loss_mean = 0

            # Train min DPO min SFT
            # assuming both train_dataloaders have the same length
            
            for step in range(train_loader_len): # important todo: 1) shuffle logic of dataloader, here the dataloader is not shuffled when depleted 2) alternating after each micro step or update step? 3) better definition of epoch
                if time.time() - start_time >self.max_training_time:
                    break
                ############################ 
                # Step 1 freeze all UL variable
                # conduct update on LL variable
                if isinstance(self.model,Actor):
                    freeze_model(self.model.model,self.model_param2) # freeze all parameters in model except for model_param2
                else:
                    freeze_model(self.model,self.model_param2)
                
                ## Calculate loss
                obj_index_UL, obj_index_LL = 1,2
                loss_fn1, data1, iter_train_dataloader_1, sample_shuffler_1 = self.get_next_batch(
                    obj_index_UL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )
                loss_fn2, data2, iter_train_dataloader_2, sample_shuffler_2 = self.get_next_batch(
                    obj_index_LL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )

                # Compute loss1 for obj_index=1
                loss1, logs1 = self.calc_loss(loss_fn1, data1, obj_index=obj_index_UL)
                # Compute loss2 for obj_index=2
                loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)

                loss = 1/self.gam* loss1 +loss2
                # loss, logs_dict = self.calc_loss(loss_fn, data, obj_index)

                # # sample an index in the next step ONLY if the model is updated in this step (i.e. this step is gradient_accumulation_boundary)
                sample_index = self.model.model.is_gradient_accumulation_boundary() if isinstance(self.model, Actor) else self.model.is_gradient_accumulation_boundary()

                # conduct back propa
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                ############################ 
                # Step 2 freeze all LL variable
                # use optim1 to conduct UL update
                if isinstance(self.model,Actor):
                    freeze_model(self.model.model,self.model_param1) # freeze all parameters in model except for model_param1
                else:
                    freeze_model(self.model,self.model_param1)
                loss1, logs1 = self.calc_loss(loss_fn1, data1, obj_index=obj_index_UL)
                # loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)
                self.strategy.backward(loss1, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # save training losses
                # Average the loss across GPUs
                reduced_loss1 = self.strategy.all_reduce(loss1).item()
                reduced_loss2 = self.strategy.all_reduce(loss2).item()
                elapsed_time = time.time() - start_time

                

                # logs/checkpoints/evaluate #todo: output both log dicts of the objs 
                eval_loss_1, _ = self.save_logs_and_checkpoints(args, global_step, step_bar, logs1, obj_index_UL)
                _, eval_loss_2 = self.save_logs_and_checkpoints(args, global_step, step_bar, logs2, obj_index_LL)

                if self.strategy.is_rank_0():
                    # Only rank 0 saves the result
                    self.train_loss1_records[epoch].append((step, reduced_loss1))
                    self.train_loss2_records[epoch].append((step, reduced_loss2))
                    self.time_elapsed[epoch].append((step, elapsed_time))

                    self.eval_loss1_records[epoch].append((step, eval_loss_1))
                    self.eval_loss2_records[epoch].append((step, eval_loss_2))

                if eval_loss_1 is not None and eval_loss_2 is not None:
                    if eval_loss_1-self.args.obj_opt_1 <= self.args.eps or eval_loss_2-self.args.obj_opt_2 <= self.args.eps:
                        OPTIM_ACHIEVED = True
                    
                    else:
                        for ref_loss_pair in self.ref_pareto_front:
                            if eval_loss_1 - ref_loss_pair[0] <= self.args.eps and eval_loss_2 - ref_loss_pair[1] <= self.args.eps:
                                OPTIM_ACHIEVED = True
                                break

                step_bar.update()
                global_step += 1

                if OPTIM_ACHIEVED:
                    break

            epoch_bar.update()

            if OPTIM_ACHIEVED:
                break

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        elif self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


    def fit_BOME(self, args):
        
        eta = 0.5
        print(f"--------------- training via BOME with eta = {eta}---------------")
        # # get the maximum train_loader length,
        # if self.train_dataloader_1.__len__() >= self.train_dataloader_2.__len__():
        #     train_loader_len = self.train_dataloader_1.__len__()
        # else:
        #     train_loader_len = self.train_dataloader_2.__len__()
        
        # take train_dataloader_1 for UL
        train_loader_len = self.train_dataloader_1.__len__()
        
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = train_loader_len  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        # flag to check whether sop learning criteria had met
        OPTIM_ACHIEVED = False

        # Setup iterable train_dataloader
        if isinstance(self.train_dataloader_1.sampler, DistributedSampler):
            sample_shuffler_1 = 0
            self.train_dataloader_1.sampler.set_epoch(sample_shuffler_1)
            sample_shuffler_1 += 1
        iter_train_dataloader_1 = iter(self.train_dataloader_1)
        if isinstance(self.train_dataloader_2.sampler, DistributedSampler):

            sample_shuffler_2 = 0
            self.train_dataloader_2.sampler.set_epoch(sample_shuffler_2)
            sample_shuffler_2 += 1
        iter_train_dataloader_2 = iter(self.train_dataloader_2)

        # setup objective sampling flag
        sample_index = True

        start_time = time.time()
                
        for epoch in range(self.epochs):

            step_bar = tqdm(
                range(train_loader_len),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model:
                self.ref_model.eval()
            if self.ref_model_2:
                self.ref_model_2.eval()
            acc_mean = 0
            loss_mean = 0

            # LL starting parameter
            head_g = None
            
            # Train SFT and DPO in alternating manner, so implement a train_dataloader agnostic loop
            # assuming both train_dataloaders have the same length
            for step in range(train_loader_len):
                # if hit max training time
                if time.time() - start_time >self.max_training_time:
                    break
                self.model.train()
                ############################ 
                # Step 1 freeze all UL variable
                # for inner LL updates
                if isinstance(self.model,Actor):
                    freeze_model(self.model.model,self.model_param2) # freeze all parameters in model except for model_param2
                else:
                    freeze_model(self.model,self.model_param2)
                
                # keep the current head (LL) parameters
                head =  getattr(self.model.model.module, self.args.output_layer_name) if isinstance(self.model, Actor) else getattr(self.model.module, self.args.output_layer_name)
                orig_head = {n: p.detach().cpu().clone() for n,p in head.named_parameters()}

                # use from last update as starting point:
                # restore head
                if head_g is not None:
                    for n,p in head.named_parameters():
                        p.data.copy_(head_g[n])

                ## Calculate loss
                obj_index_UL, obj_index_LL = 1,2
                loss_fn1, data1, iter_train_dataloader_1, sample_shuffler_1 = self.get_next_batch(
                    obj_index_UL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )
                loss_fn2, data2, iter_train_dataloader_2, sample_shuffler_2 = self.get_next_batch(
                    obj_index_LL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )
                
                for inner_step in range(args.inner_stepsize):
                    loss2, _ = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)
                    # conduct back propa
                    self.strategy.backward(loss2, self.model, self.optimizer)
                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # save head_g
                head_g = {n: p.detach().cpu().clone() for n,p in head.named_parameters()} # LL

                #############################  Calculate lambda_k
                # restore LL head
                for n,p in head.named_parameters():
                    p.data.copy_(orig_head[n])

                # collect all trainable params (both w‐ and h‐blocks)
                if isinstance(self.model,Actor):
                    freeze_model(self.model.model,self.model_param1+self.model_param2) 
                else:
                    freeze_model(self.model,self.model_param1+self.model_param2)
                params = [p for p in self.model.model.parameters() if p.requires_grad] if isinstance(self.model,Actor) else [p for p in self.model.parameters() if p.requires_grad]
                
                loss1, logs1 = self.calc_loss(loss_fn1, data1, obj_index=obj_index_UL)
                grads1 = torch.autograd.grad(loss1, params, retain_graph=True)

                loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)
                grads2 = torch.autograd.grad(loss2, params, retain_graph=True)

                # restore LL head g
                for n,p in head.named_parameters():
                    p.data.copy_(head_g[n])

                # freeze LL, unfreeze only UL
                if isinstance(self.model,Actor):
                    freeze_model(self.model.model,self.model_param1) # freeze all parameters in model except for model_param2
                else:
                    freeze_model(self.model,self.model_param1)

                params_UL = [p for p in self.model.model.parameters() if p.requires_grad] if isinstance(self.model,Actor) else [p for p in self.model.parameters() if p.requires_grad]
                params_UL_ids = { id(p) for p in params_UL }
                
                loss2_g, _ = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)
                grads2_g = torch.autograd.grad(loss2_g, params_UL, retain_graph=False)

                # restore head requires_grad and θ back to θ_k
                if isinstance(self.model,Actor):
                    freeze_model(self.model.model,self.model_param1+self.model_param2) 
                else:
                    freeze_model(self.model,self.model_param1+self.model_param2)

                # restore LL head
                for n,p in head.named_parameters():
                    p.data.copy_(orig_head[n])

                # assumble the grads
                grads_q_hat = []
                inner_iter = iter(grads2_g)
                for p, g_curr in zip(params, grads2):
                    if id(p) in params_UL_ids:
                        g_inner = next(inner_iter)
                        grads_q_hat.append(g_curr - g_inner)
                    else:
                        # head‐param: ∇_θ q̂ = ∇_θ g_current
                        grads_q_hat.append(g_curr)

                
                #    phi = η * ‖∇q̂‖²
                phi = eta * sum((g.norm()**2) for g in grads_q_hat)
                ip  = sum((gf * gq).sum() for gf, gq in zip(grads1, grads_q_hat))
                den = sum((gq*gq).sum() for gq in grads_q_hat)
                lambda_k = torch.clamp((phi - ip) / den, min=0.0)

                #############################
                combined = [gf + lambda_k * gq for gf, gq in zip(grads1, grads_q_hat)]

                self.optimizer.zero_grad()
                for p, g in zip(params, combined):
                    p.grad = g
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # #############################
                # # Step 2 unfreeze LL variable freeze all UL variable
                # # conduct LL update
                # if isinstance(self.model,Actor):
                #     freeze_model(self.model.model,self.model_param1) # freeze all parameters in model except for model_param2
                # else:
                #     freeze_model(self.model,self.model_param1)
                
                
                # loss2_g, _ = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)

                # # restore head
                # for n,p in head.named_parameters():
                #     p.data.copy_(orig_head[n])
                
                # # Compute loss2 for obj_index=2
                # loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)


                # #############################  Conduct LL update
                # # Step 2 unfreeze LL variable freeze all UL variable
                # # conduct LL update
                # if isinstance(self.model,Actor):
                #     freeze_model(self.model.model,self.model_param2) # freeze all parameters in model except for model_param2
                # else:
                #     freeze_model(self.model,self.model_param2)
                
                # # restore head
                # for n,p in head.named_parameters():
                #     p.data.copy_(orig_head[n])

                # for inner_step in range(args.inner_stepsize):
                #     # Compute loss1 for obj_index=1
                #     loss1, logs1 = self.calc_loss(loss_fn1, data1, obj_index=obj_index_UL)
                #     # Compute loss2 for obj_index=2
                #     loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)
                #     loss = 1/self.gam * loss1 + loss2

                #     self.strategy.backward(loss, self.model, self.optimizer)
                #     self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                # save 
                orig_head = {n: p.detach().cpu().clone() for n,p in head.named_parameters()}

                # #############################  Conduct UL update
                # # Step 2 unfreeze UL variable freeze all LL variable
                # # conduct UL update
                # if isinstance(self.model,Actor):
                #     freeze_model(self.model.model,self.model_param1) # freeze all parameters in model except for model_param1
                # else:
                #     freeze_model(self.model,self.model_param1)
                
                # # restore head g
                # for n,p in head.named_parameters():
                #     p.data.copy_(head_g[n])

                # loss2_g, _ = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)

                # # restore head
                # for n,p in head.named_parameters():
                #     p.data.copy_(orig_head[n])
                # # Compute loss1 for obj_index=1
                # loss1, logs1 = self.calc_loss(loss_fn1, data1, obj_index=obj_index_UL)
                # # Compute loss2 for obj_index=2
                # loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)

                # loss = loss1 + self.gam*(loss2 - loss2_g)
                # # loss, logs_dict = self.calc_loss(loss_fn, data, obj_index)

                # # # sample an index in the next step ONLY if the model is updated in this step (i.e. this step is gradient_accumulation_boundary)
                # sample_index = self.model.model.is_gradient_accumulation_boundary() if isinstance(self.model, Actor) else self.model.is_gradient_accumulation_boundary()

                # # conduct back propa
                # self.strategy.backward(loss, self.model, self.optimizer)
                # self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                
                # save training losses
                # Average the loss across GPUs
                reduced_loss1 = self.strategy.all_reduce(loss1).item()
                reduced_loss2 = self.strategy.all_reduce(loss2).item()
                elapsed_time = time.time() - start_time

                # logs/checkpoints/evaluate #todo: output both log dicts of the objs 
                eval_loss_1, _ = self.save_logs_and_checkpoints(args, global_step, step_bar, logs1, obj_index_UL)
                _, eval_loss_2 = self.save_logs_and_checkpoints(args, global_step, step_bar, logs2, obj_index_LL)

                if self.strategy.is_rank_0():
                    # Only rank 0 saves the result
                    self.train_loss1_records[epoch].append((step, reduced_loss1))
                    self.train_loss2_records[epoch].append((step, reduced_loss2))
                    self.time_elapsed[epoch].append((step, elapsed_time))

                    self.eval_loss1_records[epoch].append((step, eval_loss_1))
                    self.eval_loss2_records[epoch].append((step, eval_loss_2))


                if eval_loss_1 is not None and eval_loss_2 is not None:
                    if eval_loss_1-self.args.obj_opt_1 <= self.args.eps or eval_loss_2-self.args.obj_opt_2 <= self.args.eps:
                        OPTIM_ACHIEVED = True
                    
                    else:
                        for ref_loss_pair in self.ref_pareto_front:
                            if eval_loss_1 - ref_loss_pair[0] <= self.args.eps and eval_loss_2 - ref_loss_pair[1] <= self.args.eps:
                                OPTIM_ACHIEVED = True
                                break

                step_bar.update()
                global_step += 1

                if OPTIM_ACHIEVED:
                    break

            epoch_bar.update()

            if OPTIM_ACHIEVED:
                break

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        elif self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def fit_F2SA(self, args):
        print("--------------- training via F2SA ---------------")
        # # get the maximum train_loader length,
        # if self.train_dataloader_1.__len__() >= self.train_dataloader_2.__len__():
        #     train_loader_len = self.train_dataloader_1.__len__()
        # else:
        #     train_loader_len = self.train_dataloader_2.__len__()
        
        # take train_dataloader_1 for UL
        train_loader_len = self.train_dataloader_1.__len__()
        
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = train_loader_len  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        # For blending objectives
        prob_mass = [self.lambd, 1-self.lambd]

        # flag to check whether sop learning criteria had met
        OPTIM_ACHIEVED = False

        # Setup iterable train_dataloader
        if isinstance(self.train_dataloader_1.sampler, DistributedSampler):
            sample_shuffler_1 = 0
            self.train_dataloader_1.sampler.set_epoch(sample_shuffler_1)
            sample_shuffler_1 += 1
        iter_train_dataloader_1 = iter(self.train_dataloader_1)
        if isinstance(self.train_dataloader_2.sampler, DistributedSampler):

            sample_shuffler_2 = 0
            self.train_dataloader_2.sampler.set_epoch(sample_shuffler_2)
            sample_shuffler_2 += 1
        iter_train_dataloader_2 = iter(self.train_dataloader_2)

        # setup objective sampling flag
        sample_index = True

        start_time = time.time()
                
        for epoch in range(self.epochs):

            step_bar = tqdm(
                range(train_loader_len),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model:
                self.ref_model.eval()
            if self.ref_model_2:
                self.ref_model_2.eval()
            acc_mean = 0
            loss_mean = 0

            # LL starting parameter
            head_g = None
            
            # Train SFT and DPO in alternating manner, so implement a train_dataloader agnostic loop
            # assuming both train_dataloaders have the same length
            for step in range(train_loader_len): # important todo: 1) shuffle logic of dataloader, here the dataloader is not shuffled when depleted 2) alternating after each micro step or update step? 3) better definition of epoch
                # if hit max training time
                if time.time() - start_time >self.max_training_time:
                    break
                self.model.train()
                ############################ 
                # Step 1 freeze all UL variable
                # for inner LL updates
                if isinstance(self.model,Actor):
                    freeze_model(self.model.model,self.model_param2) # freeze all parameters in model except for model_param2
                else:
                    freeze_model(self.model,self.model_param2)
                
                # keep the current head (LL) parameters
                head =  getattr(self.model.model.module, self.args.output_layer_name) if isinstance(self.model, Actor) else getattr(self.model.module, self.args.output_layer_name)
                orig_head = {n: p.detach().cpu().clone() for n,p in head.named_parameters()}

                # use from last update as starting point:
                # restore head
                if head_g is not None:
                    for n,p in head.named_parameters():
                        p.data.copy_(head_g[n])

                ## Calculate loss
                obj_index_UL, obj_index_LL = 1,2
                loss_fn1, data1, iter_train_dataloader_1, sample_shuffler_1 = self.get_next_batch(
                    obj_index_UL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )
                loss_fn2, data2, iter_train_dataloader_2, sample_shuffler_2 = self.get_next_batch(
                    obj_index_LL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )
                
                for inner_step in range(args.inner_stepsize):
                    loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)
                    # conduct back propa
                    self.strategy.backward(loss2, self.model, self.optimizer)
                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                head_g = {n: p.detach().cpu().clone() for n,p in head.named_parameters()} # LL

                # #############################  Step 2
                # # restore LL head
                # for n,p in head.named_parameters():
                #     p.data.copy_(orig_head[n])

                # # collect all trainable params (both w‐ and h‐blocks)
                # if isinstance(self.model,Actor):
                #     freeze_model(self.model.model,self.model_param1+self.model_param2) 
                # else:
                #     freeze_model(self.model,self.model_param1+self.model_param2)
                # params = [p for p in self.model.model.parameters() if p.requires_grad] if isinstance(self.model,Actor) else [p for p in self.model.parameters() if p.requires_grad]
                
                # loss1, logs1 = self.calc_loss(loss_fn1, data1, obj_index=obj_index_UL)
                # grads1 = torch.autograd.grad(loss1, params, retain_graph=True)

                # loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)
                # grads2 = torch.autograd.grad(loss2, params, retain_graph=True)

                # # restore LL head g
                # for n,p in head.named_parameters():
                #     p.data.copy_(head_g[n])

                # # freeze LL, unfreeze only UL
                # if isinstance(self.model,Actor):
                #     freeze_model(self.model.model,self.model_param1) # freeze all parameters in model except for model_param2
                # else:
                #     freeze_model(self.model,self.model_param1)

                # params_UL = [p for p in self.model.model.parameters() if p.requires_grad] if isinstance(self.model,Actor) else [p for p in self.model.parameters() if p.requires_grad]
                # params_UL_ids = { id(p) for p in params_UL }
                
                # loss2_g, _ = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)
                # grads2_g = torch.autograd.grad(loss2_g, params_UL, retain_graph=False)

                # # restore head requires_grad and θ back to θ_k
                # if isinstance(self.model,Actor):
                #     freeze_model(self.model.model,self.model_param1+self.model_param2) 
                # else:
                #     freeze_model(self.model,self.model_param1+self.model_param2)

                # # restore LL head
                # for n,p in head.named_parameters():
                #     p.data.copy_(orig_head[n])

                # # assumble the grads
                # grads_q_hat = []
                # inner_iter = iter(grads2_g)
                # for p, g_curr in zip(params, grads2):
                #     if id(p) in params_UL_ids:
                #         g_inner = next(inner_iter)
                #         grads_q_hat.append(g_curr - g_inner)
                #     else:
                #         # head‐param: ∇_θ q̂ = ∇_θ g_current
                #         grads_q_hat.append(g_curr)
                

                # #############################
                # combined = [gf + self.gam * gq for gf, gq in zip(grads1, grads_q_hat)]

                # self.optimizer.zero_grad()
                # for p, g in zip(params, combined):
                #     p.grad = g
                # self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                #############################  Conduct LL update
                # Step 2 unfreeze LL variable freeze all UL variable
                # conduct LL update
                if isinstance(self.model,Actor):
                    freeze_model(self.model.model,self.model_param2) # freeze all parameters in model except for model_param2
                else:
                    freeze_model(self.model,self.model_param2)
                
                # restore head
                for n,p in head.named_parameters():
                    p.data.copy_(orig_head[n])

                for inner_step in range(args.inner_stepsize):
                    # Compute loss1 for obj_index=1
                    loss1, logs1 = self.calc_loss(loss_fn1, data1, obj_index=obj_index_UL)
                    # Compute loss2 for obj_index=2
                    loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)
                    loss = 1/self.gam * loss1 + loss2

                    self.strategy.backward(loss, self.model, self.optimizer)
                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                # save 
                orig_head = {n: p.detach().cpu().clone() for n,p in head.named_parameters()}



                #############################  Conduct UL update
                # Step 2 unfreeze UL variable freeze all LL variable
                # conduct UL update
                if isinstance(self.model,Actor):
                    freeze_model(self.model.model,self.model_param1) # freeze all parameters in model except for model_param1
                else:
                    freeze_model(self.model,self.model_param1)
                
                # restore head g
                for n,p in head.named_parameters():
                    p.data.copy_(head_g[n])

                loss2_g, _ = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)

                # restore head
                for n,p in head.named_parameters():
                    p.data.copy_(orig_head[n])
                # Compute loss1 for obj_index=1
                loss1, logs1 = self.calc_loss(loss_fn1, data1, obj_index=obj_index_UL)
                # Compute loss2 for obj_index=2
                loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)

                loss = loss1 + self.gam*(loss2 - loss2_g)
                # loss, logs_dict = self.calc_loss(loss_fn, data, obj_index)

                # # sample an index in the next step ONLY if the model is updated in this step (i.e. this step is gradient_accumulation_boundary)
                sample_index = self.model.model.is_gradient_accumulation_boundary() if isinstance(self.model, Actor) else self.model.is_gradient_accumulation_boundary()

                # conduct back propa
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                
                # save training losses
                # Average the loss across GPUs
                reduced_loss1 = self.strategy.all_reduce(loss1).item()
                reduced_loss2 = self.strategy.all_reduce(loss2).item()
                elapsed_time = time.time() - start_time

                # logs/checkpoints/evaluate #todo: output both log dicts of the objs 
                eval_loss_1, _ = self.save_logs_and_checkpoints(args, global_step, step_bar, logs1, obj_index_UL)
                _, eval_loss_2 = self.save_logs_and_checkpoints(args, global_step, step_bar, logs2, obj_index_LL)

                if self.strategy.is_rank_0():
                    # Only rank 0 saves the result
                    self.train_loss1_records[epoch].append((step, reduced_loss1))
                    self.train_loss2_records[epoch].append((step, reduced_loss2))
                    self.time_elapsed[epoch].append((step, elapsed_time))

                    self.eval_loss1_records[epoch].append((step, eval_loss_1))
                    self.eval_loss2_records[epoch].append((step, eval_loss_2))


                if eval_loss_1 is not None and eval_loss_2 is not None:
                    if eval_loss_1-self.args.obj_opt_1 <= self.args.eps or eval_loss_2-self.args.obj_opt_2 <= self.args.eps:
                        OPTIM_ACHIEVED = True
                    
                    else:
                        for ref_loss_pair in self.ref_pareto_front:
                            if eval_loss_1 - ref_loss_pair[0] <= self.args.eps and eval_loss_2 - ref_loss_pair[1] <= self.args.eps:
                                OPTIM_ACHIEVED = True
                                break

                step_bar.update()
                global_step += 1

                if OPTIM_ACHIEVED:
                    break

            epoch_bar.update()

            if OPTIM_ACHIEVED:
                break

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        elif self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


    def fit_sequantial(self, args, SFT_prop = 0.5 ):
        print("--------------- training on sequential ---------------")
        
        print("--------------- start with SFT ---------------")
        SFT_epochs = int(self.epochs * SFT_prop)
        # train SFT only on head
        if isinstance(self.model,Actor):
            freeze_model(self.model.model,self.model_param2) # freeze all parameters in model except for model_param2
        else:
            freeze_model(self.model,self.model_param2)
        
        train_loader_len = self.train_dataloader_1.__len__()
        
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = train_loader_len  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        # # For blending objectives
        # prob_mass = [self.lambd, 1-self.lambd]

        # flag to check whether sop learning criteria had met
        OPTIM_ACHIEVED = False

        # Setup iterable train_dataloader
        if isinstance(self.train_dataloader_1.sampler, DistributedSampler):
            sample_shuffler_1 = 0
            self.train_dataloader_1.sampler.set_epoch(sample_shuffler_1)
            sample_shuffler_1 += 1
        iter_train_dataloader_1 = iter(self.train_dataloader_1)

        if self.train_dataloader_2 is None:
            if isinstance(self.train_dataloader_2.sampler, DistributedSampler):
                sample_shuffler_2 = 0
                self.train_dataloader_2.sampler.set_epoch(sample_shuffler_2)
                sample_shuffler_2 += 1
            iter_train_dataloader_2 = iter(self.train_dataloader_2)
        else:
            if isinstance(self.train_dataloader_2.sampler, DistributedSampler):
                sample_shuffler_2 = 0
                self.train_dataloader_2.sampler.set_epoch(sample_shuffler_2)
                sample_shuffler_2 += 1
            iter_train_dataloader_2 = iter(self.train_dataloader_2)

        # # setup objective sampling flag
        # sample_index = True
        
        ############################ 
        start_time = time.time()

        for epoch in range(SFT_epochs):

            step_bar = tqdm(
                range(train_loader_len),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model:
                self.ref_model.eval()
            if self.ref_model_2:
                self.ref_model_2.eval()
            # acc_mean = 0
            # loss_mean = 0

            # Train min DPO min SFT
            # assuming both train_dataloaders have the same length
            for step in range(train_loader_len): # important todo: 1) shuffle logic of dataloader, here the dataloader is not shuffled when depleted 2) alternating after each micro step or update step? 3) better definition of epoch
                # if hit max SFT_prop of training time
                if time.time() - start_time > SFT_prop * args.max_SFT_stage_time:
                    break
                
                self.model.train()
                ## Calculate loss
                obj_index_UL, obj_index_LL = 1,2

                loss_fn1, data1, iter_train_dataloader_1, sample_shuffler_1 = self.get_next_batch(
                    obj_index_UL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )
                loss_fn2, data2, iter_train_dataloader_2, sample_shuffler_2 = self.get_next_batch(
                    obj_index_LL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )
            

                # Compute loss1 for obj_index=1
                loss1, logs1 = self.calc_loss(loss_fn1, data1, obj_index=obj_index_UL)
                # Compute loss2 for obj_index=2
                loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)



                loss = loss2
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                # logs/checkpoints/evaluate #todo: output both log dicts of the objs 
                # eval_loss_1, _ = self.save_logs_and_checkpoints(args, global_step, step_bar, logs1, obj_index_UL)
                eval_loss_1, eval_loss_2 = self.save_logs_and_checkpoints(args, global_step, step_bar, logs2, obj_index_LL)

                reduced_loss1 = self.strategy.all_reduce(loss1).item()
                reduced_loss2 = self.strategy.all_reduce(loss2).item()
                elapsed_time = time.time()-start_time
                # Only rank 0 saves results
                if self.strategy.is_rank_0():
                    try:
                        self.train_loss1_records[epoch].append((step, reduced_loss1)) 
                    except:
                        self.train_loss1_records[epoch].append((step, float("nan"))) 
                    try:
                        self.train_loss2_records[epoch].append((step, reduced_loss2)) 
                    except:
                        self.train_loss2_records[epoch].append((step, float("nan"))) 
                    
                    self.time_elapsed[epoch].append((step, elapsed_time))
                    
                    if eval_loss_1 is not None:
                        self.eval_loss1_records[epoch].append((step, eval_loss_1)) 
                    else:
                        self.eval_loss1_records[epoch].append((step, float("nan"))) 
                    if eval_loss_2 is not None:
                        self.eval_loss2_records[epoch].append((step, eval_loss_2)) 
                    else:
                        self.eval_loss2_records[epoch].append((step, float("nan"))) 
       
                if eval_loss_1 is not None and eval_loss_2 is not None:
                    if eval_loss_1-self.args.obj_opt_1 <= self.args.eps or eval_loss_2-self.args.obj_opt_2 <= self.args.eps:
                        OPTIM_ACHIEVED = True
                    
                    else:
                        for ref_loss_pair in self.ref_pareto_front:
                            if eval_loss_1 - ref_loss_pair[0] <= self.args.eps and eval_loss_2 - ref_loss_pair[1] <= self.args.eps:
                                OPTIM_ACHIEVED = True
                                break

                step_bar.update()
                global_step += 1

                if OPTIM_ACHIEVED:
                    break

            epoch_bar.update()

            if OPTIM_ACHIEVED:
                break


        print("--------------- continue with DPO ---------------")
        # train SFT only on head
        if isinstance(self.model,Actor):
            freeze_model(self.model.model,self.model_param1 + self.model_param2) # freeze all parameters in model except for model_param2
        else:
            freeze_model(self.model,self.model_param1 + self.model_param2)
        for epoch in range(self.epochs):

            step_bar = tqdm(
                range(train_loader_len),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model:
                self.ref_model.eval()
            if self.ref_model_2:
                self.ref_model_2.eval()
            # acc_mean = 0
            # loss_mean = 0

            # Train min DPO min SFT
            # assuming both train_dataloaders have the same length
            for step in range(train_loader_len): # important todo: 1) shuffle logic of dataloader, here the dataloader is not shuffled when depleted 2) alternating after each micro step or update step? 3) better definition of epoch
                # if hit max training time
                if time.time() - start_time >args.max_SFT_stage_time:
                    break
                
                self.model.train()
                ## Calculate loss
                obj_index_UL, obj_index_LL = 1,2

                loss_fn1, data1, iter_train_dataloader_1, sample_shuffler_1 = self.get_next_batch(
                    obj_index_UL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )
                loss_fn2, data2, iter_train_dataloader_2, sample_shuffler_2 = self.get_next_batch(
                    obj_index_LL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )
            

                # Compute loss1 for obj_index=1
                loss1, logs1 = self.calc_loss(loss_fn1, data1, obj_index=obj_index_UL)
                # Compute loss2 for obj_index=2
                loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)



                loss = loss1
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                # logs/checkpoints/evaluate #todo: output both log dicts of the objs 
                # eval_loss_1, _ = self.save_logs_and_checkpoints(args, global_step, step_bar, logs1, obj_index_UL)
                eval_loss_1, eval_loss_2 = self.save_logs_and_checkpoints(args, global_step, step_bar, logs1, obj_index_LL)

                reduced_loss1 = self.strategy.all_reduce(loss1).item()
                reduced_loss2 = self.strategy.all_reduce(loss2).item()
                elapsed_time = time.time()-start_time
                # Only rank 0 saves results
                if self.strategy.is_rank_0():
                    try:
                        self.train_loss1_records[epoch].append((step, reduced_loss1)) 
                    except:
                        self.train_loss1_records[epoch].append((step, float("nan"))) 
                    try:
                        self.train_loss2_records[epoch].append((step, reduced_loss2)) 
                    except:
                        self.train_loss2_records[epoch].append((step, float("nan"))) 
                    
                    self.time_elapsed[epoch].append((step, elapsed_time))
                    
                    if eval_loss_1 is not None:
                        self.eval_loss1_records[epoch].append((step, eval_loss_1)) 
                    else:
                        self.eval_loss1_records[epoch].append((step, float("nan"))) 
                    if eval_loss_2 is not None:
                        self.eval_loss2_records[epoch].append((step, eval_loss_2)) 
                    else:
                        self.eval_loss2_records[epoch].append((step, float("nan"))) 
       
                if eval_loss_1 is not None and eval_loss_2 is not None:
                    if eval_loss_1-self.args.obj_opt_1 <= self.args.eps or eval_loss_2-self.args.obj_opt_2 <= self.args.eps:
                        OPTIM_ACHIEVED = True
                    
                    else:
                        for ref_loss_pair in self.ref_pareto_front:
                            if eval_loss_1 - ref_loss_pair[0] <= self.args.eps and eval_loss_2 - ref_loss_pair[1] <= self.args.eps:
                                OPTIM_ACHIEVED = True
                                break

                step_bar.update()
                global_step += 1

                if OPTIM_ACHIEVED:
                    break

            epoch_bar.update()

            if OPTIM_ACHIEVED:
                break

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        elif self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


    def fit_obj1(self, args, head_only=True):
        print("--------------- training on obj 1 ---------------")
        # # get the maximum train_loader length,
        # if self.train_dataloader_1.__len__() >= self.train_dataloader_2.__len__():
        #     train_loader_len = self.train_dataloader_1.__len__()
        # else:
        #     train_loader_len = self.train_dataloader_2.__len__()
        
        # take train_dataloader_1 for UL
        train_loader_len = self.train_dataloader_1.__len__()
        
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = train_loader_len  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        # # For blending objectives
        # prob_mass = [self.lambd, 1-self.lambd]

        # flag to check whether sop learning criteria had met
        OPTIM_ACHIEVED = False

        # Setup iterable train_dataloader
        if isinstance(self.train_dataloader_1.sampler, DistributedSampler):
            sample_shuffler_1 = 0
            self.train_dataloader_1.sampler.set_epoch(sample_shuffler_1)
            sample_shuffler_1 += 1
        iter_train_dataloader_1 = iter(self.train_dataloader_1)

        if isinstance(self.train_dataloader_2.sampler, DistributedSampler):
            sample_shuffler_2 = 0
            self.train_dataloader_2.sampler.set_epoch(sample_shuffler_2)
            sample_shuffler_2 += 1
        iter_train_dataloader_2 = iter(self.train_dataloader_2)

        # # setup objective sampling flag
        # sample_index = True
        
        ############################ 
        for epoch in range(self.epochs):

            step_bar = tqdm(
                range(train_loader_len),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model:
                self.ref_model.eval()
            if self.ref_model_2:
                self.ref_model_2.eval()
            # acc_mean = 0
            # loss_mean = 0

            # Train min DPO min SFT
            # assuming both train_dataloaders have the same length
            for step in range(train_loader_len): # important todo: 1) shuffle logic of dataloader, here the dataloader is not shuffled when depleted 2) alternating after each micro step or update step? 3) better definition of epoch
                self.model.train()
                ## Calculate loss
                obj_index_UL, obj_index_LL = 1,2
                loss_fn1, data1, iter_train_dataloader_1, sample_shuffler_1 = self.get_next_batch(
                    obj_index_UL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )
                # loss_fn2, data2, iter_train_dataloader_2, sample_shuffler_2 = self.get_next_batch(
                #     obj_index_LL, iter_train_dataloader_1, iter_train_dataloader_2,
                #     sample_shuffler_1, sample_shuffler_2
                # )

                # Compute loss1 for obj_index=1
                loss1, logs1 = self.calc_loss(loss_fn1, data1, obj_index=obj_index_UL)
                # Compute loss2 for obj_index=2
                # loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)

                loss = loss1
                # loss, logs_dict = self.calc_loss(loss_fn, data, obj_index)

                # # sample an index in the next step ONLY if the model is updated in this step (i.e. this step is gradient_accumulation_boundary)
                # sample_index = self.model.model.is_gradient_accumulation_boundary() if isinstance(self.model, Actor) else self.model.is_gradient_accumulation_boundary()

                # conduct back propa
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                # logs/checkpoints/evaluate #todo: output both log dicts of the objs 
                eval_loss_1, eval_loss_2 = self.save_logs_and_checkpoints(args, global_step, step_bar, logs1, obj_index_UL)
                # _, eval_loss_2 = self.save_logs_and_checkpoints(args, global_step, step_bar, logs2, obj_index_LL)

                if eval_loss_1 is not None and eval_loss_2 is not None:
                    if eval_loss_1-self.args.obj_opt_1 <= self.args.eps or eval_loss_2-self.args.obj_opt_2 <= self.args.eps:
                        OPTIM_ACHIEVED = True
                    
                    else:
                        for ref_loss_pair in self.ref_pareto_front:
                            if eval_loss_1 - ref_loss_pair[0] <= self.args.eps and eval_loss_2 - ref_loss_pair[1] <= self.args.eps:
                                OPTIM_ACHIEVED = True
                                break

                step_bar.update()
                global_step += 1

                if OPTIM_ACHIEVED:
                    break

            epoch_bar.update()

            if OPTIM_ACHIEVED:
                break

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        elif self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def fit_obj2(self, args, head_only=True):
        print("--------------- training on obj 2 ---------------")

        if head_only:
            if isinstance(self.model,Actor):
                freeze_model(self.model.model,self.model_param2) # freeze all parameters in model except for model_param2
            else:
                freeze_model(self.model,self.model_param2)
        # # get the maximum train_loader length,
        # if self.train_dataloader_1.__len__() >= self.train_dataloader_2.__len__():
        #     train_loader_len = self.train_dataloader_1.__len__()
        # else:
        #     train_loader_len = self.train_dataloader_2.__len__()
        
        # take train_dataloader_1 for UL
        train_loader_len = self.train_dataloader_1.__len__()
        
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = train_loader_len  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        # # For blending objectives
        # prob_mass = [self.lambd, 1-self.lambd]

        # flag to check whether sop learning criteria had met
        OPTIM_ACHIEVED = False

        # Setup iterable train_dataloader
        if isinstance(self.train_dataloader_1.sampler, DistributedSampler):
            sample_shuffler_1 = 0
            self.train_dataloader_1.sampler.set_epoch(sample_shuffler_1)
            sample_shuffler_1 += 1
        iter_train_dataloader_1 = iter(self.train_dataloader_1)


        if isinstance(self.train_dataloader_2_2.sampler, DistributedSampler):
            sample_shuffler_2 = 0
            self.train_dataloader_2_2.sampler.set_epoch(sample_shuffler_2)
            sample_shuffler_2 += 1
        iter_train_dataloader_2 = iter(self.train_dataloader_2_2)

        # # setup objective sampling flag
        # sample_index = True
        
        ############################ 
        start_time = time.time()

        for epoch in range(self.epochs):

            step_bar = tqdm(
                range(train_loader_len),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model:
                self.ref_model.eval()
            if self.ref_model_2:
                self.ref_model_2.eval()
            # acc_mean = 0
            # loss_mean = 0

            # Train min DPO min SFT
            # assuming both train_dataloaders have the same length
            for step in range(train_loader_len): # important todo: 1) shuffle logic of dataloader, here the dataloader is not shuffled when depleted 2) alternating after each micro step or update step? 3) better definition of epoch
                # if hit max training time
                if time.time() - start_time >args.max_SFT_stage_time:
                    break
                
                self.model.train()
                ## Calculate loss
                obj_index_UL, obj_index_LL = 1,2

                loss_fn1, data1, iter_train_dataloader_1, sample_shuffler_1 = self.get_next_batch(
                    obj_index_UL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )
                loss_fn2, data2, iter_train_dataloader_2, sample_shuffler_2 = self.get_next_batch(
                    obj_index_LL, iter_train_dataloader_1, iter_train_dataloader_2,
                    sample_shuffler_1, sample_shuffler_2
                )
            

                # Compute loss1 for obj_index=1
                loss1, logs1 = self.calc_loss(loss_fn1, data1, obj_index=obj_index_UL)
                # Compute loss2 for obj_index=2
                loss2, logs2 = self.calc_loss(loss_fn2, data2, obj_index=obj_index_LL)



                loss = 1/args.gam+loss1+loss2 if args.stage_2_gam_mode else loss2
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                # logs/checkpoints/evaluate #todo: output both log dicts of the objs 
                # eval_loss_1, _ = self.save_logs_and_checkpoints(args, global_step, step_bar, logs1, obj_index_UL)
                eval_loss_1, eval_loss_2 = self.save_logs_and_checkpoints(args, global_step, step_bar, logs2, obj_index_LL)

                reduced_loss1 = self.strategy.all_reduce(loss1).item()
                reduced_loss2 = self.strategy.all_reduce(loss2).item()
                elapsed_time = time.time()-start_time
                # Only rank 0 saves results
                if self.strategy.is_rank_0():
                    try:
                        self.train_loss1_records[epoch].append((step, reduced_loss1)) 
                    except:
                        self.train_loss1_records[epoch].append((step, float("nan"))) 
                    try:
                        self.train_loss2_records[epoch].append((step, reduced_loss2)) 
                    except:
                        self.train_loss2_records[epoch].append((step, float("nan"))) 
                    
                    self.time_elapsed[epoch].append((step, elapsed_time))
                    
                    if eval_loss_1 is not None:
                        self.eval_loss1_records[epoch].append((step, eval_loss_1)) 
                    else:
                        self.eval_loss1_records[epoch].append((step, float("nan"))) 
                    if eval_loss_2 is not None:
                        self.eval_loss2_records[epoch].append((step, eval_loss_2)) 
                    else:
                        self.eval_loss2_records[epoch].append((step, float("nan"))) 
       
                if eval_loss_1 is not None and eval_loss_2 is not None:
                    if eval_loss_1-self.args.obj_opt_1 <= self.args.eps or eval_loss_2-self.args.obj_opt_2 <= self.args.eps:
                        OPTIM_ACHIEVED = True
                    
                    else:
                        for ref_loss_pair in self.ref_pareto_front:
                            if eval_loss_1 - ref_loss_pair[0] <= self.args.eps and eval_loss_2 - ref_loss_pair[1] <= self.args.eps:
                                OPTIM_ACHIEVED = True
                                break

                step_bar.update()
                global_step += 1

                if OPTIM_ACHIEVED:
                    break

            epoch_bar.update()

            if OPTIM_ACHIEVED:
                break

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        elif self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


    def fit_biobjective(self, args):
        print(" ------------------- ALRIGHT -------------------")

        # get the maximum train_loader length,
        if self.train_dataloader_1.__len__() >= self.train_dataloader_2.__len__():
            train_loader_len = self.train_dataloader_1.__len__()
        else:
            train_loader_len = self.train_dataloader_2.__len__()
        
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = train_loader_len  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        # For blending objectives
        prob_mass = [self.lambd, 1-self.lambd]

        # flag to check whether sop learning criteria had met
        OPTIM_ACHIEVED = False

        # Setup iterable train_dataloader
        if isinstance(self.train_dataloader_1.sampler, DistributedSampler):
            sample_shuffler_1 = 0
            self.train_dataloader_1.sampler.set_epoch(sample_shuffler_1)
            sample_shuffler_1 += 1
        iter_train_dataloader_1 = iter(self.train_dataloader_1)

        if isinstance(self.train_dataloader_2.sampler, DistributedSampler):
            sample_shuffler_2 = 0
            self.train_dataloader_2.sampler.set_epoch(sample_shuffler_2)
            sample_shuffler_2 += 1
        iter_train_dataloader_2 = iter(self.train_dataloader_2)

        # setup objective sampling flag
        sample_index = True

        # if isinstance(self.model,Actor):
        #     freeze_model(self.model.model,self.model_param1+self.model_param2) # freeze all parameters in model except for model_param2
        # else:
        #     freeze_model(self.model,self.model_param1+self.model_param2)
        start_time = time.time()
        for epoch in range(self.epochs):

            step_bar = tqdm(
                range(train_loader_len),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model:
                self.ref_model.eval()
            if self.ref_model_2:
                self.ref_model_2.eval()
            acc_mean = 0
            loss_mean = 0

            # Train SFT and DPO in alternating manner, so implement a train_dataloader agnostic loop
            # assuming both train_dataloaders have the same length
            
            for step in range(train_loader_len): # important todo: 1) shuffle logic of dataloader, here the dataloader is not shuffled when depleted 2) alternating after each micro step or update step? 3) better definition of epoch
                # if hit max training time
                if time.time() - start_time >self.max_training_time:
                    break
                # Choose the objective to be updated
                # syncironize the objective index across processes  
                if sample_index:
                    obj_index = np.random.choice([1, 2], p=prob_mass)
                
                # clear memory cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # train backbone
                if obj_index==1:
                    if isinstance(self.model,Actor):
                        freeze_model(self.model.model,self.model_param1 + self.model_param2)
                    else:
                        freeze_model(self.model,self.model_param1+ self.model_param2)                    
                    loss_fn = self.loss_fn_1
                    try:
                        data = next(iter_train_dataloader_1)
                    except StopIteration:
                        if isinstance(self.train_dataloader_1.sampler, DistributedSampler):
                            self.train_dataloader_1.sampler.set_epoch(sample_shuffler_1)
                            sample_shuffler_1 += 1
                        iter_train_dataloader_1 = iter(self.train_dataloader_1) 
                        data = next(iter_train_dataloader_1) 

                if obj_index==2:
                    if isinstance(self.model,Actor):
                        freeze_model(self.model.model,self.model_param2) # freeze all parameters in model except for model_param2
                    else:
                        freeze_model(self.model,self.model_param2)
                    loss_fn = self.loss_fn_2 
                    try:
                        data = next(iter_train_dataloader_2)
                    except StopIteration:
                        if isinstance(self.train_dataloader_2.sampler, DistributedSampler):
                            self.train_dataloader_2.sampler.set_epoch(sample_shuffler_2)
                            sample_shuffler_2 += 1
                        iter_train_dataloader_2 = iter(self.train_dataloader_2)
                        data = next(iter_train_dataloader_2) 
                
                loss, logs_dict = self.calc_loss(loss_fn, data, obj_index)

                # sample an index in the next step ONLY if the model is updated in this step (i.e. this step is gradient_accumulation_boundary)
                sample_index = self.model.model.is_gradient_accumulation_boundary() if isinstance(self.model, Actor) else self.model.is_gradient_accumulation_boundary()

                # # DEBUG
                # if self.strategy.is_rank_0():
                #     self._log_gpu_memory_usage('before backward')
                #     self._log_gpu_memory_from_nvdia_smi('before backward')
                # t = time.time()
                self.strategy.backward(loss, self.model, self.optimizer)
                # # DEBUG
                # if self.strategy.is_rank_0():
                #     self._log_time_elapsed('backward', time.time()-t)
                # # DEBUG
                # if self.strategy.is_rank_0():
                #     self._log_gpu_memory_usage('after backward')
                #     self._log_gpu_memory_from_nvdia_smi('after backward')
                # t = time.time()
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                # # DEBUG
                # if self.strategy.is_rank_0():
                #     self._log_time_elapsed('optimizer_step', time.time()-t)
                
                # Average the loss across GPUs
                if obj_index==1:
                    reduced_loss1 = self.strategy.all_reduce(loss).item()
                    
                if obj_index ==2:
                    reduced_loss2 = self.strategy.all_reduce(loss).item()

                # Compute time elapsed since start
                elapsed_time = time.time() - start_time

                # logs/checkpoints/evaluate #todo: output both log dicts of the objs 
                eval_loss_1, eval_loss_2 = self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, obj_index)

                # Only rank 0 saves results
                if self.strategy.is_rank_0():
                    try:
                        self.train_loss1_records[epoch].append((step, reduced_loss1)) 
                    except:
                        self.train_loss1_records[epoch].append((step, float("nan"))) 
                    try:
                        self.train_loss2_records[epoch].append((step, reduced_loss2)) 
                    except:
                        self.train_loss2_records[epoch].append((step, float("nan"))) 
                    
                    self.time_elapsed[epoch].append((step, elapsed_time))
                    
                    if eval_loss_1 is not None:
                        self.eval_loss1_records[epoch].append((step, eval_loss_1)) 
                    else:
                        self.eval_loss1_records[epoch].append((step, float("nan"))) 
                    if eval_loss_2 is not None:
                        self.eval_loss2_records[epoch].append((step, eval_loss_2)) 
                    else:
                        self.eval_loss2_records[epoch].append((step, float("nan"))) 
                        
                if eval_loss_1 is not None and eval_loss_2 is not None:
                    if eval_loss_1-self.args.obj_opt_1 <= self.args.eps or eval_loss_2-self.args.obj_opt_2 <= self.args.eps:
                        OPTIM_ACHIEVED = True
                    
                    else:
                        for ref_loss_pair in self.ref_pareto_front:
                            if eval_loss_1 - ref_loss_pair[0] <= self.args.eps and eval_loss_2 - ref_loss_pair[1] <= self.args.eps:
                                OPTIM_ACHIEVED = True
                                break

                step_bar.update()
                global_step += 1

                if OPTIM_ACHIEVED:
                    break

            epoch_bar.update()

            if OPTIM_ACHIEVED:
                break

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        elif self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


    def fit_VaFF_trail(self, args):
        print(" ------------------- VaFF2 -------------------")

        # get the maximum train_loader length,
        if self.train_dataloader_1.__len__() >= self.train_dataloader_2.__len__():
            train_loader_len = self.train_dataloader_1.__len__()
        else:
            train_loader_len = self.train_dataloader_2.__len__()
        
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = train_loader_len  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        # For blending objectives
        prob_mass = [self.lambd, 1-self.lambd]

        # flag to check whether sop learning criteria had met
        OPTIM_ACHIEVED = False

        # Setup iterable train_dataloader
        if isinstance(self.train_dataloader_1.sampler, DistributedSampler):
            sample_shuffler_1 = 0
            self.train_dataloader_1.sampler.set_epoch(sample_shuffler_1)
            sample_shuffler_1 += 1
        iter_train_dataloader_1 = iter(self.train_dataloader_1)

        if isinstance(self.train_dataloader_2.sampler, DistributedSampler):
            sample_shuffler_2 = 0
            self.train_dataloader_2.sampler.set_epoch(sample_shuffler_2)
            sample_shuffler_2 += 1
        iter_train_dataloader_2 = iter(self.train_dataloader_2)

        # setup objective sampling flag
        sample_index = True

        # if isinstance(self.model,Actor):
        #     freeze_model(self.model.model,self.model_param1+self.model_param2) # freeze all parameters in model except for model_param2
        # else:
        #     freeze_model(self.model,self.model_param1+self.model_param2)
        start_time = time.time()
        for epoch in range(self.epochs):

            step_bar = tqdm(
                range(train_loader_len),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            if self.ref_model:
                self.ref_model.eval()
            if self.ref_model_2:
                self.ref_model_2.eval()
            acc_mean = 0
            loss_mean = 0

            # Train SFT and DPO in alternating manner, so implement a train_dataloader agnostic loop
            # assuming both train_dataloaders have the same length
            
            obj_index = 2
            for step in range(train_loader_len): # important todo: 1) shuffle logic of dataloader, here the dataloader is not shuffled when depleted 2) alternating after each micro step or update step? 3) better definition of epoch
                # if hit max training time
                if time.time() - start_time >self.max_training_time:
                    break
                # Choose the objective to be updated
                # syncironize the objective index across processes  
                
                # clear memory cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # train backbone
                if obj_index==1:
                    if isinstance(self.model,Actor):
                        freeze_model(self.model.model,self.model_param1) # freeze all parameters in model except for model_param2
                    else:
                        freeze_model(self.model,self.model_param1)            
                    
                    loss_fn1 = self.loss_fn_1
                    try:
                        data1 = next(iter_train_dataloader_1)
                    except StopIteration:
                        if isinstance(self.train_dataloader_1.sampler, DistributedSampler):
                            self.train_dataloader_1.sampler.set_epoch(sample_shuffler_1)
                            sample_shuffler_1 += 1
                        iter_train_dataloader_1 = iter(self.train_dataloader_1) 
                        data1 = next(iter_train_dataloader_1) 
                    
                    loss1, logs1 = self.calc_loss(loss_fn1, data1, 1)
                    loss = loss1

                if obj_index==2:
                    if isinstance(self.model,Actor):
                        freeze_model(self.model.model,self.model_param2) # freeze all parameters in model except for model_param2
                    else:
                        freeze_model(self.model,self.model_param2)
                    loss_fn1 = self.loss_fn_1
                    loss_fn2 = self.loss_fn_2 
                    try:
                        data1 = next(iter_train_dataloader_1)
                        data2 = next(iter_train_dataloader_2)
                    except StopIteration:
                        if isinstance(self.train_dataloader_1.sampler, DistributedSampler):
                            self.train_dataloader_1.sampler.set_epoch(sample_shuffler_1)
                            sample_shuffler_1 += 1
                        iter_train_dataloader_1 = iter(self.train_dataloader_1) 
                        data1 = next(iter_train_dataloader_1) 

                        if isinstance(self.train_dataloader_2.sampler, DistributedSampler):
                            self.train_dataloader_2.sampler.set_epoch(sample_shuffler_2)
                            sample_shuffler_2 += 1
                        iter_train_dataloader_2 = iter(self.train_dataloader_2)
                        data2 = next(iter_train_dataloader_2) 
                
                    loss1, logs1 = self.calc_loss(loss_fn1, data1, 1)
                    loss2, logs2 = self.calc_loss(loss_fn2, data2, 2)
                    loss = 1/self.gam*loss1+loss2

                # # DEBUG
                # if self.strategy.is_rank_0():
                #     self._log_gpu_memory_usage('before backward')
                #     self._log_gpu_memory_from_nvdia_smi('before backward')
                # t = time.time()
                self.strategy.backward(loss, self.model, self.optimizer)
                # # DEBUG
                # if self.strategy.is_rank_0():
                #     self._log_time_elapsed('backward', time.time()-t)
                # # DEBUG
                # if self.strategy.is_rank_0():
                #     self._log_gpu_memory_usage('after backward')
                #     self._log_gpu_memory_from_nvdia_smi('after backward')
                # t = time.time()
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                # # DEBUG
                # if self.strategy.is_rank_0():
                #     self._log_time_elapsed('optimizer_step', time.time()-t)
                
                # Average the loss across GPUs
                if obj_index==1:
                    reduced_loss1 = self.strategy.all_reduce(loss1).item()
                    obj_index = 2
                else:
                    reduced_loss1 = self.strategy.all_reduce(loss1).item()
                    reduced_loss2 = self.strategy.all_reduce(loss2).item()
                    obj_index = 1

                # Compute time elapsed since start
                elapsed_time = time.time() - start_time

                # logs/checkpoints/evaluate #todo: output both log dicts of the objs 
                eval_loss_1, eval_loss_2 = self.save_logs_and_checkpoints(args, global_step, step_bar, logs1, obj_index)

                # Only rank 0 saves results
                if self.strategy.is_rank_0():
                    try:
                        self.train_loss1_records[epoch].append((step, reduced_loss1)) 
                    except:
                        self.train_loss1_records[epoch].append((step, float("nan"))) 
                    try:
                        self.train_loss2_records[epoch].append((step, reduced_loss2)) 
                    except:
                        self.train_loss2_records[epoch].append((step, float("nan"))) 
                    
                    self.time_elapsed[epoch].append((step, elapsed_time))
                    
                    if eval_loss_1 is not None:
                        self.eval_loss1_records[epoch].append((step, eval_loss_1)) 
                    else:
                        self.eval_loss1_records[epoch].append((step, float("nan"))) 
                    if eval_loss_2 is not None:
                        self.eval_loss2_records[epoch].append((step, eval_loss_2)) 
                    else:
                        self.eval_loss2_records[epoch].append((step, float("nan"))) 
                        
                if eval_loss_1 is not None and eval_loss_2 is not None:
                    if eval_loss_1-self.args.obj_opt_1 <= self.args.eps or eval_loss_2-self.args.obj_opt_2 <= self.args.eps:
                        OPTIM_ACHIEVED = True
                    
                    else:
                        for ref_loss_pair in self.ref_pareto_front:
                            if eval_loss_1 - ref_loss_pair[0] <= self.args.eps and eval_loss_2 - ref_loss_pair[1] <= self.args.eps:
                                OPTIM_ACHIEVED = True
                                break

                step_bar.update()
                global_step += 1

                if OPTIM_ACHIEVED:
                    break

            epoch_bar.update()

            if OPTIM_ACHIEVED:
                break

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        elif self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def get_next_batch(self, obj_index, 
                   iter_train_dataloader_1, iter_train_dataloader_2, 
                   sample_shuffler_1, sample_shuffler_2):
            if obj_index == 1:
                loss_fn = self.loss_fn_1
                try:
                    data = next(iter_train_dataloader_1)
                except StopIteration:
                    if isinstance(self.train_dataloader_1.sampler, DistributedSampler):
                        self.train_dataloader_1.sampler.set_epoch(sample_shuffler_1)
                        sample_shuffler_1 += 1
                    iter_train_dataloader_1 = iter(self.train_dataloader_1)
                    data = next(iter_train_dataloader_1)
                return loss_fn, data, iter_train_dataloader_1, sample_shuffler_1

            elif obj_index == 2:
                loss_fn = self.loss_fn_2
                try:
                    data = next(iter_train_dataloader_2)
                except StopIteration:
                    if isinstance(self.train_dataloader_2.sampler, DistributedSampler):
                        self.train_dataloader_2.sampler.set_epoch(sample_shuffler_2)
                        sample_shuffler_2 += 1
                    iter_train_dataloader_2 = iter(self.train_dataloader_2)
                    data = next(iter_train_dataloader_2)
                return loss_fn, data, iter_train_dataloader_2, sample_shuffler_2

            else:
                raise ValueError(f"Invalid obj_index: {obj_index}")

    def get_loss_fn(self, obj_index:int):
        obj_type = getattr(self.args, f"obj_{obj_index}")
        # when both objectives are from the same objective type
        if self.args.obj_1==self.args.obj_2 and obj_index==2:
            arg_index='_2'
        else:
            arg_index=''

        if obj_type == "SFT":
            loss_fn = GPTLMLoss()
        elif obj_type == "DPO":
            loss_fn = DPOLoss(
                getattr(self.args, f"dpo_beta{arg_index}"), 
                getattr(self.args, f"dpo_label_smoothing{arg_index}"), 
                getattr(self.args, f"dpo_ipo{arg_index}")
                )
        elif obj_type == "KD":
            loss_fn = KDLoss()
        else:
            raise NotImplementedError

        return loss_fn
    

    def calc_loss(self, loss_fn:nn.Module, data, obj_index:int):
        obj_type = getattr(self.args, f"obj_{obj_index}")
        if self.args.obj_1==self.args.obj_2 and obj_index==2:
            arg_index='_2'
        else:
            arg_index=''
        if self.args.obj_1 in ["KD","DPO"] and self.args.obj_2 in ["KD","DPO"] and obj_index==2:
            ref_model = self.ref_model_2
        else:
            ref_model = self.ref_model
        
        is_aux_loss = getattr(self.args, f"aux_loss_coef_{obj_index}") > 1e-8

        if obj_type=="SFT":
            prompts_id_lens, inputs, attention_masks, _ = data
            inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
            attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

            output = self.model(
                inputs, attention_mask=attention_mask, return_output=True,
            )

            # loss function
            labels = torch.where(
                attention_mask.bool(),
                inputs,
                loss_fn.IGNORE_INDEX,
            )
            # mixtral
            if is_aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0

            for label, source_len in zip(labels, prompts_id_lens):
                label[:source_len] = loss_fn.IGNORE_INDEX

            gpt_loss = loss_fn(output.logits, labels)
            loss = gpt_loss + aux_loss * getattr(self.args, f"aux_loss_coef_{obj_index}")

            logs_dict = {"sft loss": gpt_loss.item()}
            if is_aux_loss:
                logs_dict["aux loss"] = aux_loss.item()

        elif obj_type=="DPO":
            is_nll_loss = getattr(self.args, f"dpo_nll_loss_coef{arg_index}") > 1e-8
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

            chosen_logps, rejected_logps, aux_loss, nll_loss = self.concatenated_forward(
                self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
            )
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                    self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )

            # loss function
            preference_loss, chosen_reward, reject_reward = loss_fn(
                chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            # mixtral
            if not is_aux_loss:
                aux_loss = 0
            # nll loss
            if not is_nll_loss:
                nll_loss = 0

            loss = preference_loss + aux_loss * getattr(self.args, f"aux_loss_coef_{obj_index}") + nll_loss * getattr(self.args, f"dpo_nll_loss_coef{arg_index}")

            acc = (chosen_reward > reject_reward).float().mean().item()
            # dpo logs
            logs_dict = {
                "preference loss": preference_loss.item(),
                "acc": acc,
                "chosen_reward": chosen_reward.mean().item(),
                "reject_reward": reject_reward.mean().item(),
                "lr": self.scheduler.get_last_lr()[0],
            }
            if is_nll_loss:
                logs_dict["nll_loss"] = nll_loss.item()  
        elif obj_type=="KD":
            prompts_id_len, inputs, attention_masks, _ = data
            inputs = inputs.squeeze(1).to(torch.cuda.current_device())
            attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
            output = self.model(inputs, attention_mask=attention_mask, return_output=True)

            labels = torch.where(
                attention_mask.bool(),
                inputs,
                loss_fn.IGNORE_INDEX,
            )

            for label, source_len in zip(labels, prompts_id_len):
                label[:source_len] = loss_fn.IGNORE_INDEX

            with torch.no_grad():
                teacher_logits = ref_model(inputs, attention_mask=attention_mask, return_output=True)[
                    "logits"
                ]
            loss = loss_fn(output.logits, teacher_logits, labels)

            logs_dict = {
                "distill loss": loss.item(),
                "lr": self.scheduler.get_last_lr()[0],
            }
        else:
            raise NotImplementedError
        
        return loss, logs_dict


    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._get_batch_logps(
            all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=False
        )
        chosen_logps = all_logps_sum[: chosen_ids.shape[0]]
        rejected_logps = all_logps_sum[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: chosen_ids.shape[0]].mean()

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks, prompt_id_lens * 2

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert average_log_prob == False
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[loss_masks == False] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        return logprobs_sums, logprobs_means

    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens):
        output = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._packed_get_batch_logps(
            all_logits,
            packed_input_ids,
            packed_attention_masks,
            prompt_id_lens * 2,
            packed_seq_lens,
            average_log_prob=False,
        )
        chosen_logps = all_logps_sum[: len(packed_seq_lens) // 2]
        rejected_logps = all_logps_sum[len(packed_seq_lens) // 2 :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: len(packed_seq_lens) // 2].mean()

    def _packed_get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        packed_seq_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        assert average_log_prob == False

        if self.strategy.ring_attn_group is None:
            assert logits.shape[:-1] == labels.shape
            labels = labels[:, 1:]
            logits = logits[:, :-1, :]
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        else:
            rank = self.strategy.ring_attn_rank
            total_seq_len = labels.numel()
            local_seq_len = total_seq_len // self.strategy.ring_attn_size
            local_slice = slice(rank * local_seq_len + 1, (rank + 1) * local_seq_len + 1)
            local_label = labels[:, local_slice]
            if rank == self.strategy.ring_attn_size - 1:
                # add a dummy label to the last logit
                local_label = F.pad(local_label, (0, 1), value=0)
            local_per_token_logps = torch.gather(
                logits.log_softmax(-1), dim=2, index=local_label.unsqueeze(2)
            ).squeeze(2)
            # we may not need to all_gather the entire tensor, but it's easier to implement.
            # use the flash_attn all_gather so that the all_gather has correct backward.
            per_token_logps = all_gather(local_per_token_logps, self.strategy.ring_attn_group).reshape((1, -1))
            per_token_logps = per_token_logps[:, :-1]

        loss_masks = attention_mask.clone().bool()

        index = 0
        for i, seq_len in enumerate(packed_seq_lens):
            loss_masks[0, index : index + prompt_id_lens[i]] = False
            index = index + seq_len

        loss_masks = loss_masks[:, 1:]

        logprobs_sums = []
        logprobs_means = []
        index = 0
        for i, seq_len in enumerate(packed_seq_lens):
            seq = per_token_logps[0, index : index + seq_len - 1]
            mask = loss_masks[0, index : index + seq_len - 1]
            logprobs_sums.append((seq * mask).sum())
            logprobs_means.append((seq * mask).sum() / mask.sum())
            index = index + seq_len

        return torch.stack(logprobs_sums), torch.stack(logprobs_means)


    # logs/checkpoints/evaluate 
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, obj_index=1):
        # logs
        if global_step % args.logging_steps == 0: # todo: fix this logging logic
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            if self.strategy.is_rank_0() and global_step % self.strategy.accumulated_gradient == 0: # todo: fix this logging logic
                if self._wandb is not None:
                    logs = {f"train_{obj_index}/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs_dict.items():
                        self._tensorboard.add_scalar(f"train_{obj_index}/{k}", v, global_step)

        # eval
        eval_loss_1 = None
        eval_loss_2 = None
        if global_step % args.eval_steps == 0 or global_step==1:
            eval_loss_1 = self.evaluate[self.args.obj_1](self.eval_dataloader_1, self.loss_fn_1, obj_index, global_step)
            eval_loss_2 = self.evaluate[self.args.obj_2](self.eval_dataloader_2, self.loss_fn_2, obj_index, global_step)
        # save ckpt
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)
        
        return eval_loss_1, eval_loss_2

    def evaluate_loss(self, dataloader1= None, dataloader2 = None):
        if dataloader1 is None:
            dataloader1 = self.eval_dataloader_1
        if dataloader2 is None:
            dataloader2 = self.eval_dataloader_2
        eval_loss_1 = self.evaluate[self.args.obj_1](dataloader1, self.loss_fn_1, 1, self.args.eval_steps)
        eval_loss_2 = self.evaluate[self.args.obj_2](dataloader2, self.loss_fn_2, 2, self.args.eval_steps)
        return eval_loss_1, eval_loss_2

    def dpo_evaluate(self, eval_dataloader, loss_fn, obj_index, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            acc_sum = 0
            loss_sum = 0
            times = 0
            for data in eval_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_logps, rejected_logps, aux_loss, _ = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )
                with torch.no_grad():
                    reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                        self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )

                loss, chosen_reward, reject_reward = loss_fn(
                    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                )
                acc_sum += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                times += 1
                step_bar.update()

            logs = {
                "eval dpo loss": loss_sum / times,
                "eval dpo acc": acc_sum / times,
            }
            logs = self.strategy.all_reduce(logs)
            step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {f"eval_{obj_index}/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None: 
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval_{obj_index}/{k}", v, steps)
        self.model.train()  # reset model state

        # return logs[f"eval_{obj_index}/dpo_loss"] if self._wandb is not None and self.strategy.is_rank_0() else logs["dpo_loss"]
        return logs[f"eval_{obj_index}/dpo_loss"] if self._wandb is not None and self.strategy.is_rank_0() else logs["eval dpo loss"]
    
    def sft_evaluate(self, eval_dataloader, loss_fn, obj_index, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts_id_lens, inputs, attention_masks, infos in eval_dataloader:
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                output = self.model(
                    inputs, attention_mask=attention_mask, return_output=True,
                )

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    loss_fn.IGNORE_INDEX,
                )

                for label, source_len in zip(labels, prompts_id_lens):
                    label[:source_len] = loss_fn.IGNORE_INDEX

                loss = loss_fn(output.logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval sft loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {f"eval_{obj_index}/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None and self.strategy.is_rank_0(): 
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval_{obj_index}/{k}", v, steps)
        self.model.train()  # reset model state

        # return logs[f"eval_{obj_index}/sft_loss"] if self._wandb is not None and self.strategy.is_rank_0() else logs["sft_loss"]
        return logs[f"eval_{obj_index}/sft_loss"] if self._wandb is not None and self.strategy.is_rank_0() else logs["eval sft loss"]
    
    
    def kd_evaluate(self, eval_dataloader, loss_fn, obj_index, steps=0):
        if self.args.obj_1 in ["KD","DPO"] and self.args.obj_2 in ["KD","DPO"] and obj_index==2:
            ref_model = self.ref_model_2
        else:
            ref_model = self.ref_model
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts_id_len, inputs, attention_masks, _ in eval_dataloader:
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                output = self.model(inputs, attention_mask=attention_mask, return_output=True)

                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    loss_fn.IGNORE_INDEX,
                )

                for label, source_len in zip(labels, prompts_id_len):
                    label[:source_len] = loss_fn.IGNORE_INDEX

                teacher_logits = ref_model(inputs, attention_mask=attention_mask, return_output=True)["logits"]
                loss = loss_fn(output.logits, teacher_logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval distill loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {f"eval_{obj_index}/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None: #todo: add support for tensorboard
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval_{obj_index}/{k}", v, steps)

        self.model.train()  # reset model state

    # DEBUG #todo: add tensorboard support
    def _log_gpu_memory_usage(self, checkpoint):
        max_allocated = torch.cuda.max_memory_allocated()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()

        self._wandb.log({
            f"GPU Memory Allocated @ {checkpoint} (MB)": allocated / (1024 ** 2),
            f"Max. GPU Memory Allocated @ {checkpoint} (MB)": max_allocated / (1024 ** 2),
            # f"GPU Memory Reserved @ {checkpoint} (MB)": reserved / (1024 ** 2)
        })

    # DEBUG
    def _log_time_elapsed(self, checkpoint, t):

        self._wandb.log({
            f"Time @ {checkpoint} (s)": t,
            # f"GPU Memory Reserved @ {checkpoint} (MB)": reserved / (1024 ** 2)
        })

    # DEBUG
    def _log_gpu_memory_from_nvdia_smi(self, checkpoint, gpus=[0, 1]):
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
        try:
            memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
        except sp.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        memory_use_values = np.array([int(x.split()[0]) for i, x in enumerate(memory_use_info)])[gpus]

        self._wandb.log({
            f"NVIDIA-SMI Max. Memory-Usage @ {checkpoint} (MB)": np.max(memory_use_values),
            f"NVIDIA-SMI Avg. Memory-Usage @ {checkpoint} (MB)": np.mean(memory_use_values)
        })
        
        for i, mem in enumerate(memory_use_values):
            self._wandb.log({
                f"NVIDIA-SMI GPU {i} Memory-Usage @ {checkpoint} (MB)": mem,
            })

    # def report_DPO_performance(self, dataloader=None):
    #     """
    #     Evaluates DPO performance on the given dataloader with progress tracking.
        
    #     Args:
    #         dataloader: DPO dataloader containing chosen/rejected pairs
            
    #     Returns:
    #         dict: {
    #             'avg_reward_gap': average difference between chosen and rejected rewards,
    #             'preference_accuracy': % of times model agrees with human preference,
    #             'avg_chosen_reward': average reward for chosen responses,
    #             'avg_rejected_reward': average reward for rejected responses,
    #             'total_samples': number of examples evaluated
    #         }
    #     """
    #     if dataloader is None:
    #         dataloader = self.train_dataloader_1

    #     total_reward_gap = 0.0
    #     total_correct = 0
    #     total_chosen_reward = 0.0
    #     total_rejected_reward = 0.0
    #     total_samples = 0

    #     self.model.eval()
    #     if self.ref_model:
    #         self.ref_model.eval()

    #     # Initialize progress bar
    #     progress_bar = tqdm(
    #         dataloader,
    #         desc="Evaluating DPO",
    #         disable=not self.strategy.is_rank_0(),
    #         dynamic_ncols=True
    #     )

    #     with torch.no_grad():
    #         for batch in progress_bar:
    #             # Unpack batch
    #             chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = batch
    #             chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
    #             c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
    #             reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
    #             r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

    #             # Forward passes
    #             chosen_logps, rejected_logps, _, _ = self.concatenated_forward(
    #                 self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
    #             )
    #             ref_chosen_logps, ref_rejected_logps, _, _ = self.concatenated_forward(
    #                 self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
    #             )

    #             # Calculate metrics
    #             chosen_rewards = chosen_logps - ref_chosen_logps
    #             rejected_rewards = rejected_logps - ref_rejected_logps
    #             reward_gaps = chosen_rewards - rejected_rewards
    #             correct_preferences = (reward_gaps > 0).float()

    #             # Update totals
    #             batch_size = chosen_ids.size(0)
    #             total_reward_gap += reward_gaps.sum().item()
    #             total_correct += correct_preferences.sum().item()
    #             total_chosen_reward += chosen_rewards.sum().item()
    #             total_rejected_reward += rejected_rewards.sum().item()
    #             total_samples += batch_size

    #             # Update progress bar description
    #             if self.strategy.is_rank_0():
    #                 current_acc = total_correct / total_samples
    #                 current_gap = total_reward_gap / total_samples
    #                 progress_bar.set_postfix({
    #                     'acc': f"{current_acc:.1%}",
    #                     'gap': f"{current_gap:.2f}",
    #                     'samples': total_samples
    #                 })

    #     metrics = {
    #         'avg_reward_gap': total_reward_gap / total_samples,
    #         'preference_accuracy': total_correct / total_samples,
    #         'avg_chosen_reward': total_chosen_reward / total_samples,
    #         'avg_rejected_reward': total_rejected_reward / total_samples,
    #         'total_samples': total_samples
    #     }

    #     # Reduce metrics across processes if distributed
    #     if hasattr(self.strategy, 'all_reduce'):
    #         metrics = self.strategy.all_reduce(metrics)

    #     # Save results
    #     if self.strategy.is_rank_0():
    #         report_path = os.path.join(self.args.save_path, "DPO_report.txt")
    #         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
    #         report_content = f"""
    #         DPO Evaluation Report ({timestamp} for stage {getattr(self.args,"model_stage","unknown")})
    #         {'='*40}
    #         - Preference Accuracy: {metrics['preference_accuracy']:.1%}
    #         - Average Reward Gap: {metrics['avg_reward_gap']:.2f}
    #         - Avg Chosen Reward: {metrics['avg_chosen_reward']:.2f}
    #         - Avg Rejected Reward: {metrics['avg_rejected_reward']:.2f}
    #         - Samples Evaluated: {metrics['total_samples']}
    #         {'='*40}
    #         Additional Info:
    #         - Model: {self.model.__class__.__name__}
    #         - Ref Model: {self.ref_model.__class__.__name__ if self.ref_model else 'None'}
    #         - Device: {torch.cuda.current_device()}\n\n
    #         """

    #         os.makedirs(self.args.save_path, exist_ok=True)
    #         with open(report_path, 'w') as f:
    #             f.write(report_content)
                
    #         print(f"\nReport saved to {report_path}")

    #     return metrics
    def report_SFT_performance(self, dataloader=None):
        """
        Evaluates SFT performance and saves results to SFT_report.txt
        
        Args:
            dataloader: SFT dataloader containing chosen/rejected pairs
            
        Returns:
            dict: Metrics dictionary (also saved to file)
        """
        if dataloader is None:
            dataloader = self.train_dataloader_2_2
        tokenizer=self.tokenizer

        self.model.eval()

        # Initialize accumulators
        total_samples = 0
        total_loss = 0.0
        total_bleu = 0.0
        total_perplexity = 0.0
        
        # For BLEU calculation
        references = []
        hypotheses = []
        
        # Get the loss function
        loss_fn = self.loss_fn_2  # Assuming you have this defined


        with torch.no_grad():
            step_bar = tqdm(
                range(len(dataloader)),
                desc="Evaluating SFT",
                disable=not self.strategy.is_rank_0(),
            )

            for batch in dataloader:
                prompts_id_lens, inputs, attention_masks, infos = batch
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                
                # Forward pass
                output = self.model(
                    inputs, 
                    attention_mask=attention_mask, 
                    return_output=True,
                )
                
                # Calculate loss
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    loss_fn.IGNORE_INDEX,
                )
                
                for label, source_len in zip(labels, prompts_id_lens):
                    label[:source_len] = loss_fn.IGNORE_INDEX
                    
                loss = loss_fn(output.logits, labels)
                perplexity = torch.exp(loss).item()
                
                # Update accumulators
                batch_size = 1 #inputs.size(0)
                total_samples += batch_size
                total_loss += loss.item() * batch_size
                total_perplexity += perplexity * batch_size
                
                # Calculate BLEU if tokenizer is available
                # Inside your batch loop:
                
                if tokenizer is not None:
                    preds = torch.argmax(output.logits, dim=-1)
                    for i in range(batch_size):
                        # Get reference and hypothesis tokens
                        ref_tokens = tokenizer.convert_ids_to_tokens(
                            inputs[i][attention_mask[i].bool()].tolist(),
                            skip_special_tokens=True
                        )
                        hyp_tokens = tokenizer.convert_ids_to_tokens(
                            preds[i][attention_mask[i].bool()].tolist(),
                            skip_special_tokens=True
                        )

                        # Skip empty sequences
                        if not ref_tokens or not hyp_tokens:
                            continue

                        references.append([ref_tokens])  # Note: List of lists
                        hypotheses.append(hyp_tokens)

                        # Calculate sentence BLEU with smoothing
                        
                        smoother = SmoothingFunction().method1
                        try:
                            sent_bleu = sentence_bleu(
                                [ref_tokens],
                                hyp_tokens,
                                smoothing_function=smoother
                            )
                            total_bleu += sent_bleu
                        except:
                            continue  # Skip if calculation fails
                        
                step_bar.set_postfix({
                    'loss': f"{total_loss/total_samples:.4f}",
                    'ppl': f"{total_perplexity/total_samples:.2f}",
                    'bleu': f"{total_bleu/max(1,total_samples):.2f}" if self.tokenizer else 'N/A'
                })
                step_bar.update()

        # Calculate corpus BLEU if we have references
        # Calculate corpus BLEU if we have data
        if tokenizer is not None and len(references) > 0:
            try:
                corpus_bleu_score = nltk_corpus_bleu(
                    references,
                    hypotheses,
                    smoothing_function=SmoothingFunction().method1
                )
            except:
                corpus_bleu_score = 0.0

        # Metrics dictionary
        metrics = {
            'avg_loss': total_loss / total_samples,
            'avg_perplexity': total_perplexity / total_samples,
            'avg_sentence_bleu': total_bleu / max(1, len(references)),
            'corpus_bleu': corpus_bleu_score,
            'total_samples': total_samples
        }

        # Reduce metrics across processes if distributed
        if hasattr(self.strategy, 'all_reduce'):
            metrics = self.strategy.all_reduce(metrics)

        # Save report on rank 0
        if self.strategy.is_rank_0():
            report_path = os.path.join(self.args.save_path, "SFT_report.txt")
            os.makedirs(self.args.save_path, exist_ok=True)
            
            with open(report_path, 'a') as f:
                f.write(f"SFT Evaluation Report\n{'='*40} for model stage {self.args.model_stage} at Evaluation Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n")
                f.write(f"Average Loss: {metrics['avg_loss']:.4f}\n")
                f.write(f"Average Perplexity: {metrics['avg_perplexity']:.2f}\n")
                if self.tokenizer:
                    f.write(f"Average Sentence BLEU: {metrics['avg_sentence_bleu']:.4f}\n")
                    f.write(f"Corpus BLEU: {metrics['corpus_bleu']:.4f}\n")
                f.write(f"Total Samples: {metrics['total_samples']}\n")
                f.write(f"\n")

            print(f"\nEvaluation complete. Report saved to {report_path}")

        return metrics



    def report_DPO_performance(self, dataloader=None):
        """
        Evaluates DPO performance and saves results to DPO_report.txt
        
        Args:
            dataloader2: DPO dataloader containing chosen/rejected pairs
            
        Returns:
            dict: Metrics dictionary (also saved to file)
        """
        if dataloader is None:
            dataloader = self.train_dataloader_1

        self.model.eval()
        if self.ref_model:
            self.ref_model.eval()

        # Initialize accumulators
        total_samples = 0
        total_correct = 0.0
        total_reward_gap = 0.0
        total_chosen_reward = 0.0
        total_rejected_reward = 0.0
        total_loss = 0.0

        # Get the loss function (assuming it's defined similarly to your example)
        loss_fn = self.loss_fn_1 

        with torch.no_grad():
            # Create progress bar
            step_bar = tqdm(
                range(len(dataloader)),
                desc="Evaluating DPO",
                disable=not self.strategy.is_rank_0(),
            )

            for data in dataloader:
                # Process batch
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                # Forward passes
                chosen_logps, rejected_logps, aux_loss, _ = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )
                reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                    self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )

                # Calculate metrics
                loss, chosen_reward, reject_reward = loss_fn(
                    chosen_logps, rejected_logps, 
                    reference_chosen_logps, reference_rejected_logps
                )
                # chosen_reward = chosen_logps-reference_chosen_logps
                # reject_reward = rejected_logps-reference_rejected_logps
                batch_correct = (chosen_reward > reject_reward).float().mean().item()
                batch_reward_gap = (chosen_reward - reject_reward).mean().item()

                # Update accumulators
                batch_size = chosen_ids.size(0)
                total_samples += batch_size
                total_correct += batch_correct * batch_size
                total_reward_gap += batch_reward_gap * batch_size
                total_chosen_reward += chosen_reward.mean().item() * batch_size
                total_rejected_reward += reject_reward.mean().item() * batch_size
                total_loss += loss.item() * batch_size

                # Update progress bar
                step_bar.set_postfix({
                    'acc': f"{total_correct/total_samples:.1%}",
                    'gap': f"{total_reward_gap/total_samples:.2f}",
                    'loss': f"{total_loss/total_samples:.4f}"
                })
                step_bar.update()

        # Calculate final metrics
        metrics = {
            'preference_accuracy': total_correct / total_samples,
            'avg_reward_gap': total_reward_gap / total_samples,
            'avg_chosen_reward': total_chosen_reward / total_samples,
            'avg_rejected_reward': total_rejected_reward / total_samples,
            'avg_loss': total_loss / total_samples,
            'total_samples': total_samples
        }

        # Reduce metrics across processes if distributed
        if hasattr(self.strategy, 'all_reduce'):
            metrics = self.strategy.all_reduce(metrics)

        # Save report on rank 0
        if self.strategy.is_rank_0():
            report_path = os.path.join(self.args.save_path, "DPO_report.txt")
            os.makedirs(self.args.save_path, exist_ok=True)
            
            with open(report_path, 'a') as f:
                f.write(f"DPO Evaluation Report\n{'='*40} for model stage {self.args.model_stage} at Evaluation Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n")
                f.write(f"Preference Accuracy: {metrics['preference_accuracy']:.1%}\n")
                f.write(f"Average Reward Gap: {metrics['avg_reward_gap']:.4f}\n")
                f.write(f"Avg Chosen Reward: {metrics['avg_chosen_reward']:.4f}\n")
                f.write(f"Avg Rejected Reward: {metrics['avg_rejected_reward']:.4f}\n")
                f.write(f"Average Loss: {metrics['avg_loss']:.4f}\n")
                f.write(f"Total Samples: {metrics['total_samples']}\n")
                f.write(f"\n")

            print(f"\nEvaluation complete. Report saved to {report_path}")


        return metrics

    def evaluate_semantic(self, dataloader1=None, dataloader2=None):
        """
        go through SFT in dataloader 1; DPO in dataloader 2
        """
        if dataloader1 is None:
            dataloader1 = self.train_dataloader_1
        if dataloader2 is None:
            dataloader2 = self.train_dataloader_2

        eval_loss_1, eval_loss_2 = self.evaluate_loss(dataloader1, dataloader2)


        
    


    def restart_record(self):
        self.train_loss1_records = [[] for _ in range(self.args.max_epochs)]
        self.train_loss2_records = [[] for _ in range(self.args.max_epochs)]

        self.eval_loss1_records = [[] for _ in range(self.args.max_epochs)]
        self.eval_loss2_records = [[] for _ in range(self.args.max_epochs)]

        self.time_elapsed = [[] for _ in range(self.args.max_epochs)]

    def save_training_records(self,SFT_mode=False):
        # Define file names
        
        if SFT_mode:
            filenames = {
                "train_loss1": "SFT_train_loss1.txt",
                "train_loss2": "SFT_train_loss2.txt",
                "eval_loss1": "SFT_eval_loss1.txt",
                "eval_loss2": "SFT_eval_loss2.txt",
                "time_elapsed": "SFT_time_elapsed.txt"
            }
        else:
            filenames = {
                "train_loss1": "train_loss1.txt",
                "train_loss2": "train_loss2.txt",
                "eval_loss1": "eval_loss1.txt",
                "eval_loss2": "eval_loss2.txt",
                "time_elapsed": "time_elapsed.txt"
            }


        # Define corresponding record attributes
        records = {
            "train_loss1": self.train_loss1_records,
            "train_loss2": self.train_loss2_records,
            "eval_loss1": self.eval_loss1_records,
            "eval_loss2": self.eval_loss2_records,
            "time_elapsed": self.time_elapsed
        }

        # Save each record
        for key, filename in filenames.items():
            filepath = os.path.join(self.args.save_path, filename)
            
            with open(filepath, "w") as f:
                for epoch in range(len(records[key])):
                    # Write epoch header
                    f.write(f"Epoch {epoch}\n")
                    steps = records[key][epoch]
                    
                    for i in range(len(steps)):
                        # Write each step
                        try:
                            step, value = steps[i]
                            value = float('nan') if value is None else value

                            if key == "time_elapsed":
                                # Format time as seconds
                                # print(value,type(value))
                                f.write(f"  Step {step:04d}: {value:.2f} sec since start\n")
                            else:
                                # Format loss
                                # print(steps[i])
                                f.write(f"  Step {step:04d}: loss = {value:.6f}\n")
                        except:
                            value = steps[i]
                            value = float('nan') if value is None else value
                            if key == "time_elapsed":
                                # Format time as seconds
                                f.write(f"  Invalid tuple: {value:.2f} sec since start\n")
                            else:
                                # Format loss
                                f.write(f"  Invalid tuple: loss = {value:.6f}\n")

            
            print(f"Saved {filename} to {filepath}")


    # def record_semantic_examples(self, args, num_samples=5):
    #     """
    #     Record human-readable examples from both objectives (training and evaluation)
    #     and save them to a text file in args.save_path.
    #     """
    #     if not self.strategy.is_rank_0():
    #         return  # Only rank 0 records samples

    #     os.makedirs(args.save_path, exist_ok=True)
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = os.path.join(args.save_path, f"semantic_examples_{timestamp}.txt")
        
    #     with open(filename, 'w') as f:
    #         f.write("--------------- Semantic Examples Record ---------------\n\n")
    #         f.write(f"Configuration:\n")
    #         f.write(f"- Objective 1: {args.obj_1} (Dataset: {args.dataset_1})\n")
    #         f.write(f"- Objective 2: {args.obj_2} (Dataset: {args.dataset_2})\n\n")
            
    #         # Record training samples
    #         f.write("=== TRAINING EXAMPLES ===\n")
            
    #         # Helper function to get examples from dataset
    #         def get_examples(dataset, is_dpo=False):
    #             examples = []
    #             try:
    #                 if hasattr(dataset, 'prompts') and hasattr(dataset, 'responses'):  # SFTDataset
    #                     for i in range(min(num_samples, len(dataset.prompts))):
    #                         examples.append({
    #                             'input': dataset.prompts[i],
    #                             'output': dataset.responses[i]
    #                         })
    #                 elif hasattr(dataset, 'prompts') and hasattr(dataset, 'chosens') and hasattr(dataset, 'rejects'):  # RewardDataset
    #                     for i in range(min(num_samples, len(dataset.prompts))):
    #                         examples.append({
    #                             'prompt': dataset.prompts[i],
    #                             'chosen': dataset.chosens[i],
    #                             'rejected': dataset.rejects[i]
    #                         })
    #                 elif hasattr(dataset, 'dataset'):  # Wrapped dataset
    #                     return get_examples(dataset.dataset, is_dpo)
    #                 return examples
    #             except Exception as e:
    #                 f.write(f"Error getting examples: {str(e)}\n")
    #                 return []

    #         # Objective 1 (DPO) training samples
    #         f.write(f"\n>> Objective 1 ({args.obj_1}) Training Examples:\n")
    #         examples = get_examples(self.train_dataloader_1.dataset, is_dpo=True)
    #         for i, example in enumerate(examples):
    #             f.write(f"\nExample {i+1} (train):\n")
    #             f.write(f"Prompt: {example.get('prompt', 'N/A')}\n")
    #             f.write(f"Chosen: {example.get('chosen', 'N/A')}\n")
    #             f.write(f"Rejected: {example.get('rejected', 'N/A')}\n")
    #             f.write("-"*50 + "\n")
            
    #         # Objective 2 (SFT) training samples
    #         f.write(f"\n>> Objective 2 ({args.obj_2}) Training Examples:\n")
    #         examples = get_examples(self.train_dataloader_2.dataset)
    #         for i, example in enumerate(examples):
    #             f.write(f"\nExample {i+1} (train):\n")
    #             f.write(f"Input: {example.get('input', 'N/A')}\n")
    #             f.write(f"Output: {example.get('output', 'N/A')}\n")
    #             f.write("-"*50 + "\n")
            
    #         # Record evaluation samples
    #         f.write("\n=== EVALUATION EXAMPLES ===\n")
            
    #         # Objective 1 (DPO) evaluation samples
    #         if hasattr(self, 'eval_dataloader_1') and self.eval_dataloader_1 is not None:
    #             f.write(f"\n>> Objective 1 ({args.obj_1}) Evaluation Examples:\n")
    #             examples = get_examples(self.eval_dataloader_1.dataset, is_dpo=True)
    #             for i, example in enumerate(examples):
    #                 f.write(f"\nExample {i+1} (eval):\n")
    #                 f.write(f"Prompt: {example.get('prompt', 'N/A')}\n")
    #                 f.write(f"Chosen: {example.get('chosen', 'N/A')}\n")
    #                 f.write(f"Rejected: {example.get('rejected', 'N/A')}\n")
    #                 f.write("-"*50 + "\n")
            
    #         # Objective 2 (SFT) evaluation samples
    #         if hasattr(self, 'eval_dataloader_2') and self.eval_dataloader_2 is not None:
    #             f.write(f"\n>> Objective 2 ({args.obj_2}) Evaluation Examples:\n")
    #             examples = get_examples(self.eval_dataloader_2.dataset)
    #             for i, example in enumerate(examples):
    #                 f.write(f"\nExample {i+1} (eval):\n")
    #                 f.write(f"Input: {example.get('input', 'N/A')}\n")
    #                 f.write(f"Output: {example.get('output', 'N/A')}\n")
    #                 f.write("-"*50 + "\n")
            
    #         f.write("\n--------------- End of Records ---------------\n")
        
    #     print(f"Saved semantic examples to {filename}")


    def record_semantic_examples(self, args, num_examples=5):
        """
        Record human-readable examples from both training and evaluation datasets
        for both objectives, along with their true labels and model outputs.
        
        Args:
            num_examples: Number of examples to record for each dataset and objective
        """
        if not self.strategy.is_rank_0():
            return  # Only record on rank 0
        
        # Create a directory for the semantic examples if it doesn't exist
        os.makedirs(args.save_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(args.save_path, f"semantic_examples_{timestamp}.txt")
        
        with open(filename, "w") as f:
            f.write("=== Semantic Examples Analysis ===\n\n")
            
            # Process both objectives
            for obj_index in [1, 2]:
                obj_type = getattr(self.args, f"obj_{obj_index}")
                f.write(f"\n==== Objective {obj_index} ({obj_type}) ====\n")
                
                # Process both training and evaluation datasets
                for dataset_type in ["train", "eval"]:
                    f.write(f"\n--- {dataset_type.capitalize()} Dataset ---\n")
                    
                    # Get the appropriate dataloader
                    if dataset_type == "train":
                        dataloader = self.train_dataloader_1 if obj_index == 1 else self.train_dataloader_2
                    else:
                        dataloader = self.eval_dataloader_1 if obj_index == 1 else self.eval_dataloader_2
                    
                    # Get a few examples from the dataloader
                    examples_collected = 0
                    for batch in dataloader:
                        if examples_collected >= num_examples:
                            break
                        
                        if obj_type in ["SFT", "KD"]:
                            # SFT/KD dataset structure
                            prompts_id_len, inputs, attention_masks, infos = batch
                            inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                            attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                            prompts = infos["input"]
                            responses = infos["output"]
                            
                            # Get model output similar to calc_loss
                            with torch.no_grad():
                                output = self.model(
                                    inputs, 
                                    attention_mask=attention_mask, 
                                    return_output=True
                                )
                            
                            # Get predicted tokens (greedy decoding)
                            preds = torch.argmax(output.logits, dim=-1)
                            
                            for i in range(len(prompts)):
                                if examples_collected >= num_examples:
                                    break
                                f.write(f"\n ---- Example {examples_collected + 1} ----: \n")
                                f.write(f"\n ********** dataset **********\n")
                                f.write(f"****Prompt****: \n{prompts[i]}\n")
                                f.write(f"****True Response****: \n{responses[i]}\n")
                                
                                # Decode model's output
                                # Skip the prompt part (using prompt_id_len)
                                response_start = prompts_id_len[i]
                                model_response = self.tokenizer.decode(
                                    preds[i][response_start:], 
                                    skip_special_tokens=True
                                )
                                f.write(f"\n ********** model response **********\n")
                                f.write(f"****Model Response****: \n{model_response}\n")
                                
                                examples_collected += 1
                        
                        elif obj_type in ["DPO", "KTO"]:
                            # DPO/KTO dataset structure
                            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = batch
                            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                            chosen_logps, rejected_logps, aux_loss, nll_loss = self.concatenated_forward(
                                self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                            )
                            with torch.no_grad():
                                reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                                    self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                                )

                            
                            # Get model's logits for chosen and rejected
                            with torch.no_grad():
                                chosen_output = self.model(
                                    chosen_ids,
                                    attention_mask=c_mask,
                                    return_output=True
                                )
                                rejected_output = self.model(
                                    reject_ids,
                                    attention_mask=r_mask,
                                    return_output=True
                                )
                            
                            # Get predicted tokens
                            chosen_preds = torch.argmax(chosen_output.logits, dim=-1)
                            rejected_preds = torch.argmax(rejected_output.logits, dim=-1)
                            
                            for i in range(chosen_ids.shape[0]):
                                if examples_collected >= num_examples:
                                    break
                                

                                f.write(f"\n ---- Example {examples_collected + 1} ----: \n")
                                
                                # Decode the full sequences
                                chosen_text = self.tokenizer.decode(chosen_ids[i], skip_special_tokens=True)
                                rejected_text = self.tokenizer.decode(reject_ids[i], skip_special_tokens=True)
                                
                                # Try to split into prompt and response
                                prompt_len = prompt_id_lens[i] if isinstance(prompt_id_lens[i], int) else 0
                                
                                # Decode model's predictions
                                chosen_model_response = self.tokenizer.decode(
                                    chosen_preds[i][prompt_len:],
                                    skip_special_tokens=True
                                )
                                rejected_model_response = self.tokenizer.decode(
                                    rejected_preds[i][prompt_len:],
                                    skip_special_tokens=True
                                )
                                
                                # Display results
                                if prompt_len > 0:
                                    prompt = self.tokenizer.decode(
                                        chosen_ids[i][:prompt_len],
                                        skip_special_tokens=True
                                    )
                                    
                                    f.write(f" \n **** Prompt ****: \n{prompt}\n")
                                    f.write(f" **** Human Chosen Response ****: \n{chosen_text[prompt_len:]}\n")
                                    f.write(f" **** Model Chosen Response ****: \n{chosen_model_response}\n")
                                    f.write(f" **** Human Rejected Response ****: \n{rejected_text[prompt_len:]}\n")
                                    f.write(f" **** Model Rejected Response ****: \n{rejected_model_response}\n")
                                else:
                                    f.write(f" **** Full Chosen Text ****: \n{chosen_text}\n")
                                    f.write(f" **** Model Chosen Prediction ****: \n{chosen_model_response}\n")
                                    f.write(f" **** Full Rejected Text ****: \n{rejected_text}\n")
                                    f.write(f" **** Model Rejected Prediction ****: \n{rejected_model_response}\n")
                                
                                examples_collected += 1
            
            f.write("\n=== End of Examples ===\n")
        
        print(f"Semantic examples saved to {filename}")