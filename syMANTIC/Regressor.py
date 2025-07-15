#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:59:07 2024

@author: muthyala.7
"""

'''
##############################################################################################

Importing the required libraries 

##############################################################################################
'''

import torch
import sys
import warnings
torch.Size([])
warnings.filterwarnings('ignore')

import itertools
import math

import time 

import torch.nn as nn

import torch.optim as optim

import pdb

import sympy as sp

import numpy as np

from syMANTIC.pareto_new import pareto


class Regressor:
    
    def __init__(self, x, y, names, complexity, dimension=None, sis_features=10, 
             device='cpu', metrics=[0.06,0.995], disp=False, quantiles=None, 
             A=None, B=None, loaded_state=None):
        """
        Initialize the symbolic regression model.

        Args:
            x: Feature tensor
            y: Target tensor
            names: Feature names
            complexity: Feature complexities
            dimension: Max equation terms
            sis_features: Top features to consider
            device: Computation device
            metrics: [rmse_metric, r2_metric]
            disp: Display progress
            quantiles: Complexity quantiles
            A: Class A weight
            B: Class B weight
            loaded_state: Previous state for resuming
        """
        self.device = device
        
        # Handle loaded state
        if loaded_state:
            self.x = loaded_state.get('x').to(self.device)
            self.y = loaded_state.get('y').to(self.device)
            self.complexity = loaded_state.get('complexity').to(self.device)
            self.names = loaded_state.get('names', [])
            self.residual = loaded_state.get('residual', torch.empty(y.shape)).to(self.device)
            self.indices = loaded_state.get('indices').to(self.device)
            self.indices_clone = loaded_state.get('indices_clone').to(self.device)
            self.earlier_pareto_rmse = loaded_state.get('earlier_pareto_rmse', torch.empty(0,)).to(self.device)
            self.earlier_pareto_r2 = loaded_state.get('earlier_pareto_r2', torch.empty(0,)).to(self.device)
            self.earlier_pareto_complexity = loaded_state.get('earlier_pareto_complexity', torch.empty(0,)).to(self.device)
            self.pareto_names = loaded_state.get('pareto_names', [])
            self.pareto_coeffs = loaded_state.get('pareto_coeffs', torch.empty(0,)).to(self.device)
            self.pareto_intercepts = loaded_state.get('pareto_intercepts', torch.empty(0,)).to(self.device)
        else:
            # Initialize fresh state
            self.x = x.to(self.device)
            self.y = y.to(self.device)
            self.complexity = complexity.to(self.device)
            self.names = names
            self.residual = torch.empty(y.shape).to(self.device)
            self.indices = torch.arange(1, (dimension*sis_features+1)).view(dimension*sis_features,1).to(self.device)
            self.indices_clone = self.indices.clone()
            self.earlier_pareto_rmse = torch.empty(0,).to(self.device)
            self.earlier_pareto_r2 = torch.empty(0,).to(self.device)
            self.earlier_pareto_complexity = torch.empty(0,).to(self.device)
            self.pareto_names = []
            self.pareto_coeffs = torch.empty(0,).to(self.device)
            self.pareto_intercepts = torch.empty(0,).to(self.device)

        self.A = A
        self.B = B
        
        # Standardization
        if not loaded_state:
            self.x_mean = self.x.mean(dim=0)
            self.x_std = self.x.std(dim=0)
            self.y_mean = self.y.mean()
            #self.y_centered = self.y - self.y_mean
            self.y_centered = self.y
            #self.x_standardized = ((self.x - self.x_mean)/self.x_std)
            self.x_standardized = self.x
        else:
            self.x_mean = loaded_state.get('x_mean').to(self.device)
            self.x_std = loaded_state.get('x_std').to(self.device)
            self.y_mean = loaded_state.get('y_mean').to(self.device)
            self.y_centered = loaded_state.get('y_centered', self.y).to(self.device)
            self.x_standardized = loaded_state.get('x_standardized', self.x).to(self.device)

        self.scores = []
        self.x_std_clone = torch.clone(self.x_standardized)
        self.rmse_metric = metrics[0]
        self.r2_metric = metrics[1]
        
        if self.x.shape[1] > 1000: 
            self.sis_features1 = 1000
        else: 
            self.sis_features1 = self.x.shape[1]
        
        self.disp = disp
        
        if quantiles != None: 
            self.quantiles = quantiles
        else: 
            self.quantiles = [0.10, 0.20, 0.3, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
        
        if dimension != None: 
            self.dimension = dimension
            self.sis_features = sis_features
        else: 
            self.dimension = 3
            self.sis_features = 10
    def higher_dimension(self,iteration):

        #Indices values that needs to be assinged zero 
        '''
        

        self.x_standardized[:,ind.tolist()] = 0
        

        scores= torch.abs(torch.mm(self.residual,self.x_standardized))

        scores[torch.isnan(scores)] = 0

        self.x_standardized[:,ind.tolist()] = self.x_std_clone[:,ind.tolist()]
        '''
        ind = (self.indices[:,-1][~torch.isnan(self.indices[:,-1])]).to(self.device)
        int_indices = [int(i) for i in ind.tolist()]
        self.x_standardized[:, int_indices] = 0
        scores = torch.abs(torch.mm(self.residual, self.x_standardized))
        scores[torch.isnan(scores)] = 0
        self.x_standardized[:, int_indices] = self.x_std_clone[:, int_indices]

        
        quantile_values = torch.quantile(self.complexity, torch.tensor(self.quantiles))
        
        '''
        
        try:
            
            quantile_values = torch.quantile(self.complexity, torch.tensor(quantiles))
            
        except:
            
            print('********************* Changing to manual partitions because of large tensor size which hinders at quantile calculation **************************************** \n')
        
            bins, digitized = (lambda t, n: (b := torch.arange(t.min(), t.max() + (w := (t.max() - t.min()) / n), w).add_(1e-6), torch.bucketize(t, b)))(self.complexity, len(quantiles)-1)

            # The upper edges of the bins correspond to the quantiles
            quantile_values = bins[1:]
        '''
        
        earlier_pareto_rmse = torch.empty(0,)
        
        earlier_pareto_complexity = torch.empty(0,)
        
        for i in range(len(quantile_values)):
            
            s = time.time()
            
            self.indices = self.indices_clone
            
            if i == 0 : 
                
                ind = torch.where(self.complexity <= quantile_values[i])[0]
                
                scores1 = scores[:,ind]
                
            else: 
                
                ind = torch.where((self.complexity > quantile_values[i-1])&(self.complexity <= quantile_values[i]))[0]
                
                scores1 = scores[:,ind]
                
            if self.quantiles[i] == 1.0: 
                
                ind = torch.where(self.complexity <= quantile_values[i])[0]
                
                scores1 = scores

                
            comp1 = self.complexity#[ind]
            
            if scores1.size()[1]==0:
                
                continue
            
            try:
                
                sorted_scores, sorted_indices = torch.topk(scores1,k=self.sis_features)
                
            except:
                
                sorted_scores, sorted_indices = torch.topk(scores1,k=len(scores1))
                            
            

            sorted_indices = sorted_indices.T
            
            sorted_indices_earlier = self.indices[:((iteration-1)*self.sis_features),(iteration-1)].unsqueeze(1)
    
            sorted_indices = torch.cat((sorted_indices_earlier,sorted_indices),dim=0)
    
            if sorted_indices.shape[0] < self.indices.shape[0]:
                
                remaining = (self.sis_features*self.dimension) - int(sorted_indices.shape[0])
                
                nan = torch.full((remaining,1),float('nan')).to(self.device)
                
                sorted_indices = torch.cat((sorted_indices,nan),dim=0)
                
                self.indices = torch.cat((self.indices,sorted_indices),dim=1)
    
            else:
                
                self.indices = torch.cat((self.indices,sorted_indices),dim=1)
    
            comb1 = self.indices[:,-1][~torch.isnan(self.indices[:,-1])]
            
            combinations_generated = torch.combinations(comb1,(int(self.indices.shape[1])-1))
            
            
            y_centered_clone = self.y_centered.unsqueeze(1).repeat(len(combinations_generated.tolist()),1,1).to(self.device)
            
            #int_indices = [int(i) for i in combinations_generated.tolist()]
            '''int_indices = [int(i) for sublist in combinations_generated.tolist() for i in sublist]
            comb_tensor = self.x_standardized.T[int_indices, :]
            print("COMBTENSORSHAPE: ", comb_tensor.shape, "INT_INDICES type:", type(int_indices)) #comb_tensor.shape
            #comb_tensor = self.x_standardized.T[combinations_generated.tolist(),:]
            
            x_p = comb_tensor.unsqueeze(1).permute(0,2,1)
            '''
            combinations_generated = combinations_generated.to(torch.long)
            comb_tensor = torch.stack([
                self.x_standardized[:, comb] for comb in combinations_generated
            ])  # Shape: [num_combinations, num_samples, features_per_combination]

            x_p = comb_tensor
            comp2 = comp1[combinations_generated.to(torch.long)]
            
            comp2 = torch.sum(comp2,dim=1)
            
            comp2 = comp2+i
            
            has_nan_inf = torch.logical_or(
                torch.isnan(x_p).any(dim=1, keepdim=True).any(dim=2, keepdim=True),
                torch.isinf(x_p).any(dim=1, keepdim=True).any(dim=2, keepdim=True)
                )
            
            x_p = torch.where(has_nan_inf,torch.zeros_like(x_p),x_p)
            
            sol = None  # initialize
            try:
                sol, _, _, _ = torch.linalg.lstsq(x_p, y_centered_clone)
            except Exception as e:
                print("torch.linalg.lstsq failed:", e)
                try:
                    x2_inv = torch.linalg.pinv(x_p)
                    sol = x2_inv @ y_centered_clone
                    sol[torch.isnan(sol)] = 0
                except Exception as e2:
                    print("torch.linalg.pinv fallback also failed:", e2)
                    print("x_p shape:", x_p.shape)
                    print("y shape:", y_centered_clone.shape)
                    continue  # or continue, depending on where this is inside your loop

            # Now verify sol is valid
            if sol is None:
                print("No solution matrix; skipping this iteration")
                continue  # or continue

            try:
                predicted = torch.matmul(x_p, sol)
                residuals = y_centered_clone - predicted
                square = torch.square(residuals)
                mean = torch.mean(square, dim=1, keepdim=True)
                features_rmse = torch.sqrt(mean)[:, 0, 0]
            except Exception as e:
                print("Matrix ops failed:", e)
                print("sol shape:", sol.shape if sol is not None else "None")
                print("x_p shape:", x_p.shape)
                continue  # or continue
            
            #My modification Starts
            if self.A is not None and self.B is not None:
                print("SAIRAM 1 - FIRST BLOCK")
                eps = 1e-6  # to avoid log(0)
                cross_entropy = torch.empty_like(y_centered_clone)

                for i in range(y_centered_clone.shape[0]):
                    for j in range(y_centered_clone.shape[1]):
                        # Determine weight
                        if y_centered_clone[i,j] == 1:
                            weight = self.B / self.A
                        elif y_centered_clone[i,j] == 0:
                            weight = 1.0
                        else:
                            print("Y VALUE HAS BEEN CENTERED — unexpected shape.")
                            sys.exit()

                        # Clamp predicted to avoid log(0)
                        logits = predicted[i, j, 0]
                        p = torch.clamp(torch.sigmoid(logits), min=eps, max=1 - eps)
                        y_true = y_centered_clone[i, j, 0]

                        # Binary cross-entropy
                        ce = - (y_true * torch.log(p) + (1 - y_true) * torch.log(1 - p))
                        cross_entropy[i, j, 0] = ce * weight

                # Custom "loss" vector replacing RMSE
                features_rmse = torch.mean(cross_entropy, dim=1, keepdim=True)
            #My modification Ends.
            
            features_r2 = 1 - (torch.sum(torch.square(residuals),dim=1)/torch.sum(torch.square(self.y_centered)))
            probs = torch.sigmoid(predicted)
            preds = (probs > 0.5).float()
            features_r2 = (preds == y_centered_clone).float().mean(axis = 1, keepdims = True)
            #print("FeaturesR2 (Line 349):", features_r2)
            print("FEATURES RMSE SHAPE:", features_rmse.shape, "COMPLEXITY:", comp2.shape)           
            features_rmse = features_rmse.squeeze()
            s= pareto(features_rmse,comp2).pareto_front()
            
            coeff = torch.squeeze(sol).unsqueeze(1)
            
            coeff = coeff.squeeze(1)
            
            coeff1 = coeff.clone()
            
            combinations = combinations_generated.long()
            
            std = self.x_std[combinations]
            
            coeff = coeff/std
            
            #xx = self.x_mean[combinations_generated.to(torch.int)]
            #yy = self.x_std[combinations_generated.to(torch.int)]
            xx = self.x_mean[combinations_generated.to(torch.long)]
            yy = self.x_std[combinations_generated.to(torch.long)]
            
            nn = xx/yy
            
            ss1 = nn*coeff1
            
            ss2 = torch.sum(ss1,dim=1)
            
            non_std_intercepts = self.y.mean().repeat(coeff1.shape[0]) -  ss2
            
            self.earlier_pareto_rmse = torch.cat((self.earlier_pareto_rmse,features_rmse[s]),dim=0)
            
            #self.earlier_pareto_r2 = torch.cat((self.earlier_pareto_r2,features_r2[s].flatten()),dim=0)
            self.earlier_pareto_r2 = torch.cat((self.earlier_pareto_r2,features_r2[s].flatten()),dim=0)
            self.earlier_pareto_complexity = torch.cat((self.earlier_pareto_complexity,comp2[s]))
            
            
            if coeff.shape[1] == self.pareto_coeffs.shape[1]:
                
                self.pareto_coeffs = torch.cat((self.pareto_coeffs,coeff[s]))
            else:
                additional_columns = torch.full((self.pareto_coeffs.size(0), abs(coeff.shape[1]-self.pareto_coeffs.shape[1])), float('nan'))
                
                self.pareto_coeffs = torch.cat((self.pareto_coeffs,additional_columns),dim=1)
                
                self.pareto_coeffs = torch.cat((self.pareto_coeffs, coeff[s]))
            
            self.pareto_intercepts = torch.cat((self.pareto_intercepts,non_std_intercepts[s]))
            
            
            for comb in combinations_generated[s]:
                
                #self.pareto_names.append(np.array(self.names)[comb.to(torch.int)].tolist())
                self.pareto_names.append(
                    [self.names[int(i)] for i in comb.tolist()]  # always list of strings
                    )
                
           
        min_value, min_index = torch.min(mean, dim=0)
  
        coefs_min = torch.squeeze(sol[min_index]).unsqueeze(1)
        
        indices_min  = torch.squeeze(combinations_generated[min_index])
        
        non_std_coeff = ((coefs_min.T/self.x_std[indices_min.tolist()]))
        
        non_std_intercept = self.y.mean() - torch.dot(self.x_mean[indices_min.tolist()]/self.x_std[indices_min.tolist()],coefs_min.flatten())
        
        self.residual = self.y_centered - torch.mm(coefs_min.T,self.x_standardized[:,indices_min.tolist()].T)
        
        rmse = float(torch.sqrt(torch.mean(self.residual**2)))
        
        #My modification Starts
        if self.A is not None and self.B is not None:
                    print("SAIRAM 2 - SECOND BLOCK")

                    eps = 1e-6  # to avoid log(0)
                    cross_entropy = torch.empty_like(self.residual)

                    for i in range(self.residual.shape[0]):
                            # Determine weight
                            if self.y_centered[i] == 1:
                                weight = self.B / self.A
                            elif self.y_centered[i] == 0:
                                weight = 1.0
                            else:
                                print("Y VALUE HAS BEEN CENTERED — unexpected shape.")
                                sys.exit()

                            # Clamp predicted to avoid log(0)
                            testPred = torch.mm(coefs_min.T,self.x_standardized[:,indices_min.tolist()].T)
                            print("TESTPRED SHAPE:", testPred.shape)
                            p = torch.clamp(torch.sigmoid(testPred[0,i]), min=eps, max=1 - eps)
                            y_true = y_centered_clone[i,0]
                            #print ("XSTANDARD_SHAPE:", self.x_standardized[i,:].shape)
                            # Binary cross-entropy
                            ce = - (y_true * torch.log(p) + (1 - y_true) * torch.log(1 - p))
                            print ("CE SHAP:", ce.shape)
                            cross_entropy[i, 0] = ce * weight

                    # Custom "loss" vector replacing RMSE
                    rmse = float(torch.mean(cross_entropy))
        #My modification Ends.
        r2 = 1 - (float(torch.sum(self.residual**2))/float(torch.sum((self.y_centered)**2)))
        probs = torch.sigmoid(testPred)
        preds = (probs > 0.5).float()
        r2 = (preds == self.y_centered).float().mean()
        #print("R2 (Line 456):", r2)
        terms = []
        

        for i in range(len(non_std_coeff.squeeze())):
            
            ce = "{:.20f}".format(float(non_std_coeff.squeeze()[i]))
            
            term = str(ce) + "*" + str(self.names[int(indices_min[i])])
            
            
            terms.append(term)
            
        self.indices_clone = self.indices.clone()

        return float(rmse),terms,non_std_intercept,non_std_coeff,r2

    '''
    ##########################################################################################################################

    Defines the function to model the equation

    ##########################################################################################################################
    '''
    def regressor_fit(self):
        
        if self.x.shape[1] > self.sis_features*self.dimension:
            
            if self.disp:
                
                print()
                
                #print(f"Starting sparse model building in {self.device} \n")
            
        else:
            #print('!!Important:: Given Number of features in SIS screening is greater than the feature space created, changing the SIS features to shape of features created!!')
            
            self.sis_features = self.x.shape[1]
            
            self.indices = torch.arange(1, (self.dimension*self.sis_features+1)).view(self.dimension*self.sis_features,1).to(self.device)
            
        #Looping over the dimensions 
        for i in range(1,self.dimension+1):
            
            if i ==1:
                
                start_1D = time.time()

                #calculate the scores
                scores = torch.abs(torch.mm(self.y_centered.unsqueeze(1).T,self.x_standardized))

                #Set the NaN values claculation to zero, instead of removing 
                scores[torch.isnan(scores)] = 0

                #Sort the top number of scores based on the sis_features 
                sorted_scores, sorted_indices = torch.topk(scores,k=self.sis_features)
                
                sorted_indices = sorted_indices.T
                
                remaining = torch.tensor((self.sis_features*self.dimension) - int(sorted_indices.shape[0])).to(self.device)

                #replace the remaining indices with nan
                nan = torch.full((remaining,1),float('nan')).to(self.device)
                
                sorted_indices = torch.cat((sorted_indices,nan),dim=0)
                
                #store the sorted indices as next column
                self.indices = torch.cat((self.indices,sorted_indices),dim=1)
                
                selected_index = self.indices[0,1]
                
                quantile_values = torch.quantile(self.complexity, torch.tensor(self.quantiles))
                '''
                
                try:
                    quantile_values = torch.quantile(self.complexity, torch.tensor(quantiles))
                    
                except:
                    print('Changing to manual partitions because of large tensor size..')
                
                    bins, digitized = (lambda t, n: (b := torch.arange(t.min(), t.max() + (w := (t.max() - t.min()) / n), w).add_(1e-6), torch.bucketize(t, b)))(self.complexity, len(quantiles)-1)
    
                    # The upper edges of the bins correspond to the quantiles
                    quantile_values = bins[1:]
                '''
                earlier_pareto_rmse = torch.empty(0,)
                
                earlier_pareto_complexity = torch.empty(0,)
                
                self.earlier_pareto_rmse = torch.cat((self.earlier_pareto_rmse,torch.sqrt(torch.mean(self.y_centered**2)).unsqueeze(0)),dim=0)
                
                self.earlier_pareto_complexity = torch.cat((self.earlier_pareto_complexity,torch.tensor([0.])))
                
                self.earlier_pareto_r2 = torch.cat((self.earlier_pareto_r2,torch.tensor([0.])),dim=0)
                
                self.pareto_names.extend([str(self.y_mean.tolist())])
                
                

                for i in range(len(quantile_values)):
                    
                    
                    if i == 0 : 
                        
                        ind = torch.where(self.complexity <= quantile_values[i])[0]
                        
                        scores1 = scores[:,ind]
                        
    
                    else: 
                        
                        ind = torch.where((self.complexity > quantile_values[i-1])&(self.complexity <= quantile_values[i]))[0]
                        
                        scores1 = scores[:,ind]

                    if self.quantiles[i] == 1.0: 
                        
                        ind = torch.where(self.complexity <= quantile_values[i])[0]
                        
                        scores1 = scores

                    if scores1.size()[1]==0: 
                        
                        continue
                
                    try:
                        
                        sorted_scores, sorted_indices = torch.topk(scores1,k=self.sis_features)
                        
                    except:
                        
                        sorted_scores, sorted_indices = torch.topk(scores1,k=len(scores1))
                
                    selected_indices = sorted_indices.flatten()

                    comp1 = self.complexity[ind]
                    
                    names = np.array(self.names)[ind]

                    x1 = self.x_standardized[:,selected_indices]
                    
                    x2 = x1.unsqueeze(0).T

                    y1 = self.y_centered.unsqueeze(1).unsqueeze(0)
                    
                    if x2.shape[0] != y1.shape[0]:
                        
                        y1 = y1.repeat(x2.shape[0],1,1)

                    has_nan_inf = torch.logical_or(
                        torch.isnan(x2).any(dim=1, keepdim=True).any(dim=2, keepdim=True),
                        torch.isinf(x2).any(dim=1, keepdim=True).any(dim=2, keepdim=True)
                        )
                    
                    x2 = torch.where(has_nan_inf,torch.zeros_like(x2),x2)
                    #print("Test")
                    
                    try:
                        
                        sol,_,_,_ = torch.linalg.lstsq(x2,y1)
                        
                    except:
                        
                        x2_inv = torch.linalg.pinv(x2)
                        
                        sol = x2_inv@y1
                        
                        sol[torch.isnan(sol)] = 0
                    
                    
                    std = self.x_std[selected_indices].unsqueeze(0)
                    
                    non_std_sol = (sol/std)[:,0,0]
                    
                    xx = self.x_mean[selected_indices]
                    
                    yy = self.x_std[selected_indices]
                    
                    nn = xx/yy
                    
                    ss = nn*sol[:,0,0]
                    
                    non_std_intercepts = self.y.mean().repeat(len(ss)) - ss
                    
                    predicted = torch.matmul(x2,sol)
                    
                    residuals = y1 - predicted
                    
                    square = torch.square(residuals)
                    mean = torch.mean(square,dim=1,keepdim=True)
                    features_rmse = torch.sqrt(mean)[:,0,0]
                    '''

                    sol = None  # initialize
                    try:
                        sol, _, _, _ = torch.linalg.lstsq(x_p, y_centered_clone)
                    except Exception as e:
                        print("torch.linalg.lstsq failed:", e)
                        try:
                            x2_inv = torch.linalg.pinv(x_p)
                            sol = x2_inv @ y_centered_clone
                            sol[torch.isnan(sol)] = 0
                        except Exception as e2:
                            print("torch.linalg.pinv fallback also failed:", e2)
                            print("x_p shape:", x_p.shape)
                            print("y shape:", y_centered_clone.shape)
                            return  # or continue, depending on where this is inside your loop

                    # Now verify sol is valid
                    if sol is None:
                        print("No solution matrix; skipping this iteration")
                        return  # or continue

                    try:
                        predicted = torch.matmul(x_p, sol)
                        residuals = y_centered_clone - predicted
                        square = torch.square(residuals)
                        meanTest = torch.mean(square, dim=1, keepdim=True)
                        features_rmse = torch.sqrt(meanTest)[:, 0, 0]
                    except Exception as e:
                        print("Matrix ops failed:", e)
                        print("sol shape:", sol.shape if sol is not None else "None")
                        print("x_p shape:", x_p.shape)
                        return  # or continue
                        '''

                    #My modification Starts
                    
                    if self.A is not None and self.B is not None:
                        print("SAIRAM 3 - THIRD BLOCK")

                        eps = 1e-6  # to avoid log(0)
                        cross_entropy = torch.empty_like(y1)
                        #print(y1)
                        #print(self.y)
                        for i in range(y1.shape[0]):
                            for j in range(y1.shape[1]):
                                
                                # Determine weight
                                if y1[i,j] == 1:
                                    weight = self.B / self.A
                                elif y1[i,j] == 0:
                                    weight = 1.0
                                else:
                                    print("Y VALUE HAS BEEN CENTERED — unexpected shape.")
                                    sys.exit()

                                # Clamp predicted to avoid log(0)
                                p = torch.clamp(torch.sigmoid(predicted[i, j, 0]), min=eps, max=1 - eps)
                                y_true = y1[i, j, 0]

                                # Binary cross-entropy
                                ce = - (y_true * torch.log(p) + (1 - y_true) * torch.log(1 - p))
                                cross_entropy[i, j, 0] = ce * weight

                        # Custom "loss" vector replacing RMSE
                        
                        features_rmse = torch.mean(cross_entropy, dim=1, keepdim=True)
                    #My modification Ends.
                    
                    features_r2 = 1 - (torch.sum(torch.square(residuals),dim=1)/torch.sum(torch.square(self.y_centered)))
                    print("Pre-Code FEATURES R2.shape:",features_r2.shape)
                    probs = torch.sigmoid(predicted[:,:,0])
                    preds = (probs>0.5).float()
                    preds = preds.unsqueeze(-1)
                    print("PREDS_SHAPE:",preds.shape)
                    print("Y1_SHAPE:",y1.shape)
                    origfeatures_r2 = (preds == y1).float()
                    print("PRED MATCH SHAPE:", origfeatures_r2.shape)
                    features_r2 = (preds == y1).float().mean(dim=1, keepdim=True)
                    print("MEANfeatures_R2:",features_r2)
                    features_rmse = features_rmse.squeeze()
                    '''print("FEATURES RMSE:",features_rmse,"COMP1:", comp1[selected_indices])
                    print("SHAPE FEATURES RMSE:",features_rmse.shape(),"COMP1:", comp1[selected_indices].shape())
                    '''
                    s= pareto(features_rmse,comp1[selected_indices]).pareto_front()
                    
                    features_rmse = features_rmse.view(-1)
                    print("Earlier Pareto RMSE SHAPE:", self.earlier_pareto_rmse.shape)
                    print("Features RMSE SHAPE:", features_rmse.shape)
                    self.earlier_pareto_rmse = torch.cat((self.earlier_pareto_rmse,features_rmse[s]),dim=0)
                    print("FeaturesR2 Shape:", features_r2.shape)
                    #self.earlier_pareto_r2 = torch.cat((self.earlier_pareto_r2,features_r2[s].flatten()),dim=0)
                    self.earlier_pareto_r2 = torch.cat((self.earlier_pareto_r2,features_r2[s].flatten()),dim=0)
                    self.earlier_pareto_complexity = torch.cat((self.earlier_pareto_complexity,comp1[selected_indices[s]]))
                    
                    self.pareto_names.extend(np.array(self.names)[selected_indices.numpy()[s]].tolist())
                    
                    if non_std_sol[s].dim() ==1: 
                        coeff_ad = non_std_sol[s].unsqueeze(1)
                        
                        
                    else:coeff_ad = non_std_sol[s]
                    
                    
                    self.pareto_coeffs = torch.cat((self.pareto_coeffs,coeff_ad))
                    
                    self.pareto_intercepts = torch.cat((self.pareto_intercepts,non_std_intercepts[s]))

                x_in = self.x[:, int(selected_index)].unsqueeze(1)

                # Add a column of ones to x for the bias term
                x_with_bias = torch.cat((torch.ones_like(x_in), x_in), dim=1).to(self.device)

                #Calculate the intercept and coefficient, Non standardized
                coef1, _, _, _ = torch.linalg.lstsq(x_with_bias, self.y)

                #Calculate the residuals based on the standardized and centered values
                x_in1 = self.x_standardized[:, int(selected_index)].unsqueeze(1)

                # Add a column of ones to x for the bias term
                x_with_bias1 = torch.cat((torch.ones_like(x_in1), x_in1), dim=1)
                
                coef, _, _, _ = torch.linalg.lstsq(x_with_bias1, self.y_centered)
                
                #pdb.set_trace()

                self.residual = (self.y_centered - (coef[1]*self.x_standardized[:, int(selected_index)])).unsqueeze(1).T
                
                rmse = float(torch.sqrt(torch.mean(self.residual**2)))

                #My modification Starts
                
                if self.A is not None and self.B is not None:
                    print("SAIRAM 4 - FOURTH BLOCK")

                    eps = 1e-6  # to avoid log(0)
                    cross_entropy = torch.empty_like(self.residual)

                    for i in range(self.residual.shape[0]):
                            # Determine weight
                            if self.y_centered[i] == 1:
                                weight = self.B / self.A
                            elif self.y_centered[i] == 0:
                                weight = 1.0
                            else:
                                print("Y VALUE HAS BEEN CENTERED — unexpected shape.")
                                sys.exit()

                            # Clamp predicted to avoid log(0)
                            logits = (coef[1]*self.x_standardized[:, int(selected_index)])[i]
                            p = torch.clamp(torch.sigmoid(logits), min=eps, max=1 - eps)
                            y_true = self.y_centered[i]

                            # Binary cross-entropy
                            ce = - (y_true * torch.log(p) + (1 - y_true) * torch.log(1 - p))
                            cross_entropy[i, 0] = ce * weight

                    # Custom "loss" vector replacing RMSE
                    rmse = float(torch.mean(cross_entropy))
                #My modification Ends.
                
                r2 = 1 - (float(torch.sum(self.residual**2))/float(torch.sum((self.y_centered)**2)))
                probs = torch.sigmoid((coef[1]*self.x_standardized[:, int(selected_index)]))
                preds = (probs > 0.5).float()
                r2 = (preds == self.y_centered).float().mean()
                print("R2 (Line 812):", r2)
                coefficient = coef[1]

                intercept = self.y.mean() - torch.dot((self.x_mean[int(selected_index)]/self.x_std[int(selected_index)]).reshape(-1), coef[1].reshape(-1))#coef1[0]

                if intercept > 0:
                    
                    coefficient = coef[1]/self.x_std[int(selected_index)]
                    
                    coefficient = "{:.20f}".format(float(coefficient))
                    
                    equation = str(float(coefficient)) + '*' + str(self.names[int(selected_index)]) + '+' + str(float(intercept))
                    
                    '''
                    if self.disp:
                        print('Equation: ', equation)
                        
                        print('\n')
                        
                        print('RMSE: ', rmse)
                        
                        print('R2::',r2)
                    '''
                    
                    
                else:
                    
                    coefficient = coef[1]/self.x_std[int(selected_index)]
                    
                    coefficient = "{:.20f}".format(float(coefficient))
                    
                    equation = str(float(coefficient)) + '*' + str(self.names[int(selected_index)])  + str(float(intercept))
                    
                    '''
                    if self.disp:
                        print('Equation: ', equation)
                        
                        print('\n')
                        
                        print('RMSE: ', rmse)
                        
                        print('R2::',r2)
                    
                        print('Time taken to generate one dimensional equation: ', time.time()-start_1D,' seconds')
                        
                        print('\n')
                    '''
                
                if self.device == 'cuda':torch.cuda.empty_cache()
                print("SELF PARETO NAMES 3:",self.pareto_names)
                if rmse <= self.rmse_metric and r2>= self.r2_metric: return float(rmse),equation,r2,self.earlier_pareto_rmse,self.earlier_pareto_complexity,self.pareto_names,self.pareto_intercepts,self.pareto_coeffs,self.earlier_pareto_r2

                
                if self.pareto_coeffs.dim()==1: self.pareto_coeffs = self.pareto_coeffs.unsqueeze(1)
                
                
            else:
                
                start = time.time()
                
                self.indices_clone = self.indices.clone()

                rmse,terms,intercept,coefs,r2 = self.higher_dimension(i)
                
                equation =''
                
                for k in range(len(terms)):
                    
                    if coefs.flatten()[k] > 0:
                        
                        equation = equation + ' + ' + (str(terms[k]))+'  '
                        
                    else:
                        
                        equation = equation + (str(terms[k])) + '  '
                '''
                if self.disp:
                    print('Equation: ',equation[:len(equation)-1])
                    print('\n')
    
                    print('Intercept:', float(intercept))
                    print('\n')
    
                    print('RMSE:',float(rmse))
                    print('\n')
                    
                    print('R2::',r2)
    
                    print(f'Time taken for {i} dimension is: ', time.time()-start)
                '''
                
                if self.device == 'cuda': torch.cuda.empty_cache()
                
                if rmse <= self.rmse_metric and r2>= self.r2_metric: 
                    
                    #print("Intercept:",float(intercept))
                    print("PARETO NAMES 1: ",self.pareto_names)
                    return float(rmse),equation,r2,self.earlier_pareto_rmse,self.earlier_pareto_complexity,self.pareto_names,self.pareto_intercepts,self.pareto_coeffs,self.earlier_pareto_r2
        print("PARETO NAMES 2: ",self.pareto_names)
        return float(rmse),equation,r2,self.earlier_pareto_rmse,self.earlier_pareto_complexity,self.pareto_names,self.pareto_intercepts,self.pareto_coeffs,self.earlier_pareto_r2
