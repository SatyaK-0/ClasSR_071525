'''
##############################################################################################

Importing the required libraries

##############################################################################################
'''
import torch

import pandas as pd

import numpy as np

import warnings

import itertools

import time

from sklearn.feature_selection import mutual_info_regression        

from scipy.stats import spearmanr

from itertools import combinations

import pdb

from fractions import Fraction

from syMANTIC.pareto_new import pareto

from syMANTIC.Regressor import Regressor

from itertools import combinations, islice

def batched_combinations(n_features, r=2, batch_size=250):
    """
    Generator yielding batches of index tuples (combinations).
    """
    combo_iter = combinations(range(n_features), r)
    while True:
        batch = list(islice(combo_iter, batch_size))
        if not batch:
            break
        yield batch


class feature_space_construction:

  '''
  ##############################################################################################################

  Define global variables like number of operators and the input data frame and the operator set given

  ##############################################################################################################
  '''
  def __init__(self, operators, df, no_of_operators=None, device='cpu', 
             initial_screening=None, metrics=[0.06,0.995], disp=False, 
             pareto=False, dimension=3, sis_features=20, feature_names=False, 
             A=None, B=None, loaded_state=None):
    """
    Initialize the feature space construction.

    Args:
        operators: List of mathematical operators to use
        df: Input DataFrame
        no_of_operators: Maximum operator depth (None for auto)
        device: Computation device ('cpu' or 'cuda')
        initial_screening: Feature screening method
        metrics: List of [rmse_metric, r2_metric]
        disp: Display progress messages
        pareto: Return Pareto frontier
        dimension: Max terms in equations
        sis_features: Top features to consider
        feature_names: Return feature names
        A: Class A weight for weighted loss
        B: Class B weight for weighted loss
        loaded_state: Dictionary of previous state for resuming
    """
    self.no_of_operators = no_of_operators
    self.df = df
    
    # Handle loaded state
    if loaded_state:
        self.df_feature_values = loaded_state.get('df_feature_values')
        self.Target_column = loaded_state.get('Target_column')
        self.columns = loaded_state.get('columns', [])
        self.complexity = loaded_state.get('complexity')
        self.operators_final = loaded_state.get('operators_final')
        self.reference_tensor = loaded_state.get('reference_tensor')
        self.updated_pareto_rmse = loaded_state.get('updated_pareto_rmse', torch.empty(0,))
        self.updated_pareto_r2 = loaded_state.get('updated_pareto_r2', torch.empty(0,))
        self.updated_pareto_complexity = loaded_state.get('updated_pareto_complexity', torch.empty(0,))
        self.updated_pareto_names = loaded_state.get('updated_pareto_names', [])
        self.update_pareto_coeff = loaded_state.get('update_pareto_coeff', torch.empty(0,))
        self.update_pareto_intercepts = loaded_state.get('update_pareto_intercepts', torch.empty(0,))
    else:
        # Initialize fresh state
        self.df_feature_values = None
        self.Target_column = None
        self.columns = []
        self.complexity = None
        self.operators_final = torch.full((df.shape[1],), float('nan'))
        self.reference_tensor = None
        self.updated_pareto_rmse = torch.empty(0,)
        self.updated_pareto_r2 = torch.empty(0,)
        self.updated_pareto_complexity = torch.empty(0,)
        self.updated_pareto_names = []
        self.update_pareto_coeff = torch.empty(0,)
        self.update_pareto_intercepts = torch.empty(0,)

    self.operators = operators
    self.A = A
    self.B = B
    self.operators_indexing = torch.arange(0, len(self.operators)).to(device)
    self.operators_dict = dict(zip(self.operators, self.operators_indexing.tolist()))
    self.device = torch.device(device)

    # Filter dataframe
    self.df = self.df.select_dtypes(include=['float64','int64','float32','int32'])
    variance = self.df.var()
    zero_var_cols = variance[variance == 0].index
    self.df = self.df.drop(zero_var_cols, axis=1)
    
    if not loaded_state:
        print("*****************CHECKING SELF DF******************")
        print(self.df)
        self.Target_column = torch.tensor(self.df.pop('Target')).to(self.device)
    
    if initial_screening != None:
        self.screening = initial_screening[0]
        self.quantile = initial_screening[1]
        self.df = self.feature_space_screening(self.df)
        self.df.columns = self.df.columns.str.replace('-', '_')

    if not loaded_state:
        self.df_feature_values = torch.tensor(self.df.values).to(self.device)
        self.columns = self.df.columns.tolist()
        self.variables_indexing = torch.arange(0, len(self.columns)).reshape(1,-1).to(self.device)
        self.variables_dict = dict(zip(self.columns, self.variables_indexing.tolist()))
        self.reference_tensor = self.variables_indexing.clone().reshape(-1, 1)

    self.new_features_values = pd.DataFrame()
    self.feature_values_unary = torch.empty(self.df.shape[0],0).to(self.device)
    self.feature_names_unary = []
    self.feature_values_binary = torch.empty(self.df.shape[0],0).to(self.device)
    self.feature_names_binary = []
    self.rmse_metric = metrics[0]
    self.r2_metric = metrics[1]
    self.metrics = metrics
    self.pareto_points_identified = torch.empty(0,2).to(self.device)
    self.all_points_identified = torch.empty(0,2).to(self.device)
    self.p_exp = []
    self.np_exp = []
    self.disp = disp
    self.pareto = pareto
    self.dimension = dimension
    self.sis_features = sis_features
    self.feature_names = feature_names
    self.operators_final = torch.empty(0,).to(self.device)
    
    self.operators_final = torch.full((self.df.shape[1],), float('nan')).to(self.device)
    
    self.reference_tensor = self.variables_indexing.clone().reshape(-1, 1)
  def feature_space_screening(self,df_sub):

        if self.screening == 'spearman':
            
            spear = spearmanr(df_sub.to_numpy(),self.Target_column,axis=0)
            
            screen1 = abs(spear.statistic)
            
            
            
            if screen1.ndim>1:screen1 = screen1[:-1,-1]
            
        elif self.screening == 'mi':
            X = df_sub.to_numpy()
            y = self.Target_column.cpu().numpy()
            n_features = X.shape[1]
            mi_scores = []
            batch_size = 100  # or tune as needed
            for batch_slice in batched_indices(n_features, batch_size):
                batch_X = X[:, batch_slice]
                mi = mutual_info_regression(batch_X, y)
                mi_scores.extend(mi)
                del batch_X, mi
            screen1 = np.array(mi_scores)

        df_screening = pd.DataFrame()
        
        df_screening['Feature variables'] = df_sub.columns
        
        df_screening['screen1'] = screen1
        
        df_screening = df_screening.sort_values(by = 'screen1',ascending= False).reset_index(drop=True)
        
        quantile_screen=df_screening.screen1.quantile(self.quantile)
        
        filtered_df = df_screening[(df_screening.screen1 > quantile_screen)].reset_index(drop=True)
        
        if filtered_df.shape[0]==0:
            
            filtered_df = df_screening[:int(df_sub.shape[1]/2)]

        df_screening1 = df_sub.loc[:,filtered_df['Feature variables'].tolist()]
        
        return df_screening1
    
  def clean_tensor(self,tensor):
      
      mask = ~torch.isnan(tensor)
      
      counts = mask.sum(dim=1)
      
      max_count = counts.max()
      
      row_indices = torch.arange(tensor.shape[0]).unsqueeze(1).expand(-1, max_count)
      
      col_indices = torch.arange(max_count).unsqueeze(0).expand(tensor.shape[0], -1)
      
      valid_mask = col_indices < counts.unsqueeze(1)
      
      valid_elements = tensor[mask]
      
      result = torch.full((tensor.shape[0], max_count), float('nan'))
      
      result[row_indices[valid_mask], col_indices[valid_mask]] = valid_elements
      
      return result
  
  '''
  #####################################################################
  
  Single variable expansions..
  
  #####################################################################
  '''
  def single_variable(self, operators_set, i, batch_size=100):
    """
    Memory-efficient unary operator expansion: applies each op to features in batches.
    """
    self.feature_values_unary = torch.empty(self.df.shape[0], 0).to(self.device)
    self.feature_names_unary = []
    num_features = self.df_feature_values.shape[1]

    for op in operators_set:
        self.feature_values_11 = torch.empty(self.df.shape[0], 0).to(self.device)
        feature_names_12 = []
        feature_values_reference = torch.empty(0,).to(self.device)
        operators_reference = torch.empty(0,).to(self.device)

        for batch_slice in batched_indices(num_features, batch_size):
            batch_idx = list(range(batch_slice.start, batch_slice.stop))
            batch_vals = self.df_feature_values[:, batch_idx]
            batch_cols = [self.columns[j] for j in batch_idx]
            # --- Operator-specific transforms below ---

            if op == 'exp':
                transformed = torch.exp(batch_vals)
                names = [f'(exp({x}))' for x in batch_cols]
            elif op == 'ln':
                transformed = torch.log(batch_vals)
                names = [f'(ln({x}))' for x in batch_cols]
            elif op == 'log':
                transformed = torch.log10(batch_vals)
                names = [f'(log({x}))' for x in batch_cols]
            elif "pow" in op:
                import re
                pattern = r'\(([^)]*)\)'
                matches = re.findall(pattern, op)
                pow_val = eval(matches[0])
                transformed = torch.pow(batch_vals, pow_val)
                names = [f'({x})**{matches[0]}' for x in batch_cols]
                op_str = "pow(" + str(Fraction(pow_val)) + ")"
            elif op == 'sin':
                transformed = torch.sin(batch_vals)
                names = [f'(sin({x}))' for x in batch_cols]
            elif op == 'sinh':
                transformed = torch.sinh(batch_vals)
                names = [f'(sinh({x}))' for x in batch_cols]
            elif op == 'cos':
                transformed = torch.cos(batch_vals)
                names = [f'(cos({x}))' for x in batch_cols]
            elif op == 'cosh':
                transformed = torch.cosh(batch_vals)
                names = [f'(cosh({x}))' for x in batch_cols]
            elif op == 'tanh':
                transformed = torch.tanh(batch_vals)
                names = [f'(tanh({x}))' for x in batch_cols]
            elif op == '^-1':
                transformed = torch.reciprocal(batch_vals)
                names = [f'(({x})**-1)' for x in batch_cols]
            elif op == 'exp(-1)':
                exp = torch.exp(batch_vals)
                transformed = torch.reciprocal(exp)
                names = [f'(exp(-{x}))' for x in batch_cols]
            elif op == '+1':
                transformed = batch_vals + 1
                names = [f'({x}+1)' for x in batch_cols]
            elif op == '-1':
                transformed = batch_vals - 1
                names = [f'({x}-1)' for x in batch_cols]
            elif op == '/2':
                transformed = batch_vals / 2
                names = [f'({x}/2)' for x in batch_cols]
            else:
                raise ValueError(f"Operator '{op}' not implemented for batching.")

            self.feature_values_11 = torch.cat((self.feature_values_11, transformed), dim=1)
            feature_names_12.extend(names)

            # --- Genealogy tracking for this batch ---
            if op == "pow(" + str(Fraction(pow_val)) + ")" if "pow" in op else op:
                op_key = self.operators_dict[op if op not in ["pow"] else op_str]
            else:
                op_key = self.operators_dict[op]
            if i == 1:
                new_ref = self.reference_tensor[batch_idx, :]
                operators_ref = torch.full((new_ref.shape[0],), op_key)
            else:
                new_ref = self.reference_tensor[batch_idx, :]
                if self.operators_final.dim() == 1:
                    self.operators_final = self.operators_final.unsqueeze(1)
                operators_ref = self.operators_final[self.df.shape[1]:self.df_feature_values.shape[1], :].clone()
                initial_duplicates = self.operators_final[:self.df.shape[1], :].clone()
                operators_ref[:, -1] = op_key
                initial_duplicates[:, -1] = op_key
                operators_ref = torch.cat((initial_duplicates, operators_ref), dim=0)
            feature_values_reference = torch.cat((feature_values_reference, new_ref), dim=0)
            if operators_reference.dim() == 1:
                operators_reference = operators_reference.unsqueeze(1)
            if operators_ref.dim() == 1:
                operators_ref = operators_ref.unsqueeze(1)
            operators_reference = torch.cat((operators_reference, operators_ref), dim=0)

            del transformed, names, operators_ref, new_ref, batch_vals

        self.feature_values_unary = torch.cat((self.feature_values_unary, self.feature_values_11), dim=1)
        self.feature_names_unary.extend(feature_names_12)
        self.reference_tensor = torch.cat((self.reference_tensor, feature_values_reference), dim=0)
        self.reference_tensor = self.clean_tensor(self.reference_tensor)

        if self.operators_final.dim() == 1:
            self.operators_final = self.operators_final.unsqueeze(1)
        if operators_reference.dim() == 1:
            operators_reference = operators_reference.unsqueeze(1)
        if self.operators_final.shape[1] == operators_reference.shape[1]:
            self.operators_final = torch.cat((self.operators_final, operators_reference))
        else:
            additional_columns = torch.full((self.operators_final.size(0), abs(operators_reference.shape[1] - self.operators_final.shape[1])), float('nan'))
            self.operators_final = torch.cat((self.operators_final, additional_columns), dim=1)
            self.operators_final = torch.cat((self.operators_final, operators_reference))
        self.operators_final = self.clean_tensor(self.operators_final)
        assert self.reference_tensor.shape[0] == self.operators_final.shape[0], f"Mismatch after single_variable op: {self.reference_tensor.shape[0]}, {self.operators_final.shape[0]}"
        del self.feature_values_11, feature_names_12

    return self.feature_values_unary, self.feature_names_unary



  '''
  ################################################################################################

  Defining method to perform the combinations of the variables with the initial feature set
  ################################################################################################
  '''
  def combinations(self, operators_set, i, batch_size=1500):
    """
    Batched memory-efficient feature combination expansion for binary operators, preserving
    reference and operator genealogy.
    """
    device = self.device
    self.feature_values_binary = torch.empty(self.df.shape[0], 0).to(device)
    self.feature_names_binary = []
    # These preserve operator and input lineage for new features
    reference_rows = []
    operators_rows = []

    n_feat = self.df_feature_values.shape[1]
    ref_tensor = self.reference_tensor
    op_final = self.operators_final
    columns = self.columns

    for op in operators_set:
        op_index = self.operators_dict[op]
        feature_values11 = torch.empty(self.df.shape[0], 0).to(device)
        feature_names_11 = []
        reference_rows_op = []
        operators_rows_op = []

        for batch in batched_combinations(n_feat, r=2, batch_size=batch_size):
            idx_i = [a for a, b in batch]
            idx_j = [b for a, b in batch]

            x_i = self.df_feature_values[:, idx_i]
            x_j = self.df_feature_values[:, idx_j]

            # Create feature names and genealogy rows for the batch
            name_pairs = [(columns[a], columns[b]) for a, b in batch]
            ref_batch = torch.cat([ref_tensor[idx_i], ref_tensor[idx_j]], dim=1)
            op_batch = torch.cat([
                op_final[idx_i] if op_final.dim() > 1 else op_final[idx_i].unsqueeze(1),
                op_final[idx_j] if op_final.dim() > 1 else op_final[idx_j].unsqueeze(1),
                torch.full((len(batch), 1), op_index, device=device)
            ], dim=1)

            if op == '+':
                vals = x_i + x_j
                feature_names_11 += [f"({a}+{b})" for a, b in name_pairs]
                reference_rows_op.append(ref_batch)
                operators_rows_op.append(op_batch)
            elif op == '-':
                vals = x_i - x_j
                feature_names_11 += [f"({a}-{b})" for a, b in name_pairs]
                reference_rows_op.append(ref_batch)
                operators_rows_op.append(op_batch)
            elif op == '*':
                vals = x_i * x_j
                feature_names_11 += [f"({a}*{b})" for a, b in name_pairs]
                reference_rows_op.append(ref_batch)
                operators_rows_op.append(op_batch)
            elif op == '/':
                # a/b and b/a, with their own names, refs, ops
                vals1 = x_i / (x_j + 1e-12)
                vals2 = x_j / (x_i + 1e-12)
                vals = torch.cat([vals1, vals2], dim=1)
                feature_names_11 += [f"({a}/{b})" for a, b in name_pairs]
                feature_names_11 += [f"({b}/{a})" for a, b in name_pairs]
                # Duplicate genealogy rows for both directions
                ref_dbl = torch.cat([ref_batch, ref_batch], dim=0)
                op_dbl = torch.cat([op_batch, op_batch], dim=0)
                reference_rows_op.append(ref_dbl)
                operators_rows_op.append(op_dbl)
            else:
                raise ValueError(f"Operator '{op}' not implemented.")

            feature_values11 = torch.cat((feature_values11, vals), dim=1)
            del x_i, x_j, vals
            torch.cuda.empty_cache()

        self.feature_values_binary = torch.cat((self.feature_values_binary, feature_values11), dim=1)
        self.feature_names_binary += feature_names_11
        reference_rows.append(torch.cat(reference_rows_op, dim=0) if reference_rows_op else torch.empty(0,))
        operators_rows.append(torch.cat(operators_rows_op, dim=0) if operators_rows_op else torch.empty(0,))

        del feature_values11, feature_names_11, reference_rows_op, operators_rows_op

    # Update reference_tensor and operators_final to match new features
    if reference_rows:
        new_refs = torch.cat(reference_rows, dim=0)
        self.reference_tensor = torch.cat([self.reference_tensor, new_refs], dim=0)
    if operators_rows:
        new_ops = torch.cat(operators_rows, dim=0)
        # Expand dimension if needed to keep shapes consistent
        if self.operators_final.dim() == 1:
            self.operators_final = self.operators_final.unsqueeze(1)
        # Pad columns if needed
        col_pad = new_ops.shape[1] - self.operators_final.shape[1]
        if col_pad > 0:
            pad = torch.full((self.operators_final.size(0), col_pad), float('nan'), device=device)
            self.operators_final = torch.cat([self.operators_final, pad], dim=1)
        self.operators_final = torch.cat([self.operators_final, new_ops], dim=0)

    return self.feature_values_binary, self.feature_names_binary



  '''
  ##########################################################################################################

  Creating the space based on the given set of conditions

  ##########################################################################################################

  '''

  def feature_space(self):


    # Split the operator set into combinations set and unary set
    basic_operators = [op for op in self.operators if op in ['+', '-', '*', '/']]
    
    other_operators = [op for op in self.operators if op not in ['+', '-', '*', '/']]
    
    
    if self.no_of_operators == None:
        
        #if self.disp: print('############################################################# Implementing Automatic Expansion and construction of sparse models..!!! ######################################################################')
        
        from .Regressor import Regressor
        
        i = 1
        
        start_time = time.time()
        
        values, names = self.combinations(basic_operators,i)
    
        # Performs the feature space expansion based on the unary operator set provided
        values1, names1 = self.single_variable(other_operators,i)
    
        features_created = torch.cat((values,values1),dim=1)
        
        del values, values1
        
        names2 = names + names1
        
        del names,names1
        
        self.df_feature_values = torch.cat((self.df_feature_values,features_created),dim=1)
        
        self.columns.extend(names2)
        
        del features_created,names2
        assert self.reference_tensor.shape[0] == self.operators_final.shape[0], (
                f"Shape mismatch after initial feature expansion: reference_tensor={self.reference_tensor.shape[0]}, "
                f"operators_final={self.operators_final.shape[0]}"
            )
        if self.disp:
        
            print('****************************** Initial Feature Expansion Completed with feature space size: ',self.df_feature_values.shape[1],'*********************************************** \n')
            
            print('**************************************** Time taken to create the space is:::', time.time()-start_time, ' Seconds ********************************************\n')
            
        
        #self.dimension=None
        
        #self.sis_features=None
        
        # Replace NaNs with a value that won't interfere with counting
        tensor_replaced = torch.nan_to_num(self.reference_tensor, nan=float('inf'))

        # Mask to identify non-NaN values
        mask = self.reference_tensor == self.reference_tensor  # True where tensor is not NaN

        # Calculate the total number of numerical values per row
        num_numericals = mask.sum(dim=1)
        
        sorted_tensor, _ = torch.sort(self.reference_tensor, dim=1)
        
        diff = torch.diff(sorted_tensor, dim=1)
        
        unique_mask = torch.cat([torch.ones(self.reference_tensor.shape[0], 1).to(self.reference_tensor.device), (diff != 0).float()], dim=1)
        
        unique_counts = (unique_mask * mask).sum(dim=1)
        
        tensor_replaced1 = torch.nan_to_num(self.operators_final, nan=float('inf'))

        # Mask to identify non-NaN values
        mask = self.operators_final == self.operators_final  # True where tensor is not NaN

        # Calculate the total number of numerical values per row
        num_numericals1 = mask.sum(dim=1)
        
        sorted_tensor, _ = torch.sort(self.operators_final, dim=1)
        
        diff = torch.diff(sorted_tensor, dim=1)
        
        unique_mask = torch.cat([torch.ones(self.operators_final.shape[0], 1).to(self.operators_final.device), (diff != 0).float()], dim=1)

        unique_counts1 = (unique_mask * mask).sum(dim=1)
        print("NUM NUMERICALS: ", num_numericals)
        print("NUM NUMERICALS1: ", num_numericals1)
        print("UNIQUE COUNTS: ", unique_counts.shape)
        print("UNIQUE COUNTS1: ", unique_counts1.shape)
        complexity = (num_numericals+num_numericals1)*torch.log2(unique_counts+unique_counts1)
        
        complexity[:self.df.shape[1]] = 1
        
        print("Using Regressor from:", Regressor.__module__)
        print("FEATURE_SPACE:", self.df_feature_values,self.Target_column,self.columns)
        rmse1, equation1,r21,r,c,n,intercepts,coeffs,r2_value =  Regressor(self.df_feature_values,self.Target_column,self.columns,complexity,self.dimension,self.sis_features,self.device,metrics = self.metrics, A=self.A, B=self.B).regressor_fit()
        
        additional_columns = torch.full((1, abs(coeffs.shape[1])), float('nan'))
        
        additional_columns[:,0] = 1
        
        coeffs = torch.cat((additional_columns,coeffs))
        
        intercepts= torch.cat((torch.tensor([0]),intercepts))
        
        s= pareto(r,c,final_pareto='no').pareto_front()
        
        complexity_final = c[s]
        
        
        rmse_final = r[s]
        
        
        #names_final = np.array(n)[s].tolist()
        names_final = [n[i] for i in s]
        
        r = rmse_final
        
        c= complexity_final
        
        n = names_final
        
        coeffs = coeffs[s]
        
        self.updated_pareto_rmse = torch.cat((self.updated_pareto_rmse,r))
        
        self.updated_pareto_r2 = torch.cat((self.updated_pareto_r2,r2_value[s]))
        
        self.updated_pareto_complexity = torch.cat((self.updated_pareto_complexity,c))
        
        self.updated_pareto_names.extend(n)
        
        if coeffs.dim()==1: coeffs = coeffs.unsqueeze(1)
        if self.update_pareto_coeff.dim()==1: self.update_pareto_coeff = self.update_pareto_coeff.unsqueeze(1)
        if coeffs.shape[1] == self.update_pareto_coeff.shape[1]:
            
            self.update_pareto_coeff = torch.cat((self.update_pareto_coeff,coeffs))
            
        else:
            if self.update_pareto_coeff.shape[1] < coeffs.shape[1]:
                
                additional_columns = torch.full((self.update_pareto_coeff.size(0), abs(coeffs.shape[1]-self.update_pareto_coeff.shape[1])), float('nan'))
                
                self.update_pareto_coeff = torch.cat((self.update_pareto_coeff,additional_columns),dim=1)
                
                self.update_pareto_coeff = torch.cat((self.update_pareto_coeff, coeffs))
            else:
                additional_columns = torch.full((coeffs.size(0), abs(coeffs.shape[1]-self.update_pareto_coeff.shape[1])), float('nan'))
                
                coeffs = torch.cat((coeffs,additional_columns),dim=1)
                
                self.update_pareto_coeff = torch.cat((self.update_pareto_coeff, coeffs))
        
        self.update_pareto_intercepts=torch.cat((self.update_pareto_intercepts,intercepts[s]))
        
        
        
        
        
        if rmse1 <= self.rmse_metric and r21 >= self.r2_metric: 
        
            
            if self.pareto: final_pareto = 'yes'
            else: final_pareto = 'no'
            
            s= pareto(self.updated_pareto_rmse,self.updated_pareto_complexity,final_pareto=final_pareto).pareto_front()
            
            complexity_final = self.updated_pareto_complexity[s]
            
            
            rmse_final = self.updated_pareto_rmse[s]
            
            
            #names_final = np.array(self.updated_pareto_names)[s].tolist()
            names_final = [self.updated_pareto_names[i] for i in s]
            
            intercepts = self.update_pareto_intercepts[s]
            
            coeffs = self.update_pareto_coeff[s]
            
            r2_final = self.updated_pareto_r2[s]
            
            
            data_final = {'Loss':rmse_final,'Complexity':complexity_final,'Equations':names_final,
                          'Intercepts':intercepts.tolist(),'Coefficients':coeffs.tolist(),'Score':r2_final}
            
            
            
            #data_final = {'Loss':rmse_final,'Complexity':complexity_final,'Equations':names_final}
            
            df_final = pd.DataFrame(data_final)
            
            df_unique = df_final.drop_duplicates(subset='Complexity')
            
            df_sorted = df_unique.sort_values(by='Complexity', ascending=True)
            
            df_sorted.reset_index(drop=True,inplace=True)
            
            #print('Equation:',equation1)
            
            return rmse1,equation1,r21,df_sorted 
        
        i = 2
        
        while True:
            
            values, names = self.combinations(basic_operators,i)

        
            # Performs the feature space expansion based on the unary operator set provided
            values1, names1 = self.single_variable(other_operators,i)

        
            features_created = torch.cat((values,values1),dim=1)
            
            del values, values1
            
            names2 = names + names1
            
            del names,names1
            
            self.df_feature_values = torch.cat((self.df_feature_values,features_created),dim=1)
            
            self.columns.extend(names2)
            
            del features_created,names2
            
            if self.disp:
            
                print(f'************************************ {i} Feature Expansion Completed with feature space size:::',self.df_feature_values.shape[1],' *********************************************************** \n')
                
                print('****************************************** Time taken to create the space is:::', time.time()-start_time, ' Seconds *************************************************** \n')
                
            if self.df_feature_values.shape[1] <10000:
            
                
                unique_columns, indices = torch.unique(self.df_feature_values, sorted=False,dim=1, return_inverse=True)
                
                # Get the indices of the unique columns
                unique_indices = indices.unique()
      
                # Remove duplicate columns
                self.df_feature_values = self.df_feature_values[:, unique_indices]
                
                
                # Remove the corresponding elements from the list of feature names..
                self.columns = [self.columns[i] for i in unique_indices.tolist()]
                
                
                self.reference_tensor = self.reference_tensor[unique_indices,:]
                
                if self.operators_final.dim() ==1 : self.operators_final = self.operators_final[unique_indices] 
                
                else: self.operators_final = self.operators_final[unique_indices,:] 
                assert self.reference_tensor.shape[0] == self.operators_final.shape[0], (
                f"Shape mismatch after initial feature expansion: reference_tensor={self.reference_tensor.shape[0]}, "
                f"operators_final={self.operators_final.shape[0]}"
                )
            tensor_replaced = torch.nan_to_num(self.reference_tensor, nan=float('inf'))

            # Mask to identify non-NaN values
            mask = self.reference_tensor == self.reference_tensor  # True where tensor is not NaN

            # Calculate the total number of numerical values per row
            num_numericals = mask.sum(dim=1)
            
            sorted_tensor, _ = torch.sort(self.reference_tensor, dim=1)
            
            diff = torch.diff(sorted_tensor, dim=1)
            
            unique_mask = torch.cat([torch.ones(self.reference_tensor.shape[0], 1).to(self.reference_tensor.device), (diff != 0).float()], dim=1)
            
            unique_counts = (unique_mask * mask).sum(dim=1)
            
            tensor_replaced1 = torch.nan_to_num(self.operators_final, nan=float('inf'))

            # Mask to identify non-NaN values
            mask = self.operators_final == self.operators_final  # True where tensor is not NaN

            # Calculate the total number of numerical values per row
            num_numericals1 = mask.sum(dim=1)
            
            sorted_tensor, _ = torch.sort(self.operators_final, dim=1)
            
            diff = torch.diff(sorted_tensor, dim=1)
            
            unique_mask = torch.cat([torch.ones(self.operators_final.shape[0], 1).to(self.operators_final.device), (diff != 0).float()], dim=1)

            unique_counts1 = (unique_mask * mask).sum(dim=1)
            
            # Ensure tensors are the same size before operations
            min_size = min(num_numericals.size(0), num_numericals1.size(0), unique_counts.size(0), unique_counts1.size(0))
            
            num_numericals = num_numericals[:min_size]
            num_numericals1 = num_numericals1[:min_size]
            unique_counts = unique_counts[:min_size]
            unique_counts1 = unique_counts1[:min_size]
            
            complexity = (num_numericals+num_numericals1)*torch.log2(unique_counts+unique_counts1)
            
            complexity[:self.df.shape[1]] = 1
            
            
            rmse, equation,r2,r,c,n,intercepts,coeffs,r2_value =  Regressor(self.df_feature_values,self.Target_column,self.columns,complexity,self.dimension,self.sis_features,self.device,metrics = self.metrics, A=self.A, B = self.B).regressor_fit()
            
            additional_columns = torch.full((1, abs(coeffs.shape[1])), float('nan'))
            
            additional_columns[:,0] = 1
            
            coeffs = torch.cat((additional_columns,coeffs))
            
            intercepts= torch.cat((torch.tensor([0]),intercepts))
            
            s= pareto(r,c).pareto_front()
            
            complexity_final = c[s]
            
            rmse_final = r[s]
            
            
            #names_final = np.array(n)[s].tolist()
            names_final = [n[i] for i in s]
            
            r = rmse_final
            
            c= complexity_final
            
            n = names_final
            
            coeffs = coeffs[s]
            
            
            
            self.updated_pareto_rmse = torch.cat((self.updated_pareto_rmse,r))
            
            self.updated_pareto_r2 = torch.cat((self.updated_pareto_r2,r2_value[s]))
            
            self.updated_pareto_complexity = torch.cat((self.updated_pareto_complexity,c))
            
            self.updated_pareto_names.extend(n)
            
           
            if coeffs.shape[1] == self.update_pareto_coeff.shape[1]:
                
                self.update_pareto_coeff = torch.cat((self.update_pareto_coeff,coeffs))
                
            else:
                if self.update_pareto_coeff.shape[1] < coeffs.shape[1]:
                    
                    additional_columns = torch.full((self.update_pareto_coeff.size(0), abs(coeffs.shape[1]-self.update_pareto_coeff.shape[1])), float('nan'))
                    
                    self.update_pareto_coeff = torch.cat((self.update_pareto_coeff,additional_columns),dim=1)
                    
                    self.update_pareto_coeff = torch.cat((self.update_pareto_coeff, coeffs))
                else:
                    additional_columns = torch.full((coeffs.size(0), abs(coeffs.shape[1]-self.update_pareto_coeff.shape[1])), float('nan'))
                    
                    coeffs = torch.cat((coeffs,additional_columns),dim=1)
                    
                    self.update_pareto_coeff = torch.cat((self.update_pareto_coeff, coeffs))
                
            
            self.update_pareto_intercepts=torch.cat((self.update_pareto_intercepts,intercepts[s]))
            
           
            if rmse <= self.rmse_metric and r2 >= self.r2_metric:
                
                
                break
            if i >=2 and self.df_feature_values.shape[1]>2000:
                
                print('Expanded feature space is::',self.df_feature_values.shape[1])
                
                
                print('!!Warning:: Further feature expansions result in memory consumption, Please provide the input to consider feature expansion or to exit the run with the sparse models created!!!')
                
                #response = input("Do you wish to continue (yes/no)? ").strip().lower()
                response = 'no'
                if response == 'no' or response == 'n': 
                    
                    print("Exiting based on user input.")
                    
                    break
            i = i+1


        
        if self.pareto: final_pareto = 'yes'
        else: final_pareto = 'no'
        s= pareto(self.updated_pareto_rmse,self.updated_pareto_complexity,final_pareto=final_pareto).pareto_front()
        
        complexity_final = self.updated_pareto_complexity[s]
        
        rmse_final = self.updated_pareto_rmse[s]
        
        
        #names_final = np.array(self.updated_pareto_names)[s].tolist()
        names_final = [self.updated_pareto_names[i] for i in s]

        
        intercepts = self.update_pareto_intercepts[s]
        
        coeffs = self.update_pareto_coeff[s]
        
        r2_final = self.updated_pareto_r2[s]
        
        data_final = {'Loss':rmse_final,'Complexity':complexity_final,'Equations':names_final,
                      'Intercepts':intercepts.tolist(),'Coefficients':coeffs.tolist(),'Score':r2_final}
        
        
        
        #data_final = {'Loss':rmse_final,'Complexity':complexity_final,'Equations':names_final}
        
        df_final = pd.DataFrame(data_final)
        
        df_unique = df_final.drop_duplicates(subset='Complexity')
        
        df_sorted = df_unique.sort_values(by='Complexity', ascending=True)
        
        df_sorted.reset_index(drop=True,inplace=True)
        
        #print('Equation:',equation)
        
        return rmse,equation,r2,df_sorted

    else:
        
        for i in range(1,self.no_of_operators):
            
            start_time = time.time()
            
            if self.disp: print(f'*********************************   Starting {i} level of feature expansion******************************************** \n')
    
            #Performs the feature space expansion based on the binary operator set provided
            values, names = self.combinations(basic_operators,i)
        
            # Performs the feature space expansion based on the unary operator set provided
            values1, names1 = self.single_variable(other_operators,i)
            
        
            features_created = torch.cat((values,values1),dim=1)
            
            del values, values1
            
            names2 = names + names1
            
            del names,names1
            
            self.df_feature_values = torch.cat((self.df_feature_values,features_created),dim=1)
            
            self.columns.extend(names2)
            
            del features_created,names2
            
            unique_columns, indices = torch.unique(self.df_feature_values, sorted=False,dim=1, return_inverse=True)
            
            # Get the indices of the unique columns
            unique_indices = indices.unique()
      
            # Remove duplicate columns
            self.df_feature_values = self.df_feature_values[:, unique_indices]
            
            
            # Remove the corresponding elements from the list of feature names..
            self.columns = [self.columns[i] for i in unique_indices.tolist()]
            
            self.reference_tensor = self.reference_tensor[unique_indices,:]
            
            if self.operators_final.dim() ==1 : self.operators_final = self.operators_final[unique_indices] 
            else: self.operators_final = self.operators_final[unique_indices,:] 
            assert self.reference_tensor.shape[0] == self.operators_final.shape[0], (
                f"Shape mismatch after initial feature expansion: reference_tensor={self.reference_tensor.shape[0]}, "
                f"operators_final={self.operators_final.shape[0]}"
            )
            if self.disp:
                print(f'**************************** {i} Feature Expansion Completed with feature space size:::',self.df_feature_values.shape[1],'************************************************* \n')
                
                if self.feature_names: 
                    
                    print('Feature Names:', self.columns)
                    
                    print('\n \n \n')
                print('****************************************** Time taken to create the space is:::', time.time()-start_time, ' Seconds********************************************* \n')
            # Replace NaNs with a value that won't interfere with counting
            tensor_replaced = torch.nan_to_num(self.reference_tensor, nan=float('inf'))

            # Mask to identify non-NaN values
            mask = self.reference_tensor == self.reference_tensor  # True where tensor is not NaN

            # Calculate the total number of numerical values per row
            num_numericals = mask.sum(dim=1)
            
            sorted_tensor, _ = torch.sort(self.reference_tensor, dim=1)
            
            diff = torch.diff(sorted_tensor, dim=1)
            
            unique_mask = torch.cat([torch.ones(self.reference_tensor.shape[0], 1).to(self.reference_tensor.device), (diff != 0).float()], dim=1)
            
            unique_counts = (unique_mask * mask).sum(dim=1)
            
            tensor_replaced1 = torch.nan_to_num(self.operators_final, nan=float('inf'))

            # Mask to identify non-NaN values
            mask = self.operators_final == self.operators_final  # True where tensor is not NaN

            # Calculate the total number of numerical values per row
            num_numericals1 = mask.sum(dim=1)
            
            sorted_tensor, _ = torch.sort(self.operators_final, dim=1)
            
            diff = torch.diff(sorted_tensor, dim=1)
            
            unique_mask = torch.cat([torch.ones(self.operators_final.shape[0], 1).to(self.operators_final.device), (diff != 0).float()], dim=1)

            unique_counts1 = (unique_mask * mask).sum(dim=1)
            
            # Ensure tensors are the same size before operations
            min_size = min(num_numericals.size(0), num_numericals1.size(0), unique_counts.size(0), unique_counts1.size(0))
            
            num_numericals = num_numericals[:min_size]
            num_numericals1 = num_numericals1[:min_size]
            unique_counts = unique_counts[:min_size]
            unique_counts1 = unique_counts1[:min_size]
            print("NUM NUMERICALS: ", num_numericals)
            print("NUM NUMERICALS1: ", num_numericals1)
            complexity = (num_numericals+num_numericals1)*torch.log2(unique_counts+unique_counts1)
            
            complexity[:self.df.shape[1]] = 1
                
            
        return self.df_feature_values, self.Target_column,self.columns,complexity
    
