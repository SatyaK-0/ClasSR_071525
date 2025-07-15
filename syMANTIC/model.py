#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:22:50 2023

@author: muthyala.7
"""

from syMANTIC import FeatureSpaceConstruction as fcc
from syMANTIC import DimensionalFeatureSpaceConstruction as dfcc
import sys
import time
import pdb
import numpy as np 
import pandas as pd 
import os
from sympy import symbols
import matplotlib.pyplot as plt
import matplotlib

class ProgressSaver:
    def __init__(self, save_path="model_progress", save_frequency=5):
        self.save_path = save_path
        self.save_frequency = save_frequency  # Save every N iterations
        self.iteration = 0
        os.makedirs(save_path, exist_ok=True)
    
    def save_progress(self, model, current_state):
        """Save current progress with timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_path}/progress_iter{self.iteration}_{timestamp}.pkl"
        
        progress_data = {
            'iteration': self.iteration,
            'timestamp': timestamp,
            'current_state': current_state,
            'model_state': {
                'final_df': model.final_df,
                'operators': model.operators,
                'device': model.device,
                'dimension': model.dimension,
                'metrics': model.metrics,
                'pareto': model.pareto,
                'A': model.A,
                'B': model.B,
                'updated_pareto_rmse': getattr(model, 'updated_pareto_rmse', None),
                'updated_pareto_r2': getattr(model, 'updated_pareto_r2', None),
                'updated_pareto_complexity': getattr(model, 'updated_pareto_complexity', None),
                'updated_pareto_names': getattr(model, 'updated_pareto_names', []),
                'update_pareto_coeff': getattr(model, 'update_pareto_coeff', None),
                'update_pareto_intercepts': getattr(model, 'update_pareto_intercepts', None),
                'df': model.df if hasattr(model, 'df') else None,
                'current_iteration': model.current_iteration if hasattr(model, 'current_iteration') else 0
            }
        }
        
        try:
            import joblib
            joblib.dump(progress_data, filename)
        except:
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump(progress_data, f)
        
        self.iteration += 1
        return filename

    @staticmethod
    def load_latest_progress(save_path="model_progress"):
        """Load the most recent progress file"""
        import glob
        files = glob.glob(f"{save_path}/progress_iter*.pkl")
        if not files:
            return None
        
        latest_file = max(files, key=os.path.getctime)
        try:
            import joblib
            return joblib.load(latest_file)
        except:
            import pickle
            with open(latest_file, 'rb') as f:
                return pickle.load(f)

class SymanticModel:
    def __init__(self, df, operators=None, multi_task=None, n_expansion=None, 
                 n_term=None, sis_features=20, device='cpu', relational_units=None,
                 initial_screening=None, dimensionality=None, output_dim=None,
                 metrics=[0.06,0.995], disp=False, pareto=False, A=None, B=None,
                 save_progress=False, progress_path="model_progress", save_frequency=5):
        
        self.A = A
        self.B = B 
        self.operators = operators
        self.df = df
        self.no_of_operators = n_expansion
        self.device = device
        
        if n_term == None: 
            self.dimension = 3
        else: 
            self.dimension = n_term
        
        if sis_features == None: 
            self.sis_features = 10
        else: 
            self.sis_features = sis_features
        
        self.relational_units = relational_units
        self.initial_screening = initial_screening
        self.dimensionality = dimensionality
        self.output_dim = output_dim
        self.metrics = metrics
        self.multi_task = multi_task
        self.disp = disp
        self.pareto = pareto
        self.final_df = None
        
        # Progress saving attributes
        self.save_progress = save_progress
        self.progress_path = progress_path
        self.save_frequency = save_frequency
        self.current_iteration = 0
        self.progress_saver = ProgressSaver(progress_path, save_frequency) if save_progress else None

        if multi_task != None:
            self.multi_task_target = multi_task[0]
            self.multi_task_features = multi_task[1]

    def combine_equation(self, row):
        terms = row['Equations']
        coeffs = row['Coefficients']
        intercept = row['Intercepts']
        
        if isinstance(terms, str):
            terms = [terms]
        
        equation_parts = []
        for term, coeff in zip(terms, coeffs):
            if pd.isna(coeff):
                continue
            if coeff == 1:
                equation_parts.append(term)
            elif coeff == -1:
                equation_parts.append(f"-{term}")
            else:
                equation_parts.append(f"{coeff:.4f}*{term}")
        
        equation = " + ".join(equation_parts)
        
        if intercept != 0:
            if intercept > 0:
                equation = f"{equation} + {intercept:.4f}"
            else:
                equation = f"{equation} - {abs(intercept):.4f}"
        
        return equation

    def _save_current_progress(self, current_state):
        """Internal method to save progress"""
        if self.save_progress and self.progress_saver:
            saved_file = self.progress_saver.save_progress(self, current_state)
            if self.disp:
                print(f"Progress saved to {saved_file}")
            return saved_file
        return None

    def _load_progress(self, progress_data):
        """Load state from progress data"""
        if not progress_data:
            return False
        
        model_state = progress_data['model_state']
        self.current_iteration = progress_data['iteration']
        
        # Restore main attributes
        for key, value in model_state.items():
            setattr(self, key, value)
        
        if self.disp:
            print(f"Loaded progress from iteration {self.current_iteration}")
            print(f"Current best RMSE: {model_state.get('updated_pareto_rmse', [float('inf')]).min()}")
        
        return True

    def fit(self, A=None, B=None):
        if A is None:
            A = self.A
        if B is None:
            B = self.B
        
        # Try to load existing progress
        if self.save_progress:
            progress = ProgressSaver.load_latest_progress(self.progress_path)
            if progress and self._load_progress(progress):
                if self.disp:
                    print("Resuming from saved progress...")
        
        if self.dimensionality == None:
            if self.operators == None: 
                sys.exit('Please provide the operators set for the non dimensional Regression!!')
            
            if self.multi_task != None:
                if self.disp: 
                    print('************************************* Performing MultiTask Symbolic regression!!..**************************************************************** \n')
                
                equations = []
                for i in range(len(self.multi_task_target)):
                    if self.disp: 
                        print(f'***************************************** Performing symbolic regression of {i+1} Target variables******************************************** \n')
                    
                    list1 = []
                    list1.extend([self.multi_task_target[i]] + self.multi_task_features[i])
                    df1 = self.df.iloc[:, list1]
                    
                    if self.no_of_operators == None:
                        st = time.time()
                        rmse, equation, r2, _ = fcc.feature_space_construction(
                            self.operators, df1, self.no_of_operators, self.device,
                            self.initial_screening, self.metrics, dimension=self.dimension,
                            sis_features=self.sis_features, disp=self.disp, pareto=self.pareto,
                            A=self.A, B=self.B
                        ).feature_space()
                        
                        if self.disp: 
                            print(f'************************************************ Autodepth regression completed in:: {time.time()-st} seconds ************************************************ \n')
                        
                        equations.append(equation)
                        
                        if i+1 == len(self.multi_task_target):
                            if self.disp: 
                                print('Equations found::', equations)
                            return rmse, equation, r2, equations
                        else:
                            continue
                    else:
                        x, y, names, complexity = fcc.feature_space_construction(
                            self.operators, df1, self.no_of_operators, self.device,
                            self.initial_screening, disp=self.disp, pareto=self.pareto,
                            A=self.A, B=self.B
                        ).feature_space()
                        
                        from .Regressor import Regressor
                        rmse, equation, r2, _, _, _, _, _ = Regressor(
                            x, y, names, complexity, self.dimension, self.sis_features,
                            self.device, A=self.A, B=self.B
                        ).regressor_fit()
                        
                        equations.append(equation)
                        
                        if i+1 == len(self.multi_task_target):
                            if self.disp: 
                                print('Equations found::', equations)
                            return rmse, equation, r2, equations
                        else:
                            continue
            
            elif self.no_of_operators == None:
                st = time.time()
                print("TESTING FOR UPDATES IN CODE")
                rmse, equation, r2, final = fcc.feature_space_construction(
                    self.operators, self.df, self.no_of_operators, self.device,
                    self.initial_screening, self.metrics, dimension=self.dimension,
                    sis_features=self.sis_features, disp=self.disp, pareto=self.pareto,
                    A=self.A, B=self.B
                ).feature_space()
                
                if self.disp: 
                    print(f'************************************************ Autodepth regression completed in:: {time.time()-st} seconds ************************************************ \n')
                
                self.final_df = final
                final['Normalized_Loss'] = (final['Loss'] - final['Loss'].min()) / (final['Loss'].max() - final['Loss'].min())
                final['Normalized_Complexity'] = (final['Complexity'] - final['Complexity'].min()) / (final['Complexity'].max() - final['Complexity'].min())
                final['Distance_to_Utopia'] = np.sqrt(final['Normalized_Loss']**2 + final['Normalized_Complexity']**2)

                utopia_row = final['Distance_to_Utopia'].idxmin()

                final_edited = pd.DataFrame()
                final_edited['Loss'] = final['Loss']
                final_edited['Complexity'] = final['Complexity']
                final_edited['R2'] = final['Score']
                final_edited['Equation'] = final.apply(self.combine_equation, axis=1)
                
                print('************************************************  Please take a look at the entire pareto set generated!!! *******************************************************')
                res = {
                    'utopia': {
                        'expression': final_edited.Equation[utopia_row],
                        'Modified Binary Cross-Entropy': final_edited.Loss[utopia_row],
                        'Accuracy': final_edited.R2[utopia_row],
                        'complexity': final_edited.Complexity[utopia_row],
                    }
                }
                
                # Save final progress
                if self.save_progress:
                    self._save_current_progress({
                        'status': 'completed',
                        'best_equation': res['utopia']['expression'],
                        'best_rmse': res['utopia']['Modified Binary Cross-Entropy'],
                        'best_r2': res['utopia']['Accuracy']
                    })
                
                return res, final_edited
            
            else:
                x, y, names, complexity = fcc.feature_space_construction(
                    self.operators, self.df, self.no_of_operators, self.device,
                    self.initial_screening, disp=self.disp, A=self.A, B=self.B
                ).feature_space()
                
                from .Regressor import Regressor
                rmse, equation, r2, _, _, _, _, _ = Regressor(
                    x, y, names, complexity, self.dimension, self.sis_features,
                    self.device, A=self.A, B=self.B
                ).regressor_fit()
                
                return rmse, equation, r2
        
        else:
            # Dimensional feature space construction case
            if self.multi_task != None:
                if self.disp: 
                    print('************************************************ Performing MultiTask Symbolic regression!!..************************************************ \n')
                
                equations = []
                for i in range(len(self.multi_task_target)):
                    if self.disp: 
                        print(f'************************************************ Performing symbolic regression of {i+1} Target variables....************************************************ \n')
                    
                    list1 = []
                    list1.extend([self.multi_task_target[i]] + self.multi_task_features[i])
                    df1 = self.df.iloc[:, list1]
                    
                    if self.no_of_operators == None:
                        st = time.time()
                        rmse, equation, r2, final = dfcc.feature_space_construction(
                            df1, self.operators, self.relational_units,
                            self.initial_screening, self.no_of_operators, self.device,
                            self.dimensionality, self.metrics, self.output_dim,
                            disp=self.disp, pareto=self.pareto
                        ).feature_expansion()
                        
                        if self.disp: 
                            print(f'************************************************ Autodepth regression completed in:: {time.time()-st} seconds ************************************************ \n')
                        
                        equations.append(equation)
                        
                        if i+1 == len(self.multi_task_target):
                            print('Equations found::', equations)
                            return rmse, equation, r2, equations
                        else:
                            continue
                    else:
                        x, y, names, dim, complexity = dfcc.feature_space_construction(
                            df1, self.operators, self.relational_units,
                            self.initial_screening, self.no_of_operators, self.device,
                            self.dimensionality, disp=self.disp, pareto=self.pareto
                        ).feature_expansion()
                        
                        from .DimensionalRegressor import Regressor
                        rmse, equation, r2, _, _, _, _, _, _ = Regressor(
                            x, y, names, dim, complexity, self.dimension,
                            self.sis_features, self.device, self.output_dim,
                            disp=self.disp, pareto=self.pareto
                        ).regressor_fit()
                        
                        equations.append(equation)
                        
                        if i+1 == len(self.multi_task_target):
                            print('Equations found::', equations)
                            return rmse, equation, r2, equations
                        else:
                            continue
            
            if self.no_of_operators == None:
                st = time.time()
                rmse, equation, r2, final = dfcc.feature_space_construction(
                    self.df, self.operators, self.relational_units,
                    self.initial_screening, self.no_of_operators, self.device,
                    self.dimensionality, self.metrics, self.output_dim,
                    disp=self.disp, pareto=self.pareto
                ).feature_expansion()
                
                if self.disp: 
                    print(f'************************************************ Autodepth regression completed in:: {time.time()-st} seconds ************************************************ \n')
                
                self.final_df = final
                final['Normalized_Loss'] = (final['Loss'] - final['Loss'].min()) / (final['Loss'].max() - final['Loss'].min())
                final['Normalized_Complexity'] = (final['Complexity'] - final['Complexity'].min()) / (final['Complexity'].max() - final['Complexity'].min())
                final['Distance_to_Utopia'] = np.sqrt(final['Normalized_Loss']**2 + final['Normalized_Complexity']**2)

                utopia_row = final['Distance_to_Utopia'].idxmin()

                final_edited = pd.DataFrame()
                final_edited['Loss'] = final['Loss']
                final_edited['Complexity'] = final['Complexity']
                final_edited['R2'] = final['Score']
                final_edited['Equation'] = final.apply(self.combine_equation, axis=1)
                
                print('************************************************  Please take a look at the entire pareto set generated!!! *******************************************************')
                res = {
                    'utopia': {
                        'expression': final_edited.Equation[utopia_row],
                        'rmse': final_edited.Loss[utopia_row],
                        'r2': final_edited.R2[utopia_row],
                        'complexity': final_edited.Complexity[utopia_row],
                    }
                }
                
                # Save final progress
                if self.save_progress:
                    self._save_current_progress({
                        'status': 'completed',
                        'best_equation': res['utopia']['expression'],
                        'best_rmse': res['utopia']['rmse'],
                        'best_r2': res['utopia']['r2']
                    })
                
                return res, final_edited
            
            else:
                x, y, names, dim, complexity = dfcc.feature_space_construction(
                    self.df, self.operators, self.relational_units,
                    self.initial_screening, self.no_of_operators, self.device,
                    self.dimensionality, disp=self.disp, pareto=self.pareto
                ).feature_expansion()
                
                from .DimensionalRegressor import Regressor
                rmse, equation, r2, _, _, _, _, _, _ = Regressor(
                    x, y, names, dim, complexity, self.dimension,
                    self.sis_features, self.device, self.output_dim,
                    disp=self.disp, pareto=self.pareto
                ).regressor_fit()
                
                return rmse, equation, r2

    def plot_pareto_front(self):
        import matplotlib.pyplot as plt
        import matplotlib
        
        if self.final_df is None:
            print("No Pareto frontier data available. Please run fit() first.")
            return
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.final_df['Complexity'], self.final_df['Loss'], c='red', label='Pareto front')
        plt.step(self.final_df['Complexity'], self.final_df['Loss'], 'r-', where='post', label='Pareto Line')
        plt.scatter(self.final_df['Complexity'].min(), self.final_df['Loss'].min(), c='green', label='Utopia', marker='*', s=100)
        plt.xlabel(r'Complexity = $k \log n$ (bits)', weight='bold')
        plt.ylabel('Accuracy (RMSE)', weight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.title('Pareto Frontier')
        plt.show()

    def save_model(self, filepath):
        """
        Save the trained symbolic regression model to a file.
        """
        import pickle
        import joblib
        
        model_data = {
            'final_df': self.final_df,
            'operators': self.operators,
            'device': self.device,
            'dimension': self.dimension,
            'sis_features': self.sis_features,
            'metrics': self.metrics,
            'pareto': self.pareto,
            'A': self.A,
            'B': self.B,
            'df': self.df
        }
        
        try:
            joblib.dump(model_data, filepath)
        except:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filepath, df=None):
        """
        Load a saved symbolic regression model.
        """
        import pickle
        import joblib
        
        try:
            model_data = joblib.load(filepath)
        except:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        
        if df is None:
            if 'df' in model_data:
                df = model_data['df']
            else:
                raise ValueError("No DataFrame provided and no DataFrame found in saved model")
        
        model = cls(
            df=df,
            operators=model_data['operators'],
            device=model_data['device'],
            n_expansion=None,
            n_term=model_data.get('dimension', 3),
            sis_features=model_data.get('sis_features', 20),
            metrics=model_data.get('metrics', [0.06, 0.995]),
            pareto=model_data.get('pareto', False),
            A=model_data.get('A', None),
            B=model_data.get('B', None)
        )
        
        model.final_df = model_data['final_df']
        return model

    def get_best_equation(self):
        """
        Get the best equation from the trained model.
        """
        if self.final_df is None:
            raise ValueError("Model has not been trained yet")
            
        best_idx = self.final_df['Loss'].idxmin()
        
        return {
            'equation': self.final_df.loc[best_idx, 'Equation'],
            'loss': self.final_df.loc[best_idx, 'Loss'],
            'complexity': self.final_df.loc[best_idx, 'Complexity'],
            'r2': self.final_df.loc[best_idx, 'R2']
        }