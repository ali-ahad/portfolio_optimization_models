import numpy as np
import pandas as pd
from scipy.optimize import minimize

class MarkowitzOptimization:
  def __init__(self, returns_list: list, cov_matrix: list, risk_free_rate: float):
      self.returns_list = returns_list
      self.cov_matrix = cov_matrix
      self.risk_free_rate = risk_free_rate
  
  def __max_sharpe_ratio_objective_function(self, weights):
    portfolio_return = np.dot(weights, self.returns_list)
    excess_return = portfolio_return - self.risk_free_rate
    
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    return -excess_return / portfolio_risk
  
  def __minimum_variance_objective_function(self, weights):
    return np.dot(weights.T, np.dot(self.cov_matrix, weights))
  
  def find_tangency_portfolio_with_short_selling(self):
    cov_inversed = np.linalg.inv(self.cov_matrix)
    
    # get a vector of risk free rate equal to the length of expected returns
    risk_free_rate_vector = np.full(len(self.returns_list), self.risk_free_rate)
    
    # get the z weights
    z = np.matmul(cov_inversed, self.returns_list - risk_free_rate_vector)
    
    # get the normalized weights
    weights = z / np.sum(z)
    
    return weights
  
  def find_tangency_portfolio_no_short_selling(self):
    # constraints - sum of weights must be 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # bounds - weights must be between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for i in range(len(self.returns_list)))
    
    # initial guess - equal weights
    w0 = np.full(len(self.returns_list), 1 / len(self.returns_list))
    
    # minimize the negative sharpe ratio
    result = minimize(self.__max_sharpe_ratio_objective_function, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
      raise Exception(result.message)
    return result.x
  
  def find_tangency_portfolio_no_short_selling_and_max_weight(self, max_wieght_constraint: float):
    # constraints - sum of weights must be 1
    # constraint - each stock weight must be less than max_wieght_constraint
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'ineq', 'fun': lambda x: max_wieght_constraint - x})
    
    # bounds - weights must be between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for i in range(len(self.returns_list)))
    
    # initial guess - equal weights
    w0 = np.full(len(self.returns_list), 1 / len(self.returns_list))
    
    # minimize the negative sharpe ratio
    result = minimize(self.__max_sharpe_ratio_objective_function, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
      raise Exception(result.message)
    return result.x
  
  def minimum_variance_portfolio(self, target_return: float):
    # constraints - sum of weights must be 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, 
                   {'type': 'eq', 'fun': lambda x: np.dot(x, self.returns_list) - target_return})
    
    # bounds - short selling is allowed
    bounds = tuple((-1, 1) for i in range(len(self.returns_list)))
    
    # initial guess - equal weights
    w0 = np.full(len(self.returns_list), 1 / len(self.returns_list))
    
    # minimize the variance
    result = minimize(self.__minimum_variance_objective_function, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
      raise Exception(result.message)
    return result.x