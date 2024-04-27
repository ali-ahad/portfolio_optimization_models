import statsmodels.api as sm
import numpy as np
import pandas as pd

class SingleFactorModel:
  def __init__(self, asset_returns: list[float], market_returns: list[float]):
    self.__asset_returns = asset_returns
    self.__market_returns = market_returns
    self.__model_result = None
    
  def fit(self):
    x = sm.add_constant(self.__market_returns)
    self.__model_result = sm.OLS(self.__asset_returns, x).fit()
  
  def get_summary(self):
    return self.__model_result.summary()
  
  def get_params(self):
    params = self.__model_result.params
    pvalues = self.__model_result.pvalues
    
    return {
      "alpha": {
        "value": params[0],
        "pvalue": pvalues[0]
      },
      "beta": {
        "value": params[1],
        "pvalue": pvalues[1]
      },
      "tau": {
        "value": np.sqrt(self.__model_result.mse_resid),
        "pvalue": None
      }
    }
  
class SingleFactorModelOptimization:
  def __init__(self, returns_vector: list[float], beta_vector: list[float], tau_vector: list[float], market_variance: float, risk_free_return: float) -> None:
    self.__returns_vector = returns_vector
    self.__beta_vector = beta_vector
    self.__tau_vector = tau_vector
    self.__risk_free_return = risk_free_return
    
    self.__market_variance = market_variance
    self.__tau_vector_square = np.square(tau_vector)
  
  # find the weights when short selling is allowed
  def find_weights_short_selling_allowed(self) -> list[float]:
    
    cutoff_numerator = 0.0
    for i in range(len(self.__returns_vector)):
      cutoff_numerator += (self.__beta_vector[i] * (self.__returns_vector[i] - self.__risk_free_return)) / self.__tau_vector_square[i]
    cutoff_numerator = cutoff_numerator * self.__market_variance
    
    cutoff_denominator = 0.0
    for i in range(len(self.__returns_vector)):
      cutoff_denominator += (self.__beta_vector[i] ** 2) / self.__tau_vector_square[i]
    cutoff_denominator = 1 + (cutoff_denominator * self.__market_variance)
    
    cutoff_constant = cutoff_numerator / cutoff_denominator
    
    # find Zi
    non_normalized_weights = []
    for i in range(len(self.__returns_vector)):
      treynor_index = (self.__returns_vector[i] - self.__risk_free_return) / self.__beta_vector[i]
      z = (self.__beta_vector[i] / self.__tau_vector_square[i]) * (treynor_index - cutoff_constant)
      non_normalized_weights.append(z)
    
    # normalize the weights
    sum_weights = sum(non_normalized_weights)
    return [weight / sum_weights for weight in non_normalized_weights]
  
  # find weights when short selling is not allowed
  def find_weights_no_short_selling(self) -> list[float]:
    
    # find the rank which is the treynor index
    ranks_vector = []
    for i in range(len(self.__returns_vector)):
      treynor_index = (self.__returns_vector[i] - self.__risk_free_return) / self.__beta_vector[i]
      ranks_vector.append(treynor_index)
      
    # create a datafram with execpted returns, residuals, beta and ranks
    data = {
      'Expected Returns': self.__returns_vector, 
      'Tau': self.__tau_vector, 
      'Beta': self.__beta_vector, 
      'Ranks': ranks_vector}
    df = pd.DataFrame(data)
    
    # sort the dataframe by ranks in descending order
    df = df.sort_values(by='Ranks', ascending=False)
    
    # find the cutoff rank and therefore the cutoff constant
    # if the cumulative cutoff constant gets less than the corresponding rank,
    # then every asset from that rank will have a weight of 0
    cutoff_constant = 0.0
    cutoff_idx = 0 
    
    cumulative_numerator = 0.0
    cumulative_denomonator = 0.0
    
    count = 0
    for _, row in df.iterrows():
      cumulative_numerator += (row['Beta'] * (row['Expected Returns'] - self.__risk_free_return)) / row['Tau'] ** 2
      cumulative_denomonator += row['Beta'] ** 2 / row['Tau'] ** 2
      cutoff_constant = cumulative_numerator / ((1 + cumulative_denomonator) * self.__market_variance)

      
      if row['Ranks'] < cutoff_constant:
        cutoff_idx = count
        break
      count += 1  
    cutoff_idx = count # hence, we will trade all assets in buy

    # at this point we will have the cutoff constant and the cutoff index
    # we will assign 0 weights to all assets from the cutoff index to the end
    # and then we will normalize the weights
    non_normalized_weights = [0] * len(self.__returns_vector)
    
    count = 0
    for idx, row in df.iterrows():
      
      if count >= cutoff_idx:
        break
      treynor_index = (row['Expected Returns'] - self.__risk_free_return) / row['Beta']
      z = (row['Beta'] / row['Tau'] ** 2) * (treynor_index - cutoff_constant)
      non_normalized_weights[idx] = z
      count += 1
    
    sum_weights = sum(non_normalized_weights)
    
    return [weight / sum_weights for weight in non_normalized_weights]
  
    