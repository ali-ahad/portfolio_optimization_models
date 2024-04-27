import pandas as pd

class ConstantCorrelationModelOptimization:
  def __init__(self, expected_returns: list[float], standard_deviation: list[float], correlation: float, risk_free_return: float) -> None:
    self.__expected_returns = expected_returns
    self.__standard_deviation = standard_deviation
    self.__corelation = correlation
    self.__risk_free_return = risk_free_return
    
  def find_weights_short_selling_allowed(self):
    k = len(self.__expected_returns)
    
    # find the sum of all treynor indexes
    sum_treynor_indexes = 0.0
    for i in range(k):
      sum_treynor_indexes += (self.__expected_returns[i] - self.__risk_free_return) / self.__standard_deviation[i]
    cutoff_const = (self.__corelation / (1 - self.__corelation + (k * self.__corelation))) * sum_treynor_indexes

    # find the non normalized weights
    non_normalized_weights = []
    for i in range(k):
      treynor_index = (self.__expected_returns[i] - self.__risk_free_return) / self.__standard_deviation[i]
      z = (1 / ((1 - self.__corelation) * self.__standard_deviation[i])) * (treynor_index - cutoff_const)
      non_normalized_weights.append(z)
    
    # normalize the weights
    sum_weights = sum(non_normalized_weights)
    return [weight / sum_weights for weight in non_normalized_weights]
  
  def find_weights_no_short_selling(self):
    # find the ranks as treynor indexes
    ranks_vector = []
    for i in range(len(self.__expected_returns)):
      treynor_index = (self.__expected_returns[i] - self.__risk_free_return) / self.__standard_deviation[i]
      ranks_vector.append(treynor_index)
      
    # create a datafram with execpted returns, standard deviation and ranks
    data = {
      'Expected Returns': self.__expected_returns, 
      'Standard Deviation': self.__standard_deviation, 
      'Ranks': ranks_vector}
    df = pd.DataFrame(data)
    
    # sort the dataframe by ranks in descending order
    df = df.sort_values(by='Ranks', ascending=False)
    
    # find the cutoff rank and therefore the cutoff constant
    # if the cumulative cutoff constant gets less than the corresponding rank,
    # then every asset from that rank will have a weight of 0
    cumulative_treynor_indx = 0.0
    cutoff_constant = 0.0
    cutoff_idx = 0
    
    count = 0
    for _, row in df.iterrows():
      cumulative_treynor_indx += (row['Expected Returns'] - self.__risk_free_return) / row['Standard Deviation']
      
      b = count + 1
      cutoff_constant = (self.__corelation / (1 - self.__corelation + (b * self.__corelation))) * cumulative_treynor_indx
      
      if row['Ranks'] < cutoff_constant:
        cutoff_idx = count
        break
      count += 1
    cutoff_idx = count
      
    # at this point we will have the cutoff constant and the cutoff index
    # we will assign 0 weights to all assets from the cutoff index to the end
    # and then we will normalize the weights
    non_normalized_weights = [0] * len(self.__expected_returns)
    
    count = 0
    for idx, row in df.iterrows():
      if count >= cutoff_idx:
        break
      treynor_index = (row['Expected Returns'] - self.__risk_free_return) / row['Standard Deviation']
      z = (1 / ((1 - self.__corelation) * row['Standard Deviation'])) * (treynor_index - cutoff_constant)
      non_normalized_weights[idx] = z
      count += 1
    
    sum_weights = sum(non_normalized_weights)
    return [weight / sum_weights for weight in non_normalized_weights]
    