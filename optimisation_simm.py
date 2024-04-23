class SingleFactorModelOptimization:
  def __init__(self, returns_vector: list[float], beta_vector: list[float], residuals_vector: list[float], market_return: float, risk_free_return: float) -> None:
    self.__returns_vector = returns_vector
    self.__beta_vector = beta_vector
    self.__residuals_vector = residuals_vector
    self.__market_return = market_return
    self.__risk_free_return = risk_free_return
    
    self.__market_return_square = market_return ** 2
    self.__residuals_vector_square = np.square(residuals_vector)
  
  # find the weights when short selling is allowed
  def find_weights_short_selling_allowed(self) -> list[float]:
    
    cutoff_numerator = 0.0
    for i in range(len(self.__returns_vector)):
      cutoff_numerator += (self.__beta_vector[i] * (self.__returns_vector[i] - self.__risk_free_return)) / self.__residuals_vector_square[i]
    cutoff_numerator = cutoff_numerator * self.__market_return_square
    
    cutoff_denominator = 0.0
    for i in range(len(self.__returns_vector)):
      cutoff_denominator += (self.__beta_vector[i] ** 2) / self.__residuals_vector_square[i]
    cutoff_denominator = 1 + (cutoff_denominator * self.__market_return_square)
    
    cutoff_constant = cutoff_numerator / cutoff_denominator
    
    # find Zi
    non_normalized_weights = []
    for i in range(len(self.__returns_vector)):
      treynor_index = (self.__returns_vector[i] - self.__risk_free_return) / self.__beta_vector[i]
      z = (self.__beta_vector[i] / self.__residuals_vector_square[i]) * (treynor_index - cutoff_constant)
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
      'Residuals': self.__residuals_vector, 
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
    
    for idx, row in df.iterrows():
      numerator = (row['Beta'] * (row['Expected Returns'] - self.__risk_free_return)) / (row['Residuals'] ** 2)
      numerator = numerator * self.__market_return_square
      
      denominator = (row['Beta'] ** 2) / (row['Residuals'] ** 2)
      denominator = 1 + (denominator * self.__market_return_square)
      
      cutoff_constant += numerator / denominator
      if row['Ranks'] < cutoff_constant:
        cutoff_idx = idx
        break
    
    # at this point we will have the cutoff constant and the cutoff index
    # we will assign 0 weights to all assets from the cutoff index to the end
    # and then we will normalize the weights
    non_normalized_weights = [0] * len(self.__returns_vector)
    
    for idx, row in df.iterrows():
      if idx >= cutoff_idx:
        break
      treynor_index = (row['Expected Returns'] - self.__risk_free_return) / row['Beta']
      z = (row['Beta'] / row['Residuals'] ** 2) * (treynor_index - cutoff_constant)
      non_normalized_weights[idx] = z
    
    sum_weights = sum(non_normalized_weights)
    return [weight / sum_weights for weight in non_normalized_weights]
  
    