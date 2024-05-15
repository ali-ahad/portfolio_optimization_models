import yfinance as yf
import numpy as np

class Utils:
  
  # get Date and Adj Close
  @staticmethod
  def get_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Adj Close']]
    return data
  
  # since, there is no missing data, we can go to the next step
  # find simple return over the entire year
  # Rt(k) = (Pt(k) - Pt(k-1)) / Pt(k-1)
  # Rt(k) = (1 + Rt) * (1 + Rt-1) * ... * (1 + R1) - 1
  @staticmethod
  def calculate_simple_return(data):
    data['simple_return'] = data['Adj Close'].pct_change()
    data['simple_return'] = data['simple_return'].fillna(0)
    data['simple_return'] = data['simple_return'] + 1
    data['simple_return'] = data['simple_return'].cumprod() - 1
    
    return data.iloc[-1]['simple_return']
  
  # find the daily returns
  @staticmethod
  def calculate_daily_returns(data):
    daily_returns = data['Adj Close'].pct_change()
    daily_returns = daily_returns.dropna()
    
    return daily_returns
  
  # calculate portfolio returns and risk
  @staticmethod
  def calculate_markowitz_portfolio_risk(weights, cov_matrix):
      risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
      
      return risk
    
  # calculate sharpe ratio
  @staticmethod
  def calculate_sharpe_ratio(expected_returns, risk_free_rate, risk):
    return (expected_returns - risk_free_rate) / risk
  
  @staticmethod
  def calculate_portfolio_mean_return(mean_returns,weights,annulized_factor):
    portfolio_mean_return = np.dot(mean_returns , weights) #daily

    return portfolio_mean_return *np.sqrt(annulized_factor)
  
  def calculate_ccm_model_percentage_portfolio_risk(weights, constant_correlation, std_dev_list):
    # beta values are the standard_deviation
    beta_values = std_dev_list
    
    # market_variance is the constant_correlation
    market_variance = constant_correlation
    
    # tau squared_list is (1- constant_correlation) * std_dev ** 2
    tau_squared_list = [(1 - constant_correlation) * beta ** 2 for beta in beta_values]
    squared_weights = [weight ** 2 for weight in weights]
    
    portfolio_beta = np.dot(weights, beta_values)
    portfolio_beta_squared = portfolio_beta ** 2
    portfolio_tau_squared = np.dot(squared_weights, tau_squared_list)
  
    portfolio_variance = (portfolio_beta_squared * market_variance) + portfolio_tau_squared
    portfolio_risk = np.sqrt(portfolio_variance)
    
    return portfolio_risk
    
  @staticmethod
  def calculate_factor_model_percentage_portfolio_risk(weights, expected_market_variance, alpha_list, beta_list, tau_list):
    squared_weights = [weight ** 2 for weight in weights]
    tau_squared = [tau ** 2 for tau in tau_list]

    portfolio_alpha = np.dot(weights, beta_list)
    portfolio_beta = np.dot(weights, alpha_list)

    portfolio_beta = np.dot(weights, beta_list)
    portfolio_beta_squared = portfolio_beta ** 2
    
    portfolio_tau_squared = np.dot(squared_weights, tau_squared)
    
    portfolio_varinace = (portfolio_beta_squared * expected_market_variance) + portfolio_tau_squared
    portfolio_risk = np.sqrt(portfolio_varinace)
    
    return portfolio_risk