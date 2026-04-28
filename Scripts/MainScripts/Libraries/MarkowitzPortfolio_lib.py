import numpy as np
from scipy.optimize import minimize
from datetime import datetime
from FinanceFunctions_lib import get_current_prices

class Portfolio:
    def __init__(self):
        self.tags = []
        self.expected_returns = []
        self._covariance_matrix = None
        self._weights = np.array([])
        self.investments = {}
        self.Non_sellable_tickers=[]

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        new_weights = np.array(new_weights)
        total_weight = np.sum(new_weights)
        if total_weight == 0:
            n = len(new_weights)
            self._weights = np.ones(n) / n
        else:
            self._weights = new_weights / total_weight

    def add_financial_asset(self, tag, expected_return):
            self.tags.append(tag)
            self.expected_returns.append(expected_return)
            n = len(self.tags)
            self.weights = np.ones(n) / n
            
            # Initialize investment tracking for the new asset at 0, adding 'total_shares'
            self.investments[tag] = {
                'total_invested': 0.0,
                'total_shares': 0.0,  # New attribute to track the amount of assets purchased
                'transaction_history': []
            }

    def add_money(self, contributions, current_prices, transaction_date=None):
            """
            Adds money to the specified assets by delegating to the single-asset method.
            """
            if transaction_date is None:
                transaction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
            for tag, amount in contributions.items():
                if tag in self.tags:
                    if tag not in current_prices:
                        print(f"Error: No price provided for '{tag}'. Transaction skipped.")
                        continue
                    
                    # Delegate the math and recording to the specialized method
                    self.add_money_to_asset_at_price(tag, amount, current_prices[tag], transaction_date)
                else:
                    print(f"Error: Asset '{tag}' is not in the portfolio. Please add it first.")
                    
    def calculate_rebalance_contribution(self, new_contribution, current_prices):
        """
        Calculates how to distribute a new monetary contribution to realign 
        the current market value of the portfolio with the target weights.
        Ensures mathematical integrity even if some assets lack current prices
        by falling back to the last known transaction price.
        """
        if not self.tags:
            print("[ERROR] Portfolio is empty. Cannot calculate rebalance.")
            return {}

        # 1. Identify valid tags to dynamically re-normalize weights and prevent cash vaporization
        valid_weights_sum = 0.0
        for i, tag in enumerate(self.tags):
            if tag in current_prices:
                valid_weights_sum += self.weights[i]
        
        if valid_weights_sum == 0:
            print("[ERROR] No valid prices available for any asset. Halting rebalance.")
            return {}
            
        weight_normalization_factor = 1.0 / valid_weights_sum

        # 2. Calculate the true CURRENT MARKET VALUE of the portfolio
        current_market_values = {}
        current_total_value = 0.0
        
        for tag in self.tags:
            shares = self.investments[tag].get('total_shares', 0.0)
            
            if tag not in current_prices:
                # Find the last known price from transaction history to avoid capital leakage
                last_known_price = 0.0
                history = self.investments[tag].get('transaction_history', [])
                if history:
                    # Extract the price from the most recent recorded transaction
                    last_known_price = history[-1].get('price_per_share', 0.0)
                
                raise ValueError(f"[ERROR] Missing live price for '{tag}'. Using last known price (${last_known_price:.2f}) for valuation.")
            else:
                market_val = shares * current_prices[tag]
                current_market_values[tag] = market_val
                current_total_value += market_val
            
        # 3. The new target total after adding the cash contribution
        target_total = current_total_value + new_contribution

        adjustments = {}
        
        print(f"Current Market Value: ${current_total_value:.2f}")
        print(f"New Contribution: ${new_contribution:.2f}")
        print(f"Target Total: ${target_total:.2f}")
        print("-" * 60)
        
        for i, tag in enumerate(self.tags):
            # Ignore assets with missing prices for the actual buying/selling execution
            if tag not in current_prices:
                raise ValueError(f"[INFO] Skipping adjustments for '{tag}' due to missing live execution price.")
                
                
            # Re-normalize the target weight strictly to the available active assets
            target_weight = self.weights[i] * weight_normalization_factor
            target_amount = target_total * target_weight
            
            current_amount = current_market_values[tag] 
            
            amount_to_invest = target_amount - current_amount
            adjustments[tag] = amount_to_invest
            
            if amount_to_invest >= 0:
                print(f"[{tag}] Current Value: ${current_amount:.2f} | Target Weight: {target_weight*100:.1f}% | Need to add: ${amount_to_invest:.2f}")
            else:
                print(f"[{tag}] Current Value: ${current_amount:.2f} | Target Weight: {target_weight*100:.1f}% | Overweight! Sell: ${abs(amount_to_invest):.2f}")
                
        print("-" * 60)
        return adjustments

    def set_covariance_matrix(self, matrix):
        matrix = np.array(matrix)
        n = len(self.tags)
        if matrix.shape != (n, n):
            raise ValueError(f"Matrix shape error. Expected ({n}, {n}), got {matrix.shape}.")
        self._covariance_matrix = matrix

    def get_current_portfolio_state(self, current_prices, annualized_variances=None):
            """
            Calculates the current value of the portfolio using externally provided prices.
            
            :param current_prices: Dictionary mapping tags to their current price.
            :param annualized_variances: Optional dictionary mapping tags to their variance.
            :return: A tuple containing a dictionary with individual asset states and the total portfolio value.
            """
            asset_states = {}
            total_portfolio_value = 0.0
            
            for tag in self.tags:
                if tag not in current_prices:
                    raise ValueError(f"[ERROR] No price provided for '{tag}'")
                    
                    
                current_price = current_prices[tag]
                total_shares = self.investments[tag].get('total_shares', 0.0)
                current_value = total_shares * current_price
                
                variance = annualized_variances.get(tag, 0.0) if annualized_variances else 0.0
                
                asset_states[tag] = {
                    'current_price': current_price,
                    'current_value': current_value,
                    'annualized_variance': variance
                }
                
                total_portfolio_value += current_value
                print(f"[{tag}] Price: ${current_price:.2f} | Value: ${current_value:.2f} | Variance: {variance:.6f}")
                    
            print(f"Calculation complete. Total Portfolio Value: ${total_portfolio_value:.2f}")
            return asset_states, total_portfolio_value
    
    def get_portfolio_return(self, custom_weights=None):
        w = custom_weights if custom_weights is not None else self.weights
        return np.dot(w, self.expected_returns)

    def get_portfolio_variance(self, custom_weights=None):
        w = custom_weights if custom_weights is not None else self.weights
        return np.dot(w.T, np.dot(self._covariance_matrix, w))

    def optimize_minimize_risk(self, target_return):
        n = len(self.tags)
        bounds = tuple((0, 1) for _ in range(n))
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: self.get_portfolio_return(w) - target_return}
        ]
        result = minimize(self.get_portfolio_variance, self.weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success: self.weights = result.x
        return result

    def optimize_maximize_return(self, target_variance):
        n = len(self.tags)
        bounds = tuple((0, 1) for _ in range(n))
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: target_variance - self.get_portfolio_variance(w)}
        ]
        result = minimize(lambda w: -self.get_portfolio_return(w), self.weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success: self.weights = result.x
        return result

    def optimize_maximize_utility(self, risk_aversion):
        n = len(self.tags)
        bounds = tuple((0, 1) for _ in range(n))
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(lambda w: -(self.get_portfolio_return(w) - 0.5 * risk_aversion * self.get_portfolio_variance(w)), self.weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success: self.weights = result.x
        return result

    def optimize_maximize_sharpe_ratio(self, risk_free_rate=0.0):
            """
            Calculates the optimal weights for the Tangency Portfolio (Maximum Sharpe Ratio),
            which represents the optimal risk-adjusted point on the efficient frontier.
            
            :param risk_free_rate: The theoretical return of a zero-risk investment.
            """
            n = len(self.tags)
            bounds = tuple((0, 1) for _ in range(n))
            
            # Scipy only minimizes, so we minimize the negative Sharpe Ratio
            def negative_sharpe_ratio(w):
                p_return = self.get_portfolio_return(w)
                p_variance = self.get_portfolio_variance(w)
                p_volatility = np.sqrt(p_variance)
                
                # Prevent division by zero in extreme edge cases
                if p_volatility == 0:
                    return 0
                    
                sharpe_ratio = (p_return - risk_free_rate) / p_volatility
                return -sharpe_ratio
                
            # The only constraint is that weights must sum to 100%
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
            
            result = minimize(
                negative_sharpe_ratio, 
                self.weights, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            if result.success:
                self.weights = result.x
                
            return result
    
    def add_money_to_asset_at_price(self, tag, amount, price, transaction_date=None):
            """
            Invests a specific amount of money into a specific asset at a manually provided price.
            
            :param tag: String representing the asset tag (e.g., 'SPY').
            :param amount: Float representing the amount of money to invest.
            :param price: Float representing the exact price per share to use for the transaction.
            :param transaction_date: String representing the date (optional). If None, current date is used.
            """
            if tag not in self.tags:
                raise ValueError(f"Error: Asset '{tag}' is not in the portfolio. Please add it first.")
                
            if price <= 0:
                raise ValueError("Error: The price per share must be strictly greater than zero.")

            if transaction_date is None:
                transaction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Calculate the amount of assets (shares) purchased
            shares_bought = amount / price

            # Update attributes
            self.investments[tag]['total_invested'] += amount
            self.investments[tag]['total_shares'] += shares_bought

            # Save transaction details including price and shares
            self.investments[tag]['transaction_history'].append({
                'date': transaction_date,
                'amount': amount,
                'price_per_share': price,
                'shares_bought': shares_bought
            })

            print(f"Success: Added ${amount:.2f} to '{tag}' on {transaction_date}. "
                f"Bought {shares_bought:.6f} shares at ${price:.2f} each.")
            
    def calculate_current_sharpe_ratio(self, risk_free_rate=0.0):
            """
            Calculates and returns the Sharpe Ratio of the portfolio using its 
            current weights, expected returns, and covariance matrix.
            Does not perform any optimization.
            
            :param risk_free_rate: The annualized risk-free rate. Default is 0.0.
            :return: The computed Sharpe Ratio (float).
            """
            
            if not self.weights or not self.expected_returns or self._covariance_matrix is None:
                print("Error: Missing weights, expected returns, or covariance matrix. Cannot compute Sharpe Ratio.")
                return 0.0
                
            weights_array = np.array(self.weights)
            returns_array = np.array(self.expected_returns)
            
            # 1. Calculate the expected return of the current portfolio
            portfolio_return = np.dot(weights_array, returns_array)
            
            # 2. Calculate the volatility (standard deviation) of the current portfolio
            portfolio_variance = np.dot(weights_array.T, np.dot(self._covariance_matrix, weights_array))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # 3. Handle edge cases (division by zero)
            if portfolio_volatility == 0:
                print("Warning: Portfolio volatility is exactly zero. Sharpe Ratio is undefined.")
                return 0.0
                
            # 4. Compute the Sharpe Ratio
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            # 5. Output the metrics to the console
            print("\n--- Current Portfolio Metrics ---")
            print(f"Expected Return: {portfolio_return:>7.2%}")
            print(f"Volatility:      {portfolio_volatility:>7.2%}")
            print(f"Risk-Free Rate:  {risk_free_rate:>7.2%}")
            print(f"Sharpe Ratio:    {sharpe_ratio:>7.4f}")
            print("-" * 33 + "\n")
            
            return sharpe_ratio
    
    def get_amount_by_removing_assets(self, tags_to_remove, current_prices, transaction_date=None):
            """
            Liquidates and removes specific assets from the portfolio.
            Returns the total cash generated and a list of records for the Excel report.
            """
            if not tags_to_remove:
                return 0.0, []

            if transaction_date is None:
                transaction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            total_cash_generated = 0.0
            liquidation_records = []
            
            remaining_tags = []
            remaining_expected_returns = []
            
            for i, tag in enumerate(self.tags):
                if tag in tags_to_remove:
                    if tag not in current_prices:
                        print(f"[WARNING] Missing price to liquidate '{tag}'. Using $0.0.")
                        price = 0.0
                    else:
                        price = current_prices[tag]
                    
                    # Liquidate the asset
                    shares_owned = self.investments[tag].get('total_shares', 0.0)
                    cash_value = shares_owned * price
                    total_cash_generated += cash_value
                    
                    if shares_owned > 0 and price > 0:
                        # Record the sell transaction in the object's history
                        self.investments[tag]['total_shares'] = 0.0
                        self.investments[tag]['transaction_history'].append({
                            'date': transaction_date,
                            'amount': -cash_value,
                            'price_per_share': price,
                            'shares_bought': -shares_owned
                        })
                        print(f"Liquidated '{tag}': Sold {shares_owned:.6f} shares at ${price:.2f} (Total: ${cash_value:.2f})")
                        
                    # Create the record for the Excel sheet
                    liquidation_records.append({
                        'Ticker': tag,
                        'Target_Weight': "0.00%",
                        'Price': price,
                        'Shares_Owned': 0.0,
                        'Current_Market_Value': 0.0,
                        'Recommended_Action': -cash_value,
                        'Action_Taken': -cash_value
                    })
                else:
                    remaining_tags.append(tag)
                    if len(self.expected_returns) > i:
                        remaining_expected_returns.append(self.expected_returns[i])

            # Update the portfolio state by deleting the tags
            self.tags = remaining_tags
            self.expected_returns = remaining_expected_returns
            
            # Reset weights to avoid dimension mismatch (they will be overwritten by Markowitz anyway)
            n = len(self.tags)
            if n > 0:
                self.weights = np.ones(n) / n
            else:
                self.weights = np.array([])
                
            return total_cash_generated, liquidation_records
    def remove_financial_asset(self, tag):
        """
        Completely removes an asset from the active portfolio tracking.
        """
        if tag in self.tags:
            idx = self.tags.index(tag)
            self.tags.pop(idx)
            
            if len(self.expected_returns) > idx:
                self.expected_returns.pop(idx)
                
            # Reset weights to avoid dimension mismatches
            n = len(self.tags)
            if n > 0:
                self.weights = np.ones(n) / n
            else:
                self.weights = np.array([])
                
            # Safely remove from covariance matrix if it is already initialized
            if self._covariance_matrix is not None and self._covariance_matrix.shape == (n + 1, n + 1):
                self._covariance_matrix = np.delete(self._covariance_matrix, idx, axis=0)
                self._covariance_matrix = np.delete(self._covariance_matrix, idx, axis=1)
                
            print(f"System: Asset '{tag}' successfully removed from the active portfolio.")
        else:
            print(f"Warning: Asset '{tag}' is not in the portfolio.")

    def liquidate_and_remove_assets(self, tags_to_remove, current_prices, transaction_date=None):
        """
        Liquidates the specified assets, generating cash, and then explicitly removes them.
        Returns the total cash generated and a list of records for the Excel report.
        """
        if not tags_to_remove:
            return 0.0, []

        if transaction_date is None:
            transaction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        total_cash_generated = 0.0
        liquidation_records = []
        
        # Iterate over a copy to safely remove items from the original list during the loop
        for tag in list(tags_to_remove):
            if tag in self.tags:
                price = current_prices.get(tag, 0.0)
                if price <= 0.0:
                    print(f"[WARNING] Missing or invalid price to liquidate '{tag}'. Using $0.0.")
                    price = 0.0
                
                shares_owned = self.investments.get(tag, {}).get('total_shares', 0.0)
                cash_value = shares_owned * price
                total_cash_generated += cash_value
                
                if shares_owned > 0 and price > 0:
                    # Record the sell transaction
                    self.investments[tag]['total_shares'] = 0.0
                    self.investments[tag]['transaction_history'].append({
                        'date': transaction_date,
                        'amount': -cash_value,
                        'price_per_share': price,
                        'shares_bought': -shares_owned
                    })
                    print(f"Liquidated '{tag}': Sold {shares_owned:.6f} shares at ${price:.2f} (Total: ${cash_value:.2f})")
                    
                # Create the Excel record
                liquidation_records.append({
                    'Ticker': tag,
                    'Target_Weight': "0.00%",
                    'Price': price,
                    'Shares_Owned': 0.0,
                    'Current_Market_Value': 0.0,
                    'Recommended_Action': -cash_value,
                    'Action_Taken': -cash_value
                })
                
                # Explicitly remove the asset from active tracking metrics
                self.remove_financial_asset(tag)

        return total_cash_generated, liquidation_records
    
    def update_universe_full_asset_rotation(self, new_assets, current_prices, transaction_date=None):
        """
        Updates the portfolio's universe by liquidating unapproved assets 
        and adding new approved ones.
        
        :param new_assets: List of tickers that passed the current filters.
        :param current_prices: Dictionary mapping tickers to their current price.
        :param transaction_date: String representing the date (optional).
        :return: A tuple containing the extra cash generated and the liquidation records.
        """
        # 1. Identify assets that are in the portfolio but did NOT pass the current month's filters
        assets_to_sell = [tag for tag in self.tags if tag not in new_assets]
        
        # 2. Liquidate them to free up capital AND remove them completely
        extra_cash, liquidation_records = self.liquidate_and_remove_assets(
            assets_to_sell, current_prices, transaction_date=transaction_date
        )
        
        # 3. The new universe strictly consists of the approved equities (legacy assets have been removed)
        for tag in new_assets:
            if tag not in self.tags:
                self.add_financial_asset(tag, 0.0)
                
        return extra_cash, liquidation_records
    
    def update_expected_returns(self, expected_returns_dict):
        """
        Updates the expected returns of the portfolio, ensuring they are strictly aligned
        with the current portfolio tags. Prints a warning for any missing tags.
        
        :param expected_returns_dict: Dictionary mapping tickers to their expected returns.
        """
        aligned_returns = []
        
        for tag in self.tags:
            value = expected_returns_dict.get(tag)
            if value is None:
                raise ValueError(f"[ERROR] Missing expected return for '{tag}'.")
            else:
                aligned_returns.append(value)
                
        self.expected_returns = aligned_returns
    
    def operate_adjustments(self, adjustments, current_prices, transaction_date=None):
        """
        Executes the portfolio rebalancing based on previously calculated adjustments.
        """
        
        if transaction_date is None:
            transaction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        for tag in self.tags:
            action_amount = adjustments.get(tag, 0.0)
            price_used = current_prices.get(tag, 0.0)
            
            # Execute the rebalance
            if action_amount != 0 and price_used > 0:
                self.add_money_to_asset_at_price(tag, action_amount, price_used, transaction_date=transaction_date)

    def update_universe_fundamental_floor(self, new_candidates, healthy_universe, ranked_df, current_prices, transaction_date=None, percentile_exit_threshold=90.0):
            print(f"\n[INFO] Executing Fundamental Floor Strategy (Exit Threshold: {percentile_exit_threshold}th Percentile)")
            
            # Ensure all non-sellable tickers are present in the portfolio tracking
            for tag in self.Non_sellable_tickers:
                if tag not in self.tags:
                    self.add_financial_asset(tag, 0.0)
                    print(f"[INFO] Initialized missing protected asset in tracking: {tag}")

            assets_to_sell = []
            
            for tag in list(self.tags): 
                # Check 1: Fundamental health check
                if tag not in healthy_universe and tag not in self.Non_sellable_tickers:
                    assets_to_sell.append(tag)
                    print(f" -> {tag}: SELL - Failed structural health check")
                    continue 
                    
                # Check 2: Attractive ranking
                asset_data = ranked_df[ranked_df['Ticker'] == tag]
                
                if not asset_data.empty:
                    current_rank = asset_data['Final_Percentile_Rank'].values[0]
                    if current_rank < percentile_exit_threshold and tag not in self.Non_sellable_tickers:
                        assets_to_sell.append(tag)
                        print(f" -> {tag}: SELL - Dropped in ranking (Percentile: {current_rank:.2f})")
                else:
                    if tag not in self.Non_sellable_tickers:
                        assets_to_sell.append(tag)
                        print(f" -> {tag}: SELL - Missing updated ranking data")

            # 2. Liquidate and remove the identified assets
            extra_cash, liquidation_records = self.liquidate_and_remove_assets(
                assets_to_sell, current_prices, transaction_date=transaction_date
            )
            
            # 3. Add new candidates from the Top N list that are not already present
            # This expands the universe slightly but prioritizes keeping existing strong positions.
            for tag in new_candidates:
                if tag not in self.tags:
                    self.add_financial_asset(tag, 0.0)
                    print(f"[INFO] Adding new high-rank candidate: {tag}")
                    
            return extra_cash, liquidation_records
    
    def optimize_sharpe_no_sell_iterative(self, new_contribution, current_prices, risk_free_rate=0.02):
        """
        Optimizes the Sharpe Ratio while preventing sales of existing assets.
        Iteratively adds constraints, starting with the asset the optimizer 
        most wants to sell, until no sales occur.
        """
        n = len(self.tags)
        target_total = 0.0
        current_vals = []
        
        # 1. Calculate the current value and the total target value
        for tag in self.tags:
            shares = self.investments[tag].get('total_shares', 0.0)
            price = current_prices.get(tag, 0.0)
            val = shares * price
            current_vals.append(val)
            target_total += val
            
        target_total += new_contribution
        
        # 2. Define minimum weights (floors) to prevent sales
        # The minimum weight for each asset is its current value / future total
        min_weights_allowed = [v / target_total for v in current_vals]
        
        # List of indices we will force not to sell
        locked_indices = []
        
        while True:
            # Define bounds: if locked, its minimum is its current weight. 
            # Otherwise, it's 0 (we let the optimizer decide).
            current_bounds = []
            for i in range(n):
                low = min_weights_allowed[i] if i in locked_indices else 0.0
                current_bounds.append((low, 1.0))

            def negative_sharpe_ratio(w):
                p_return = self.get_portfolio_return(w)
                p_variance = self.get_portfolio_variance(w)
                p_volatility = np.sqrt(p_variance)
                if p_volatility == 0: return 0
                return -(p_return - risk_free_rate) / p_volatility

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            result = minimize(
                negative_sharpe_ratio, 
                self.weights, 
                method='SLSQP', 
                bounds=tuple(current_bounds), 
                constraints=constraints
            )

            if not result.success:
                print("[WARNING] Optimization did not converge, using last valid weights.")
                break

            # 3. Calculate theoretical adjustments with this result
            temp_weights = result.x
            potential_sales = []
            
            for i in range(n):
                target_amount = temp_weights[i] * target_total
                adjustment = target_amount - current_vals[i]
                
                # If the adjustment is negative (sale) and it's not already locked
                if adjustment < -0.01 and i not in locked_indices:
                    potential_sales.append((adjustment, i))
            
            # 4. Exit condition: if no sales are detected
            if not potential_sales:
                self.weights = temp_weights
                break
            
            # 5. "Largest to smallest" logic: block the asset with the largest proposed sale
            potential_sales.sort() # Sort by the most negative value
            _, worst_idx = potential_sales[0]
            locked_indices.append(worst_idx)
            
            print(f"[ITER] Blocking sale for {self.tags[worst_idx]} (Adjustment: ${potential_sales[0][0]:.2f}). Re-optimizing...")

        return result
    
    def optimize_sharpe_no_sell_bounded(self, new_contribution, current_prices, risk_free_rate=0.02):
        """
        Optimizes the Sharpe Ratio while strictly preventing sales of existing assets.
        Instead of iterating, it solves the convex optimization problem in a single 
        pass by setting the lower bound of each asset's weight to its current diluted weight.
        """
        n = len(self.tags)
        target_total = 0.0
        current_vals = []
        
        # 1. Calculate the current value of each asset and the new target total portfolio value
        for tag in self.tags:
            shares = self.investments[tag].get('total_shares', 0.0)
            price = current_prices.get(tag, 0.0)
            val = shares * price
            current_vals.append(val)
            target_total += val
            
        target_total += new_contribution
        
        # Guard clause to prevent division by zero in empty portfolios
        if target_total == 0:
            print("[ERROR] Target total is zero. Cannot optimize bounds.")
            return None

        # 2. Define minimum weights (floors) to strictly prevent any sales
        # The lowest weight an asset can have is its current absolute value over the new total
        bounds = []
        for i in range(n):
            min_weight = current_vals[i] / target_total
            bounds.append((min_weight, 1.0))
            
        bounds = tuple(bounds)

        # 3. Define the objective function (Minimize Negative Sharpe Ratio)
        def negative_sharpe_ratio(w):
            p_return = self.get_portfolio_return(w)
            p_variance = self.get_portfolio_variance(w)
            p_volatility = np.sqrt(p_variance)
            
            # Prevent division by zero if volatility crashes
            if p_volatility == 0: 
                return 0
                
            return -(p_return - risk_free_rate) / p_volatility

        # 4. Constraint: The sum of all weights must be exactly 1.0 (100%)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        print("[INFO] Executing bounded No-Sell Sharpe optimization")
        
        result = minimize(
            negative_sharpe_ratio, 
            self.weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )

        if result.success:
            w = result.x
            # Zero out sub-threshold weights to prevent ghost positions from floating-point noise.
            # SLSQP can return values like 1e-15 for assets with zero current value and zero
            # target weight; these propagate into shares via calculate_rebalance_contribution.
            w[w < 1e-6] = 0.0
            w /= w.sum()
            self.weights = w
        else:
            raise ValueError(f"[WARNING] Optimization did not converge: {result.message}. Retaining previous weights.")

        return result
    
def initialize_portfolio(tickers, current_money):
    """
    Creates a new Portfolio object, adds the specified financial assets, 
    fetches their current market prices, and logs the initial cash investments.
    
    :param tickers: List of strings representing the asset tickers.
    :param current_money: Dictionary mapping tickers to the initial invested fiat amount.
    :return: An initialized and funded Portfolio object.
    """
    print("\n--- INITIALIZING NEW PORTFOLIO ---")
    my_portfolio = Portfolio()

    # Add the chosen assets. Expected returns are set to 0.0 initially
    # as they will be overwritten by the quantitative estimators later.
    for ticker in tickers:
        my_portfolio.add_financial_asset(ticker, 0.0)

    print("Fetching current market prices for the initial setup...")
    current_prices = get_current_prices(tickers)
    
    print("Logging initial investments and calculating total shares...")
    # Inject the fiat money into the portfolio using the real market prices
    my_portfolio.add_money(current_money, current_prices)
    
    print("--- PORTFOLIO INITIALIZATION COMPLETE ---\n")
    return my_portfolio