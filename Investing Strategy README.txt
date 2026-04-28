Optimaize weights monthly
Operate between 15:30 and 17:30 Europe

Finanical Logic:

At each month: 

	Download all tickers for global índices in euros,
	Download data across a range of years,

	Discard companies with poor financial health,
	Calculate multipliers (EV_UFCF, EV_NOPAT, EV_IC, EV_EBITDA_CAPEX)
	Calculate market performance parameters (alpha_6F, betas, Momentum-Z)
	Rank by Windsor Zscores and keep the best performers on average,

	Add new recomended assets to the portfolio,
	Liquidate asstes that turned to be unhealthy and are not in a cheapest percentage
	Under the condition of not selling any ticker to prevent asset rotation:
		Optimize using the Sharpe ratio with James-Stein and Ledoit-Wolf fixing mínimum weights
	Finally, Operate.