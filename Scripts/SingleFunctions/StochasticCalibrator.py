import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MainScripts', 'Libraries'))
from SDE_lib import OUModel, CIRModel,GBMModel, HestonModelOptions, HestonModelWindowSigma


def main():
    print("=========================================")
    print("      SDE CALIBRATION TOOLKIT            ")
    print("=========================================")

    print("\n-----------------------------------------")

    # Correct Ticker for Apple
    ticker = "IDR.MC" 
    start = "2025-04-02" 
    end = "2026-04-02" 
    
    # 1. Run Geometric Brownian Motion
    gbm_model = GBMModel(ticker)
    gbm_model.fetch_data(start, end)
    gbm_model.calibrate()
    gbm_model.plot_calibration()
    
    print("\nSimulating future scenarios...")
    scenarios_gbm = gbm_model.simulate_scenarios(n_paths=1000, days_to_simulate=126)
    
    if scenarios_gbm is not None:
        gbm_model.plot_scenarios(scenarios_gbm)

    print("\n-----------------------------------------")

        # 1. Run Geometric Brownian Motion
    hws_model = HestonModelWindowSigma(ticker)
    hws_model.fetch_data(start, end)
    hws_model.calibrate()
    hws_model.plot_calibration()
    
    print("\nSimulating future scenarios...")
    scenarios_hws = hws_model.simulate_scenarios(n_paths=1000, days_to_simulate=126)
    
    if scenarios_hws is not None:
        hws_model.plot_scenarios(scenarios_hws)

    print("\n-----------------------------------------")
    
    # 2. Run Heston Model by Options
    heston_model = HestonModelOptions(ticker)
    heston_model.fetch_data(start, end)
    
    print("\nFetching Options Data...")
    options_df = heston_model.fetch_implied_volatility_surface()
    
    if options_df is not None and not options_df.empty:
        heston_model.calibrate()
        
        print("\nSimulating future scenarios...")
        scenarios_heston = heston_model.simulate_scenarios(n_paths=1000, days_to_simulate=126)
        
        # SAFETY CHECK: Only plot if simulation was successful
        if scenarios_heston[0] is not None:
            heston_model.plot_scenarios(scenarios_heston)
        else:
            print(f"Plotting skipped: Simulation failed for {ticker}.")
    else:
        print(f"Skipping Heston calibration: No options data available for {ticker}.")

if __name__ == "__main__":
    main()
