import pickle
from MarkowitzPortfolio_lib import initialize_portfolio

def main():
    # 1. Define your universe of assets (European tickers + Crypto)
    tickers = ['SXR8.DE', 'IDR.MC', 'EGLN.L', 'BTC-EUR','JNJ.F']

    current_money = {
        'SXR8.DE': 0.0,
        'IDR.MC': 450.07,
        'EGLN.L': 313.19,
        'BTC-EUR': 190.77,
        'JNJ.F':0
    }
    
    # 3. Create and fund the portfolio
    my_portfolio = initialize_portfolio(tickers, current_money )
    
    # 4. Save the configured object to a local file for future updates
    filename = "Current_Diverse_Portfolio.pkl"
    print(f"Saving portfolio object to '{filename}'...")

    with open(filename, "wb") as file:
        pickle.dump(my_portfolio, file)

    print(f"Success! Your portfolio is now saved as '{filename}' and ready for the monthly report.")

if __name__ == "__main__":
    main()