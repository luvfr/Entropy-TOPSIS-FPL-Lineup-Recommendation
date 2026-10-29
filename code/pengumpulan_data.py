import os
import sys

import requests
import pandas as pd

def get_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()
    
    players = pd.DataFrame(data['elements'])
    
    out = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/players_raw_20252026.csv'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    players.to_csv(out, index=False)
    
    print("Data saved to CSV files")
    
if __name__ == "__main__":
    get_fpl_data()