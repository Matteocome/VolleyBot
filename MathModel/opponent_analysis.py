import pandas as pd
import numpy as np

# Match data from your table
matches = [
    {'Date': 'Tue 11 Nov 2025', 'Time': '18:35', 'Court': 'Sports Hall', 'Opposition': 'VolleyBoE', 'Result': '2 - 0'},
    {'Date': 'Tue 18 Nov 2025', 'Time': '20:50', 'Court': 'Sports Hall', 'Opposition': 'Eurostars', 'Result': '2 - 0'},
    {'Date': 'Tue 25 Nov 2025', 'Time': '19:20', 'Court': 'Sports Hall', 'Opposition': "Let's get tipsy!", 'Result': '1 - 2'},
    {'Date': 'Tue 02 Dec 2025', 'Time': '20:50', 'Court': 'Sports Hall', 'Opposition': 'Nacho Libero', 'Result': '2 - 1'},
    {'Date': 'Tue 09 Dec 2025', 'Time': '19:20', 'Court': 'Sports Hall', 'Opposition': 'See It, Set It, Sorted', 'Result': '3 - 0'},
    {'Date': 'Tue 16 Dec 2025', 'Time': '20:50', 'Court': 'Sports Hall', 'Opposition': 'Set and the City', 'Result': '2 - 1'},
    {'Date': 'Tue 06 Jan 2026', 'Time': '18:35', 'Court': 'Sports Hall', 'Opposition': 'The Big Slappers', 'Result': None},
    {'Date': 'Tue 13 Jan 2026', 'Time': '20:05', 'Court': 'Sports Hall', 'Opposition': 'VolleyBoE', 'Result': None},
    {'Date': 'Tue 20 Jan 2026', 'Time': '18:35', 'Court': 'Sports Hall', 'Opposition': 'Eurostars', 'Result': None},
    {'Date': 'Tue 27 Jan 2026', 'Time': '19:20', 'Court': 'Sports Hall', 'Opposition': "Let's get tipsy!", 'Result': None},
]

# Create dataframe
df = pd.DataFrame(matches)

# Categorize opponents based on performance
def categorize_opponent(opposition, matches_df):
    """
    Categorize opponent as WEAK, MEDIUM, or STRONG based on match results.
    - WEAK: Won 2-0 or 3-0 (clean sweep)
    - MEDIUM: Won 2-1 (close match)
    - STRONG: Lost or no pattern yet
    """
    opponent_matches = matches_df[matches_df['Opposition'] == opposition]
    
    if opponent_matches.empty:
        return 'UNKNOWN'
    
    # Get results where we have data
    played = opponent_matches[opponent_matches['Result'].notna()]
    
    if played.empty:
        return 'UNKNOWN'
    
    # Parse first result (most recent/definitive)
    result_str = played.iloc[0]['Result']
    our_score, their_score = map(int, result_str.split(' - '))
    
    # Categorize
    if our_score > their_score:
        # We won
        if our_score >= 2 and their_score == 0:
            return 'WEAK'
        elif our_score == 3:
            return 'WEAK'
        else:
            return 'MEDIUM'
    else:
        # We lost or tied
        return 'STRONG'

# Apply categorization
df['Strength'] = df['Opposition'].apply(lambda x: categorize_opponent(x, df))

# Create opponent summary dataframe (unique opponents with strength)
opponent_summary = df[['Opposition', 'Strength', 'Result']].drop_duplicates(subset=['Opposition']).sort_values('Opposition')

print("\n" + "="*80)
print("OPPONENT CATEGORIZATION")
print("="*80)
print(opponent_summary.to_string(index=False))

# Group by strength
print("\n" + "="*80)
print("OPPONENTS BY STRENGTH")
print("="*80)

for strength in ['WEAK', 'MEDIUM', 'STRONG', 'UNKNOWN']:
    opponents = opponent_summary[opponent_summary['Strength'] == strength]['Opposition'].tolist()
    if opponents:
        print(f"\n{strength}:")
        for opp in opponents:
            result = opponent_summary[opponent_summary['Opposition'] == opp]['Result'].iloc[0]
            status = f"  {opp:30s} (Result: {result})" if result else f"  {opp:30s} (No result yet)"
            print(status)

# Save to CSV
opponent_summary.to_csv('MathModel/opponent_strength_analysis.csv', index=False)
print("\n\nSaved to MathModel/opponent_strength_analysis.csv")

# Also create a summary count
print("\n" + "="*80)
print("STRENGTH DISTRIBUTION")
print("="*80)
strength_counts = opponent_summary['Strength'].value_counts()
print(strength_counts)

# Create indexed opponent database
print("\n" + "="*80)
print("INDEXED OPPONENT DATABASE")
print("="*80)
opponent_summary_sorted = opponent_summary.sort_values('Opposition').reset_index(drop=True)
opponent_summary_sorted['Index'] = range(len(opponent_summary_sorted))

print("\nUse the index below to select an opponent:")
print(opponent_summary_sorted[['Index', 'Opposition', 'Strength']].to_string(index=False))

# Save indexed database
opponent_summary_sorted.to_csv('MathModel/opponent_index.csv', index=False)
print("\n\nSaved indexed database to MathModel/opponent_index.csv")

# Create lookup functions that can be imported
def get_opponent_by_index(index):
    """Return (opponent_name, strength) given an index."""
    if index < 0 or index >= len(opponent_summary_sorted):
        return None, None
    row = opponent_summary_sorted.iloc[index]
    return row['Opposition'], row['Strength']

def print_opponent_menu():
    """Print the opponent selection menu."""
    print("\n" + "="*80)
    print("SELECT OPPONENT BY INDEX")
    print("="*80)
    print(opponent_summary_sorted[['Index', 'Opposition', 'Strength']].to_string(index=False))
    return opponent_summary_sorted

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Example: get_opponent_by_index(0) returns:", get_opponent_by_index(0))
    print("="*80)
