"""
Enhanced Volleyball Rotation Strategy Model
============================================

Key Features:
1. 9 players available, must select best 6 for the court
2. Remaining 3 players sit on bench (no substitutions during play)
3. Optimize BOTH player selection AND rotation order
4. Based on opponent strength
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
from itertools import combinations, permutations
import pandas as pd
import sys

# Configure output path to match workspace structure
OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Documents', 'GitHub', 'AutoMess', 'volleyball-checker', 'MathModel')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to import opponent analysis utilities
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from opponent_analysis import get_opponent_by_index, print_opponent_menu
    OPPONENT_DATA_AVAILABLE = True
except ImportError:
    OPPONENT_DATA_AVAILABLE = False
    print("Warning: opponent_analysis module not found. Will use generic opponent strength only.")


def safe_savefig(fig, target_path, **savefig_kwargs):
    """Save a matplotlib `fig` to `target_path` via a temporary file and atomic replace.

    This avoids Windows "file in use" overwrite errors by writing to a temp file
    in the same directory and then using ``os.replace`` to atomically move it.
    """
    dirname = os.path.dirname(os.path.abspath(target_path)) or '.'
    fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(target_path)[1], dir=dirname)
    os.close(fd)
    try:
        fig.savefig(tmp, **savefig_kwargs)
        plt.close(fig)
        os.replace(tmp, target_path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise

class Player:
    """Player with three key attributes
        attack, defense, serving and overall rating [1 to 5] with 5 max skill
        """
    
    def __init__(self, name, defense, attack, serving):
        """
        Parameters:
        -----------
        name : str
        defense : float (1-5)
        attack : float (1-5)  
        serving : float (1-5)
        """
        self.name = name
        self.defense = defense
        self.attack = attack
        self.serving = serving
        self.overall = (defense + attack + serving) / 3
    
    def __repr__(self):
        return f"{self.name}(D:{self.defense:.1f}, A:{self.attack:.1f}, S:{self.serving:.1f}, O:{self.overall:.1f})"


class EnhancedRotationModel:
    """
    Model that optimizes:
    1. Which 6 players from 9 should be on court
    2. What rotation order should they follow
    """

    def __init__(self, 
                 all_players, 
                 opponent_strength='medium', 
                 min_defense_threshold=2.5, 
                 max_low_defense=2, 
                 forbid_adjacent_low=True):
        """
        Enhanced model with defense constraints
        #we are insuring that there are no rotations where more than 2 players have defense lower than 2.5

        Parameters:
        -----------
        all_players : list of 9 Player objects
        opponent_strength : str ('weak', 'medium', 'strong')
        model = EnhancedRotationModel(players, min_defense_threshold=3.0, max_low_defense=2)
        This prevents any rotation where more than 2 players have defense < 3.0.
        """
        assert len(all_players) >= 6, "Must have at least 6 players"
        self.all_players = all_players
        self.opponent_strength = opponent_strength
        # Defense constraint settings (disabled by default)
        # If any rotation has more than `max_low_defense` players with
        # `defense < min_defense_threshold`, the lineup will be considered invalid.
        self.min_defense_threshold = min_defense_threshold
        self.max_low_defense = max_low_defense
        # If True, forbid any two adjacent (neighboring) players both having
        # defense < min_defense_threshold in any rotation state.
        self.forbid_adjacent_low = forbid_adjacent_low
        
        # Position importance weights (by position 1-6)
        self.position_weights = { #weights sum to 1 as we are gicing importance to the action in each position State (1 to 6)
            1: {'serving': 0.7, 'attack': 0.1, 'defense': 0.2},  # Server position [bottom right]
            2: {'attack': 0.7, 'defense': 0.3, 'serving': 0},  # Right front [top right]
            3: {'attack': 0.3, 'defense': 0.7, 'serving': 0},  # Middle blocker [top middle]
            4: {'attack': 0.7, 'defense': 0.3, 'serving': 0},  # Outside hitter [top left]
            5: {'defense': 0.8, 'attack': 0.2, 'serving': 0},  # Left back [bottom left]
            6: {'defense': 0.7, 'attack': 0.3, 'serving': 0},  # Middle back [bottom middle]
        }
    

    #calculate the total team strength based on the current rotation
    def rotation_strength(self, players, rotation_state):
        """
        Calculate strength of current rotation configuration
        
        Parameters:
        -----------
        players : list of 6 Player objects
        rotation_state : int (0-5)
            Current rotation position (which player is in position 1)
            
        Returns:
        --------
        float : rotation strength score (1-5 scale)
        """
        total_strength = 0
        
        for position in range(1, 7):
            # Which player is in this position?
            player = players[(rotation_state + position - 1) % 6] #to loop through the 6 players
            
            # Get position weights
            weights = self.position_weights[position]
            
            # Calculate weighted contribution
            contribution = (weights['defense'] * player.defense +
                          weights['attack'] * player.attack +
                          weights['serving'] * player.serving)
            
            total_strength += contribution
        
        return total_strength / 6

    def lineup_violates_defense_constraint(self, players):
        """
        Check whether any rotation state causes too many low-defense players.

        Parameters:
        -----------
        players : list of 6 Player objects

        Returns:
        --------
        bool : True if the lineup violates the defense constraint
        """
        # If no constraint is active, treat as disabled
        if (self.min_defense_threshold <= 0.0 and self.max_low_defense >= 6 and not self.forbid_adjacent_low):
            return False

        for rot in range(6):
            # low flags for positions 1..6 (0..5)
            low = [players[(rot + pos - 1) % 6].defense < self.min_defense_threshold for pos in range(1, 7)]

            # 1) adjacency constraint: no two neighboring low-defense players
            if self.forbid_adjacent_low:
                for i in range(6):
                    if low[i] and low[(i + 1) % 6]:
                        return True

            # 2) overall low-count constraint
            if self.max_low_defense < 6:
                if sum(low) > self.max_low_defense:
                    return True

        return False
    
    #check we are penalising the serving team slightly #####################
    def point_win_probability(self, rotation_strength, serving=False):
        """
        Calculate probability of winning next point
        
        Parameters:
        -----------
        rotation_strength : float (1-5)
        serving : bool
            
        Returns:
        --------
        float : probability (0-1)
        """
        # Base probability from rotation strength
        base_prob = (rotation_strength - 1) / 4 #since max is 5 then (5-1)/4=1
        
        # Opponent adjustment
        opponent_adj = {'weak': 0.15, 'medium': 0.0, 'strong': -0.15}[self.opponent_strength] 
        #based on the opponent strength we modigy the probability to win the point
        
        # Serving adjustment
        serving_adj = -0.03 if serving else 0.03 #slight advantage to the serving team
        
        # Final probability
        prob = np.clip(base_prob + opponent_adj + serving_adj, 0.05, 0.95) #capped and floored
        
        return prob
    
    def simulate_set(self, players, n_rallies=100, starting_rotation=0):
        """
        Simulate a full set with given 6 players
        
        Returns:
        --------
        dict : simulation results
        """
        # Start serving in specified rotation
        rotation_state = starting_rotation
        serving = True
        
        points_us = 0
        points_them = 0
        
        rotation_time = np.zeros(6)  # Time spent in each rotation
        
        for _ in range(n_rallies):
            strength = self.rotation_strength(players, rotation_state) #for each rally we calculate the strength of the current rotation
            win_prob = self.point_win_probability(strength, serving) #for each rally -> get prob of winning the point
            
            rotation_time[rotation_state] += 1
            
            if np.random.random() < win_prob: #if random draw < then winning prob [we score the point]
                # We win the point
                points_us += 1
                if not serving: #default True at start
                    # Side-out: we get serve and rotate
                    serving = True #when we win the point while not serving we get the serve
                    rotation_state = (rotation_state + 1) % 6 #rotation state update
                #else: we keep serving, same rotation
            else:
                # They win the point
                points_them += 1
                if serving:
                    # We lose serve
                    serving = False
                # else: they keep serving, we don't rotate
        
        return {
            'points_us': points_us,
            'points_them': points_them,
            'score_diff': points_us - points_them,
            'rotation_time': rotation_time
        }
    
    def evaluate_lineup(self, player_indices, n_simulations=100):
        """
        Evaluate a specific 6-player lineup by trying different rotation orders
        
        Parameters:
        -----------
        player_indices : list of 6 indices into self.all_players
        
        Returns:
        --------
        dict : evaluation results
        """
        players = [self.all_players[i] for i in player_indices]

        # Enforce defense constraint: skip/eject lineups that would create
        # rotations where too many players have defense below threshold.
        if self.lineup_violates_defense_constraint(players):
            print(f"Skipping lineup {player_indices}: violates defense constraint (min_def={self.min_defense_threshold}, max_low={self.max_low_defense})")
            return {
                'player_indices': player_indices,
                'players': players,
                'best_strategy': {'strategy': 'Invalid - defense constraint', 'order': list(range(6)), 'avg_score': -999.0, 'std_score': 0.0},
                'all_strategies': []
            }
        
        # Try different strategic rotation orders
        results = []
        
        # Strategy 1: Natural order (as provided)
        scores = []
        for _ in range(n_simulations // 3):
            result = self.simulate_set(players, n_rallies=100) #default to not serving first
            scores.append(result['score_diff'])
        results.append({
            'strategy': 'Natural order',
            'order': list(range(6)),
            'avg_score': np.mean(scores),
            'std_score': np.std(scores)
        })
        
        # Strategy 2: Best servers first
        serve_order = sorted(range(6), key=lambda i: players[i].serving, reverse=True) #ordered by serving skill
        reordered_players = [players[i] for i in serve_order]
        scores = []
        for _ in range(n_simulations // 3):
            result = self.simulate_set(reordered_players, n_rallies=100)
            scores.append(result['score_diff'])
        results.append({
            'strategy': 'Best servers first',
            'order': serve_order,
            'avg_score': np.mean(scores),
            'std_score': np.std(scores)
        })
        
        # Strategy 3: Balanced (alternating strong/weak)
        overall_order = sorted(range(6), key=lambda i: players[i].overall, reverse=True)
        # Alternate: strong, weak, strong, weak, strong, weak
        balanced_order = [overall_order[0], overall_order[5], 
                         overall_order[1], overall_order[4],
                         overall_order[2], overall_order[3]]
        reordered_players = [players[i] for i in balanced_order]
        scores = []
        for _ in range(n_simulations // 3):
            result = self.simulate_set(reordered_players, n_rallies=50)
            scores.append(result['score_diff'])
        results.append({
            'strategy': 'Balanced alternating',
            'order': balanced_order,
            'avg_score': np.mean(scores),
            'std_score': np.std(scores)
        })
        
        # Return best strategy
        best = max(results, key=lambda x: x['avg_score'])
        
        return {
            'player_indices': player_indices,
            'players': players,
            'best_strategy': best,
            'all_strategies': results
        }
    
    def find_optimal_lineup_and_rotation(self, n_simulations=100):
        """
        Find optimal 6 players from 9 AND their rotation order
        
        This is the main optimization function that solves the complete problem:
        1. Which 6 players to play
        2. What rotation order
        
        Returns:
        --------
        dict : complete optimization results
        """
        print(f"\n{'='*70}")
        print(f"OPTIMIZING LINEUP SELECTION AND ROTATION ORDER")
        print(f"Opponent: {self.opponent_strength.upper()}")
        print(f"{'='*70}")
        
        # Step 1: Generate all possible 6-player combinations from 9
        all_combinations = list(combinations(range(9), 6))
        print(f"\nTotal possible lineups: {len(all_combinations)} (C(9,6) = 84)")
        
        # Step 2: Evaluate each combination
        lineup_results = []
        
        print(f"\nEvaluating lineups...")
        for i, lineup_indices in enumerate(all_combinations):
            if (i + 1) % 20 == 0:
                print(f"  Evaluated {i+1}/{len(all_combinations)} lineups...")
            
            result = self.evaluate_lineup(lineup_indices, n_simulations) #we are running through the different strategies players lineups
            lineup_results.append(result)
        
        # Step 3: Sort by performance
        lineup_results.sort(key=lambda x: x['best_strategy']['avg_score'], reverse=True)
        
        # Step 4: Identify best lineup
        best_lineup = lineup_results[0]
        best_players = best_lineup['players']
        best_strategy = best_lineup['best_strategy']
        bench_indices = [i for i in range(9) if i not in best_lineup['player_indices']]
        bench_players = [self.all_players[i] for i in bench_indices]
        
        print(f"\n{'='*70}")
        print(f"OPTIMAL LINEUP FOUND")
        print(f"{'='*70}")
        print(f"\nExpected Score Differential: {best_strategy['avg_score']:.2f} ± {best_strategy['std_score']:.2f}")
        print(f"Rotation Strategy: {best_strategy['strategy']}")
        
        print(f"\n--- STARTING 6 (ON COURT) ---")
        rotation_order = best_strategy['order']
        for i, order_idx in enumerate(rotation_order):
            player = best_players[order_idx]
            print(f"Position {i+1}: {player.name:20s} {player}")
        
        print(f"\n--- BENCH (3 PLAYERS) ---")
        for player in bench_players:
            print(f"  {player.name:20s} {player}")
        
        # Analyze why these players are benched
        print(f"\n--- BENCH ANALYSIS ---")
        court_avg = np.mean([p.overall for p in best_players])
        bench_avg = np.mean([p.overall for p in bench_players])
        print(f"Average ability on court: {court_avg:.2f}")
        print(f"Average ability on bench: {bench_avg:.2f}")
        print(f"Difference: {court_avg - bench_avg:.2f}")
        
        # Show attribute breakdown
        court_attrs = {
            'defense': np.mean([p.defense for p in best_players]),
            'attack': np.mean([p.attack for p in best_players]),
            'serving': np.mean([p.serving for p in best_players])
        }
        bench_attrs = {
            'defense': np.mean([p.defense for p in bench_players]),
            'attack': np.mean([p.attack for p in bench_players]),
            'serving': np.mean([p.serving for p in bench_players])
        }
        
        print(f"\nAttribute comparison:")
        print(f"  Defense - Court: {court_attrs['defense']:.2f}, Bench: {bench_attrs['defense']:.2f}")
        print(f"  Attack  - Court: {court_attrs['attack']:.2f}, Bench: {bench_attrs['attack']:.2f}")
        print(f"  Serving - Court: {court_attrs['serving']:.2f}, Bench: {bench_attrs['serving']:.2f}")
        
        return {
            'best_lineup': best_lineup,
            'all_lineups': lineup_results,
            'bench_players': bench_players,
            'bench_indices': bench_indices,
            'court_avg': court_avg,
            'bench_avg': bench_avg
        }
    
    def visualize_complete_analysis(self, results):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        best_lineup = results['best_lineup']
        best_players = best_lineup['players']
        bench_players = results['bench_players']
        all_lineups = results['all_lineups']
        
        # Plot 1: Top 15 lineup performances
        ax1 = fig.add_subplot(gs[0, :2])
        top_15 = all_lineups[:15]
        scores = [l['best_strategy']['avg_score'] for l in top_15]
        labels = [f"Lineup {i+1}" for i in range(15)]
        colors = ['green' if i == 0 else 'steelblue' for i in range(15)]
        
        ax1.barh(labels, scores, color=colors, alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Expected Score Differential')
        ax1.set_title('Top 15 Lineup Combinations')
        ax1.invert_yaxis()
        
        # Plot 2: Player attributes - Court vs Bench
        ax2 = fig.add_subplot(gs[0, 2])
        attributes = ['Defense', 'Attack', 'Serving', 'Overall']
        court_vals = [
            np.mean([p.defense for p in best_players]),
            np.mean([p.attack for p in best_players]),
            np.mean([p.serving for p in best_players]),
            np.mean([p.overall for p in best_players])
        ]
        bench_vals = [
            np.mean([p.defense for p in bench_players]),
            np.mean([p.attack for p in bench_players]),
            np.mean([p.serving for p in bench_players]),
            np.mean([p.overall for p in bench_players])
        ]
        
        x = np.arange(len(attributes))
        width = 0.35
        ax2.bar(x - width/2, court_vals, width, label='Court', color='green', alpha=0.7)
        ax2.bar(x + width/2, bench_vals, width, label='Bench', color='red', alpha=0.7)
        ax2.set_ylabel('Average Rating')
        ax2.set_title('Court vs Bench Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(attributes, rotation=45)
        ax2.legend()
        ax2.set_ylim(0, 5)
        
        # Plot 3: Individual player comparison (all 9)
        ax3 = fig.add_subplot(gs[1, :])
        player_names = [p.name for p in self.all_players]
        on_court = [i in best_lineup['player_indices'] for i in range(9)]
        
        defense_vals = [p.defense for p in self.all_players]
        attack_vals = [p.attack for p in self.all_players]
        serving_vals = [p.serving for p in self.all_players]
        
        x = np.arange(9)
        width = 0.25
        
        colors_d = ['green' if on_court[i] else 'lightcoral' for i in range(9)]
        colors_a = ['darkgreen' if on_court[i] else 'coral' for i in range(9)]
        colors_s = ['lime' if on_court[i] else 'salmon' for i in range(9)]
        
        ax3.bar(x - width, defense_vals, width, label='Defense', color=colors_d, alpha=0.8)
        ax3.bar(x, attack_vals, width, label='Attack', color=colors_a, alpha=0.8)
        ax3.bar(x + width, serving_vals, width, label='Serving', color=colors_s, alpha=0.8)
        
        ax3.set_ylabel('Rating')
        ax3.set_title('All 9 Players - Attributes (Green=Court, Red=Bench)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(player_names, rotation=0, ha='right')
        ax3.legend()
        ax3.axhline(y=3, color='gray', linestyle='--', alpha=0.3)
        ax3.set_ylim(0, 5.5)
        
        # Plot 4: Score distribution histogram
        ax4 = fig.add_subplot(gs[2, 0])
        all_scores = [l['best_strategy']['avg_score'] for l in all_lineups]
        ax4.hist(all_scores, bins=25, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.axvline(x=best_lineup['best_strategy']['avg_score'], 
                   color='red', linestyle='--', linewidth=2, label='Best')
        ax4.set_xlabel('Score Differential')
        ax4.set_ylabel('Number of Lineups')
        ax4.set_title('Distribution of Lineup Performance')
        ax4.legend()
        
        # Plot 5: Rotation strength by position
        ax5 = fig.add_subplot(gs[2, 1])
        rotation_strengths = []
        for rot in range(6):
            strength = self.rotation_strength(best_players, rot)
            rotation_strengths.append(strength)
        
        ax5.plot(range(1, 7), rotation_strengths, 'o-', linewidth=2, markersize=8, color='green')
        ax5.fill_between(range(1, 7), rotation_strengths, alpha=0.3, color='green')
        ax5.set_xlabel('Rotation State')
        ax5.set_ylabel('Rotation Strength')
        ax5.set_title('Strength Across Rotations')
        ax5.set_xticks(range(1, 7))
        ax5.axhline(y=np.mean(rotation_strengths), color='red', 
                   linestyle='--', label=f'Avg: {np.mean(rotation_strengths):.2f}')
        ax5.legend()
        ax5.set_ylim(min(rotation_strengths) - 0.2, max(rotation_strengths) + 0.2)
        
        # Plot 6: Strategy comparison
        ax6 = fig.add_subplot(gs[2, 2])
        strategies = best_lineup['all_strategies']
        strat_names = [s['strategy'] for s in strategies]
        strat_scores = [s['avg_score'] for s in strategies]
        
        ax6.barh(strat_names, strat_scores, color='orange', alpha=0.7)
        ax6.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Score Differential')
        ax6.set_title('Rotation Strategy Comparison')
        
        plt.suptitle(f'Complete Lineup Optimization Analysis - {self.opponent_strength.upper()} Opponent', 
                    fontsize=16, fontweight='bold')
        
        return fig

    def simulate_rally_progression(self, players, n_rallies=40, n_simulations=200):
        """Simulate many short sets and return average cumulative points_us per rally.

        Returns:
        - avg_series: array shape (n_rallies,) average points_us at each rally index
        - std_series: array shape (n_rallies,) std dev across simulations
        """
        all_series = np.zeros((n_simulations, n_rallies))

        for s in range(n_simulations):
            rotation_state = 0
            serving = True
            points_us = 0
            series = []
            for r in range(n_rallies):
                strength = self.rotation_strength(players, rotation_state)
                win_prob = self.point_win_probability(strength, serving)
                if np.random.random() < win_prob:
                    points_us += 1
                    if not serving:
                        serving = True
                        rotation_state = (rotation_state + 1) % 6
                else:
                    if serving:
                        serving = False
                all_series[s, r] = points_us

        avg_series = np.mean(all_series, axis=0)
        std_series = np.std(all_series, axis=0)
        return avg_series, std_series

#team details
def create_9_player_team():
    """Create a realistic 9-player team with varied abilities"""
    return [
        Player("Sofia", defense=4.5, attack=5.0, serving=5.0),      # Elite all-around
        Player("Giacomo", defense=4.0, attack=4.5, serving=4.0),        # Defensive specialist
        Player("Vlad", defense=3.5, attack=3.5, serving=3.5),    # Offensive star
        Player("Comez", defense=3.5, attack=4.0, serving=3.5),       # Defense-first
        Player("Michi", defense=3.0, attack=3.0, serving=3.0),  # Solid all-around
        Player("Mati", defense=3.5, attack=3.0, serving=3.0),    # Serving specialist
        Player("Vitto", defense=3.0, attack=3.0, serving=4.5),  # Balanced player
        Player("Albi", defense=3.5, attack=3.0, serving=3.5),    # Developing player
        Player("Ele", defense=2.0, attack=2.0, serving=4.0),       # Bench/reserve
    ]


def run_complete_9player_analysis():
    """Run complete analysis for 9-player team"""
    
    print("\n" + "="*70)
    print("9-PLAYER VOLLEYBALL LINEUP OPTIMIZATION")
    print("="*70)
    print("\nObjective: Select best 6 from 9 players AND determine rotation order")
    print("Constraint: 3 players must sit on bench (no substitutions)")
    print("="*70)
    
    # Create team
    players = create_9_player_team()
    
    print("\n--- FULL ROSTER (9 PLAYERS) ---")
    for i, player in enumerate(players):
        print(f"{i+1}. {player}")
    
    # Analyze against different opponents
    results_by_opponent = {}
    
    for opponent in ['weak', 'medium', 'strong']:
        print(f"\n\n{'='*70}")
        print(f"ANALYSIS VS {opponent.upper()} OPPONENT")
        print(f"{'='*70}")
        
        model = EnhancedRotationModel(players, opponent_strength=opponent)
        results = model.find_optimal_lineup_and_rotation(n_simulations=150)
        
        results_by_opponent[opponent] = results
        
        # Show top 3 alternative lineups
        print(f"\n--- TOP 3 ALTERNATIVE LINEUPS ---")
        for i, lineup in enumerate(results['all_lineups'][1:4], 2):
            player_names = [p.name for p in lineup['players']]
            score = lineup['best_strategy']['avg_score']
            print(f"\n{i}. Score: {score:.2f}")
            print(f"   Players: {', '.join(player_names)}")
            bench = [players[j].name for j in range(9) if j not in lineup['player_indices']]
            print(f"   Bench: {', '.join(bench)}")
        
        # Visualize
        print(f"\nGenerating comprehensive visualization...")
        fig = model.visualize_complete_analysis(results)
        safe_savefig(fig, os.path.join(OUTPUT_DIR, f'lineup_optimization_{opponent}.png'), 
                 dpi=300, bbox_inches='tight')
    
    # Comparative summary
    print(f"\n\n{'='*70}")
    print("COMPARATIVE SUMMARY ACROSS OPPONENTS")
    print(f"{'='*70}")
    
    for opponent in ['weak', 'medium', 'strong']:
        results = results_by_opponent[opponent]
        best = results['best_lineup']
        
        print(f"\n{opponent.upper()} OPPONENT:")
        print(f"  Expected score: {best['best_strategy']['avg_score']:.2f}")
        print(f"  Court: {', '.join([p.name for p in best['players']])}")
        print(f"  Bench: {', '.join([p.name for p in results['bench_players']])}")
    
    # Create comparative visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Performance vs opponent
    opponents = ['weak', 'medium', 'strong']
    scores = [results_by_opponent[o]['best_lineup']['best_strategy']['avg_score'] 
             for o in opponents]
    
    axes[0, 0].bar(opponents, scores, color=['green', 'orange', 'red'], alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 0].set_ylabel('Expected Score Differential')
    axes[0, 0].set_title('Optimal Performance vs Opponent Strength')
    for i, (opp, score) in enumerate(zip(opponents, scores)):
        axes[0, 0].text(i, score, f'{score:.1f}', ha='center', 
                       va='bottom' if score > 0 else 'top')
    
    # Plot 2: Bench consistency
    bench_counts = {}
    for player in players:
        bench_counts[player.name] = 0
    
    for opponent in opponents:
        for player in results_by_opponent[opponent]['bench_players']:
            bench_counts[player.name] += 1
    
    names = list(bench_counts.keys())
    counts = list(bench_counts.values())
    colors_bench = ['red' if c == 3 else 'orange' if c == 2 else 'yellow' if c == 1 else 'green' 
                   for c in counts]
    
    axes[0, 1].barh(names, counts, color=colors_bench, alpha=0.7)
    axes[0, 1].set_xlabel('Times Benched (out of 3 scenarios)')
    axes[0, 1].set_title('Player Bench Frequency')
    axes[0, 1].set_xlim(0, 3)
    
    # Plot 3: Attribute importance by opponent
    ax3 = axes[1, 0]
    opponent_labels = []
    court_defense = []
    court_attack = []
    court_serving = []
    
    for opp in opponents:
        opponent_labels.append(opp.capitalize())
        players_on_court = results_by_opponent[opp]['best_lineup']['players']
        court_defense.append(np.mean([p.defense for p in players_on_court]))
        court_attack.append(np.mean([p.attack for p in players_on_court]))
        court_serving.append(np.mean([p.serving for p in players_on_court]))
    
    x = np.arange(len(opponents))
    width = 0.25
    
    ax3.bar(x - width, court_defense, width, label='Defense', alpha=0.8)
    ax3.bar(x, court_attack, width, label='Attack', alpha=0.8)
    ax3.bar(x + width, court_serving, width, label='Serving', alpha=0.8)
    
    ax3.set_ylabel('Average Rating')
    ax3.set_title('Court Composition by Opponent')
    ax3.set_xticks(x)
    ax3.set_xticklabels(opponent_labels)
    ax3.legend()
    ax3.set_ylim(0, 5)
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    table_data.append(['Opponent', 'Court Avg', 'Bench Avg', 'Difference', 'Score'])
    
    for opp in opponents:
        results = results_by_opponent[opp]
        table_data.append([
            opp.capitalize(),
            f"{results['court_avg']:.2f}",
            f"{results['bench_avg']:.2f}",
            f"{results['court_avg'] - results['bench_avg']:.2f}",
            f"{results['best_lineup']['best_strategy']['avg_score']:.1f}"
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('9-Player Team Optimization Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # save the summary atomically
    safe_savefig(fig, os.path.join(OUTPUT_DIR, '9player_summary.png'), dpi=300, bbox_inches='tight')
    
    print("\n\nAll visualizations saved!")
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    return results_by_opponent


if __name__ == "__main__":
    # Show opponent menu and get user selection
    if OPPONENT_DATA_AVAILABLE:
        print("\n" + "="*80)
        print("OPPONENT SELECTION")
        print("="*80)
        opponent_db = print_opponent_menu()
        
        try:
            opponent_index = int(input("\nEnter opponent index (or press Enter to use generic 'medium'): ").strip() or "-1")
            if opponent_index >= 0:
                opp_name, opp_strength = get_opponent_by_index(opponent_index)
                if opp_name:
                    print(f"\nSelected: {opp_name} (Strength: {opp_strength})")
                    # Convert strength name to model strength
                    strength_map = {'WEAK': 'weak', 'MEDIUM': 'medium', 'STRONG': 'strong'}
                    selected_opponent_strength = strength_map.get(opp_strength, 'medium')
                else:
                    print("Invalid index. Using default 'medium'.")
                    selected_opponent_strength = 'medium'
            else:
                print("Using default 'medium' opponent strength.")
                selected_opponent_strength = 'medium'
        except ValueError:
            print("Invalid input. Using default 'medium' opponent strength.")
            selected_opponent_strength = 'medium'
    else:
        print("\nOpponent data not available. Using generic opponent strength.")
        selected_opponent_strength = 'medium'
    
    # Example: check constraint behavior before running full analysis
    players = create_9_player_team()

    # Configure constraints (tweak as needed)
    model = EnhancedRotationModel(players,
                                  opponent_strength=selected_opponent_strength,
                                  min_defense_threshold=3.5,
                                  max_low_defense=2,
                                  forbid_adjacent_low=True)

    print(f"\nChecking constraints: min_defense_threshold={model.min_defense_threshold}, ")
    print(f"max_low_defense={model.max_low_defense}, forbid_adjacent_low={model.forbid_adjacent_low}\n")

    all_combinations = list(combinations(range(9), 6))

    def check_lineup_constraints(indices):
        """Return (violates: bool, rot: int or None, reason: 'adjacent'|'count'|None, low_flags: list or None)"""
        six = [players[i] for i in indices]
        for rot in range(6):
            low = [six[(rot + pos - 1) % 6].defense < model.min_defense_threshold for pos in range(1, 7)]
            if model.forbid_adjacent_low:
                for i in range(6):
                    if low[i] and low[(i + 1) % 6]:
                        return True, rot, 'adjacent', low
            if model.max_low_defense < 6 and sum(low) > model.max_low_defense:
                return True, rot, 'count', low
        return False, None, None, None

    # Print diagnostics for first 10 lineups as example
    print("Sample lineup constraint diagnostics (first 10 lineups):")
    for indices in all_combinations[:10]:
        violates, rot, reason, low = check_lineup_constraints(indices)
        six = [players[i] for i in indices]
        lineup_names = ", ".join([f"{p.name}({p.defense:.1f})" for p in six])
        if not violates:
            print(f"  Lineup [{lineup_names}]: OK")
        else:
            # Show the rotation order where the violation occurs (player names + defense)
            rotated = [six[(rot + pos - 1) % 6] for pos in range(1, 7)]
            rotated_str = ", ".join([f"{p.name}({p.defense:.1f})" for p in rotated])
            print(f"  Lineup [{lineup_names}]: VIOLATES -> reason={reason}, rotation={rot}, rotation_order=[{rotated_str}]")

    # After quick checks, run the full analysis (unchanged behaviour)
    results = run_complete_9player_analysis()

    # --- additional chart: average score progression to 40 rallies for best lineup (based on selected opponent)
    print(f'\nGenerating progression chart (40 rallies) for best lineup vs {selected_opponent_strength.upper()} opponent...')
    best = results[selected_opponent_strength]['best_lineup']
    # best_players_all includes the 6 on-court players followed by the bench players
    best_players_all = best['players'] + results[selected_opponent_strength]['bench_players']
    # Keep the on-court six separate for simulation
    best_on_court = best['players']
    
    # Simulate ONE detailed 40-rally match to track rotation patterns
    model_for_progression = EnhancedRotationModel(best_on_court, opponent_strength=selected_opponent_strength)
    rotation_state = 0
    serving = True
    points_us = 0
    cumulative_points = []
    rotation_counts = np.zeros((len(best_on_court), 6))  # player x rotation_state
    
    for r in range(40):
        strength = model_for_progression.rotation_strength(best_on_court, rotation_state)
        win_prob = model_for_progression.point_win_probability(strength, serving)
        
        # Track which player is in which rotation state
        for pos in range(1, 7):
            player_idx = (rotation_state + pos - 1) % 6
            rotation_counts[player_idx, rotation_state] += 1
        
        # Simulate rally outcome
        if np.random.random() < win_prob:
            points_us += 1
            if not serving:
                serving = True
                rotation_state = (rotation_state + 1) % 6
        else:
            if serving:
                serving = False
        
        cumulative_points.append(points_us)
    
    # Simulate many runs for avg ± std (50 runs, faster than 500)
    avg_series, std_series = model_for_progression.simulate_rally_progression(best_on_court, n_rallies=40, n_simulations=100)

    # Build figure with 3 subplots
    fig2 = plt.figure(figsize=(16, 5))
    
    # Subplot 1: Cumulative points over 40 rallies
    ax1 = plt.subplot(1, 3, 1)
    x = np.arange(1, 41)
    ax1.plot(x, avg_series, color='green', linewidth=2, label='Avg points (us)')
    ax1.fill_between(x, avg_series - std_series, avg_series + std_series, color='green', alpha=0.2)
    ax1.set_xlabel('Rally')
    ax1.set_ylabel('Cumulative Points (us)')
    ax1.set_title(f'Cumulative Points Over 40 Rallies\n(vs {selected_opponent_strength.upper()})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Subplot 2: Bench duration for all 9 players (bench = full 40, on-court = 0)
    ax2 = plt.subplot(1, 3, 2)
    bench_times = []
    labels = []
    on_court_flags = []
    for p in players:
        labels.append(p.name)
        if p in best_players_all:
            if p in best_on_court:
                bench_times.append(0)
                on_court_flags.append(True)
            else:
                bench_times.append(40)
                on_court_flags.append(False)
        else:
            bench_times.append(40)
            on_court_flags.append(False)
    
    colors = ['green' if on else 'lightcoral' for on in on_court_flags]
    ax2.barh(labels, bench_times, color=colors)
    ax2.set_xlabel('Rallies (bench = 40, court = 0)')
    ax2.set_title('Bench Duration per Player\n(No Substitutions)')
    ax2.set_xlim(0, 45)
    
    # Subplot 3: Rotation time distribution for on-court players (how long in each rotation state)
    ax3 = plt.subplot(1, 3, 3)
    player_names_court = [p.name for p in best_on_court]
    rotation_state_names = [f'Rot {i+1}' for i in range(6)]
    
    # Normalize rotation counts to show distribution
    rotation_times_normalized = rotation_counts / rotation_counts.sum(axis=1, keepdims=True)
    
    x_pos = np.arange(len(best_on_court))
    bottom = np.zeros(len(best_on_court))
    colors_rot = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    for rot in range(6):
        ax3.bar(x_pos, rotation_times_normalized[:, rot], bottom=bottom, 
                label=rotation_state_names[rot], color=colors_rot[rot], alpha=0.8)
        bottom += rotation_times_normalized[:, rot]
    
    ax3.set_ylabel('Proportion of 40 Rallies')
    ax3.set_title('Rotation State Distribution\n(On-Court Players Only)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(player_names_court, rotation=45, ha='right')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_ylim(0, 1)

    plt.tight_layout()
    safe_savefig(fig2, os.path.join(OUTPUT_DIR, 'progression_40_best_lineup.png'), dpi=200, bbox_inches='tight')
    print('Saved progression_40_best_lineup.png to MathModel/')
    print(f"\nChart interpretation:")
    print(f"  - Left: Average cumulative points across 100 simulations (±1 std)")
    print(f"  - Middle: On-court players (green, 0 bench) vs bench players (red, 40 rallies benched)")
    print(f"  - Right: Rotation state time distribution for each on-court player")
    print(f"    (Good servers → fewer rotations → stay in same rotation longer)")
