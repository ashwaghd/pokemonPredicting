#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Pokemon battle rankings')
    
    parser.add_argument('--combats-file', type=str, default='archive/combats.csv',
                        help='Path to the combats dataset')
    parser.add_argument('--pokemon-file', type=str, default='archive/pokemon.csv',
                        help='Path to the Pokemon dataset')
    parser.add_argument('--min-battles', type=int, default=10,
                        help='Minimum number of battles required for ranking')
    parser.add_argument('--output-dir', type=str, default='rankings',
                        help='Directory to save output files')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print(f"Loading combat data from {args.combats_file}...")
    combats_df = pd.read_csv(args.combats_file)
    
    print(f"Loading Pokemon data from {args.pokemon_file}...")
    pokemon_df = pd.read_csv(args.pokemon_file)
    
    # Calculate win counts for each Pokemon
    winners = combats_df['Winner'].value_counts().reset_index()
    winners.columns = ['Pokemon_ID', 'Wins']
    
    # Calculate total battle counts
    first_pokemon = combats_df['First_pokemon'].value_counts().reset_index()
    first_pokemon.columns = ['Pokemon_ID', 'First_Battles']
    
    second_pokemon = combats_df['Second_pokemon'].value_counts().reset_index()
    second_pokemon.columns = ['Pokemon_ID', 'Second_Battles']
    
    # Merge battle counts
    battle_counts = pd.merge(first_pokemon, second_pokemon, on='Pokemon_ID', how='outer').fillna(0)
    battle_counts['Total_Battles'] = battle_counts['First_Battles'] + battle_counts['Second_Battles']
    
    # Merge with win counts
    pokemon_stats = pd.merge(battle_counts, winners, on='Pokemon_ID', how='left').fillna(0)
    
    # Calculate win rates
    pokemon_stats['Win_Rate'] = pokemon_stats['Wins'] / pokemon_stats['Total_Battles']
    
    # Filter by minimum battles
    pokemon_stats = pokemon_stats[pokemon_stats['Total_Battles'] >= args.min_battles]
    
    # Sort by win rate
    pokemon_stats = pokemon_stats.sort_values('Win_Rate', ascending=False)
    
    # Add Pokemon names and types
    pokemon_info = pokemon_df[['#', 'Name', 'Type 1', 'Type 2']]
    pokemon_info.columns = ['Pokemon_ID', 'Name', 'Type_1', 'Type_2']
    
    pokemon_rankings = pd.merge(pokemon_stats, pokemon_info, on='Pokemon_ID', how='left')
    
    # Save full rankings
    rankings_file = f"{args.output_dir}/pokemon_full_rankings.csv"
    pokemon_rankings.to_csv(rankings_file, index=False)
    print(f"Full rankings saved to {rankings_file}")
    
    # Display top 20 Pokemon
    print(f"\nTop 20 Pokemon by Win Rate (min. battles: {args.min_battles}):")
    top20 = pokemon_rankings.head(20)
    display_cols = ['Pokemon_ID', 'Name', 'Type_1', 'Type_2', 'Win_Rate', 'Wins', 'Total_Battles']
    print(top20[display_cols].to_string(index=False))
    
    # Create visualizations
    
    # Win rate distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(pokemon_rankings['Win_Rate'], bins=20, kde=True)
    plt.title('Distribution of Pokemon Win Rates')
    plt.xlabel('Win Rate')
    plt.ylabel('Number of Pokemon')
    plt.savefig(f"{args.output_dir}/win_rate_distribution.png")
    
    # Top 20 win rates
    plt.figure(figsize=(12, 8))
    top20_plot = top20.sort_values('Win_Rate')
    sns.barplot(x='Win_Rate', y='Name', data=top20_plot)
    plt.title('Top 20 Pokemon by Win Rate')
    plt.xlabel('Win Rate')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/top20_win_rates.png")
    
    # Win rates by type
    type_stats = pokemon_rankings.groupby('Type_1').agg({
        'Win_Rate': 'mean',
        'Pokemon_ID': 'count'
    }).reset_index()
    type_stats.columns = ['Type', 'Avg_Win_Rate', 'Count']
    type_stats = type_stats[type_stats['Count'] >= 5]  # Only types with at least 5 Pokemon
    type_stats = type_stats.sort_values('Avg_Win_Rate', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Type', y='Avg_Win_Rate', data=type_stats)
    plt.title('Average Win Rate by Pokemon Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/type_win_rates.png")
    
    # Save type stats
    type_stats.to_csv(f"{args.output_dir}/type_performance.csv", index=False)
    
    print(f"\nAdditional stats and visualizations saved to {args.output_dir}/")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 