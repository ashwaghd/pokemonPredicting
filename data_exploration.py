#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# Load the datasets
print("Loading Pokemon data...")
pokemon_df = pd.read_csv("archive/pokemon.csv")
combats_df = pd.read_csv("archive/combats.csv")

print("\n====== POKEMON DATASET EXPLORATION ======")
print(f"Pokemon dataset shape: {pokemon_df.shape}")
print("\nPokemon dataset info:")
print(pokemon_df.info())

print("\nPokemon dataset sample:")
print(pokemon_df.head())

print("\nPokemon dataset statistics:")
print(pokemon_df.describe())

print("\nChecking for missing values in Pokemon dataset:")
print(pokemon_df.isnull().sum())

# Check for potential data issues
print("\nChecking if any Pokemon names are missing:")
missing_names = pokemon_df[pokemon_df['Name'].isnull()]
print(f"Number of Pokemon with missing names: {len(missing_names)}")

# If there are any empty names, we should investigate
if len(missing_names) > 0:
    print("Pokemon with missing names:")
    print(missing_names)

# Check unique Pokemon types
print("\nUnique Pokemon Type 1 values:")
print(pokemon_df['Type 1'].unique())
print(f"Number of unique Type 1 values: {pokemon_df['Type 1'].nunique()}")

print("\nUnique Pokemon Type 2 values:")
print(pokemon_df['Type 2'].unique())
print(f"Number of unique Type 2 values: {pokemon_df['Type 2'].nunique()}")

# Plot distribution of Pokemon types
plt.figure(figsize=(12, 6))
type1_counts = pokemon_df['Type 1'].value_counts()
sns.barplot(x=type1_counts.index, y=type1_counts.values)
plt.title('Distribution of Pokemon Type 1')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('plots/pokemon_type1_distribution.png')

# Plot distributions of Pokemon stats
stats_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
plt.figure(figsize=(15, 10))
for i, col in enumerate(stats_cols):
    plt.subplot(2, 3, i+1)
    sns.histplot(pokemon_df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('plots/pokemon_stats_distributions.png')

# Check correlation between stats
plt.figure(figsize=(10, 8))
numeric_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
corr_matrix = pokemon_df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Between Pokemon Stats')
plt.tight_layout()
plt.savefig('plots/pokemon_stats_correlation.png')

# Legendary Pokemon analysis
legendary_count = pokemon_df['Legendary'].value_counts()
print("\nLegendary Pokemon distribution:")
print(legendary_count)

# Compare stats between legendary and non-legendary Pokemon
plt.figure(figsize=(15, 10))
for i, col in enumerate(stats_cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='Legendary', y=col, data=pokemon_df)
    plt.title(f'{col} by Legendary Status')
plt.tight_layout()
plt.savefig('plots/legendary_vs_normal_stats.png')

print("\n====== COMBATS DATASET EXPLORATION ======")
print(f"Combats dataset shape: {combats_df.shape}")
print("\nCombats dataset info:")
print(combats_df.info())

print("\nCombats dataset sample:")
print(combats_df.head())

print("\nChecking for missing values in Combats dataset:")
print(combats_df.isnull().sum())

# Check how many unique Pokemon are in combat dataset
unique_pokemon_in_combats = set(combats_df['First_pokemon'].unique()) | set(combats_df['Second_pokemon'].unique())
print(f"\nNumber of unique Pokemon in combats: {len(unique_pokemon_in_combats)}")

# Check if all Pokemon in combats exist in the Pokemon dataset
pokemon_ids = set(pokemon_df.iloc[:, 0].values)  # Assuming first column is Pokemon ID
missing_pokemon = unique_pokemon_in_combats - pokemon_ids
print(f"Number of Pokemon IDs in combats that don't exist in Pokemon dataset: {len(missing_pokemon)}")
if len(missing_pokemon) > 0:
    print(f"Missing Pokemon IDs: {missing_pokemon}")

# Analyze win rates by Pokemon
win_counts = combats_df['Winner'].value_counts()
battle_counts = pd.concat([combats_df['First_pokemon'], combats_df['Second_pokemon']]).value_counts()

# Create a DataFrame to analyze win rates
win_rates = pd.DataFrame({
    'Pokemon_ID': win_counts.index,
    'Wins': win_counts.values,
    'Battles': battle_counts[win_counts.index].values
})
win_rates['Win_Rate'] = win_rates['Wins'] / win_rates['Battles']

# Sort by win rate for Pokemon with at least 10 battles
frequent_battlers = win_rates[win_rates['Battles'] >= 10].sort_values('Win_Rate', ascending=False)
print("\nTop 10 Pokemon by win rate (min 10 battles):")
print(frequent_battlers.head(10))

# Plot win rates for top 20 frequent battlers
plt.figure(figsize=(12, 6))
top_battlers = frequent_battlers.head(20)
sns.barplot(x='Pokemon_ID', y='Win_Rate', data=top_battlers)
plt.title('Win Rates for Top 20 Pokemon (min 10 battles)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('plots/top_pokemon_win_rates.png')

# Now let's prepare the data for machine learning
print("\n====== DATA PREPARATION ======")

# Merge combat data with Pokemon stats
def prepare_battle_features(battles_df, pokemon_df):
    # Create copies to avoid modifying the original DataFrames
    battles = battles_df.copy()
    pokemon = pokemon_df.copy()
    
    # Ensure pokemon ID is the index
    pokemon.set_index(pokemon.columns[0], inplace=True)
    
    # Create feature columns for first Pokemon
    first_pokemon_stats = pokemon.loc[battles['First_pokemon']].reset_index()
    for col in first_pokemon_stats.columns:
        if col != 'index':  # Skip the ID column
            battles[f'First_{col}'] = first_pokemon_stats[col].values
    
    # Create feature columns for second Pokemon
    second_pokemon_stats = pokemon.loc[battles['Second_pokemon']].reset_index()
    for col in second_pokemon_stats.columns:
        if col != 'index':  # Skip the ID column
            battles[f'Second_{col}'] = second_pokemon_stats[col].values
    
    # Create target column (1 if first Pokemon won, 0 if second Pokemon won)
    battles['FirstWon'] = (battles['Winner'] == battles['First_pokemon']).astype(int)
    
    return battles

# Prepare the battle features
print("Preparing battle features...")
battles_with_features = prepare_battle_features(combats_df, pokemon_df)

print("\nBattle features dataset shape:", battles_with_features.shape)
print("\nBattle features sample:")
print(battles_with_features.head())

# Handle categorical features (Pokemon types)
print("\nEncoding categorical features...")
# One-hot encode the Pokemon types
battles_encoded = pd.get_dummies(
    battles_with_features, 
    columns=['First_Type 1', 'First_Type 2', 'Second_Type 1', 'Second_Type 2'],
    drop_first=False
)

# Define features and target
X = battles_encoded.drop(['First_pokemon', 'Second_pokemon', 'Winner', 'FirstWon', 'First_Name', 'Second_Name'], axis=1)
y = battles_encoded['FirstWon']

print("\nFeatures shape after encoding:", X.shape)

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\nTrain set size: {X_train.shape[0]} ({X_train.shape[0]/len(X):.2%})")
print(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0]/len(X):.2%})")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X):.2%})")

# Scale numerical features
print("\nScaling numerical features...")
# Identify numerical columns (excluding one-hot encoded columns)
numeric_cols = [col for col in X.columns if not col.startswith(('First_Type', 'Second_Type'))]

# Initialize scaler and fit on training data
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("\nData preparation complete!")

# Save prepared datasets
print("\nSaving prepared datasets...")
# Save the scaler
import joblib
joblib.dump(scaler, 'pokemon_scaler.joblib')

# Save the processed datasets
X_train.to_csv('pokemon_X_train.csv', index=False)
y_train.to_csv('pokemon_y_train.csv', index=False)
X_val.to_csv('pokemon_X_val.csv', index=False)
y_val.to_csv('pokemon_y_val.csv', index=False)
X_test.to_csv('pokemon_X_test.csv', index=False)
y_test.to_csv('pokemon_y_test.csv', index=False)

print("\nDatasets saved successfully!")

# Print additional insights for the report
print("\n====== ADDITIONAL INSIGHTS FOR REPORT ======")
print(f"Total number of Pokemon: {len(pokemon_df)}")
print(f"Total number of battles: {len(combats_df)}")
print(f"Number of features after preparation: {X.shape[1]}")
print(f"Class balance (percentage of first Pokemon winning): {y.mean():.2%}")

# Look at the feature importance using a simple model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 most important features according to Random Forest:")
print(feature_importances.head(10))

# Save feature importances plot
plt.figure(figsize=(12, 8))
top_features = feature_importances.head(20)
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')

print("\nData analysis complete! Plots saved in the 'plots' directory.") 