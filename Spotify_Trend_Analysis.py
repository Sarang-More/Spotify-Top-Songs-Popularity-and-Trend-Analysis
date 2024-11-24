import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv('songs_data_with_mentions.csv')

# Step 2: Filter data for the specific song and extract the relevant columns (streaming counts for 16 weeks)
song_data = df[df['track_name'] == 'Timeless (with Playboi Carti)'][['Week_1', 'Week_2', 'Week_3', 'Week_4', 'Week_5', 
                                                'Week_6', 'Week_7', 'Week_8', 'Week_9', 'Week_10', 
                                                'Week_11', 'Week_12', 'Week_13', 'Week_14', 'Week_15', 
                                                'Week_16', 'popularity']].reset_index(drop=True)

# Step 3: Reshape the data into a time series format (1 column for streaming counts, 1 column for popularity)
streaming_counts = song_data.iloc[0, :-1].values  # Get the streaming counts (first row)
popularity = song_data['popularity'].iloc[0]  # Get the popularity value (same for all weeks in this example)

# Create a time series index (weeks 1-16)
weeks = range(1, 17)

# Step 4: Prepare the exogenous variable (popularity)
# Since popularity is constant, we'll just replicate it for each of the 16 weeks
popularity_data = [popularity] * 16

# Convert to DataFrame for easy handling
ts_data = pd.DataFrame({
    'week': weeks,
    'stream_count': streaming_counts,
    'popularity': popularity_data
})

# Step 5: Fit SARIMAX model (with popularity as exogenous variable)
# We will use (p, d, q) = (1, 1, 1) and seasonal order of (1, 1, 1, 4) since we have 4 quarters in a year or weekly seasonality
model = SARIMAX(ts_data['stream_count'], exog=ts_data['popularity'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
model_fit = model.fit(disp=False)

# Step 6: Forecast for the next 4 weeks
steps_ahead = 10

# For simplicity, let's assume the popularity stays the same for the next 4 weeks
future_popularity = [popularity] * steps_ahead  # Use the same popularity for forecast

# Forecast the next 4 weeks using the SARIMAX model
forecast = model_fit.forecast(steps=steps_ahead, exog=future_popularity)
print(f"Predicted stream counts for the next {steps_ahead} weeks: {forecast.tolist()}")

# Step 7: Plot historical and forecasted data
historical_index = ts_data.index
future_index = range(len(ts_data), len(ts_data) + steps_ahead)

plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(historical_index, ts_data['stream_count'], label='Historical Data', color='blue', linewidth=2)

# Plot forecasted data
plt.plot(future_index, forecast, label='Forecasted Data', color='red', linestyle='dashed', marker='o')

# Add a vertical line to separate historical and forecasted data
plt.axvline(x=len(ts_data)-1, color='gray', linestyle='dotted', linewidth=1.5, label='Forecast Start')

# Formatting the plot
plt.title('Stream Count: Historical vs. Forecasted Data')
plt.xlabel('Weeks')
plt.ylabel('Stream Count')
plt.legend()
plt.grid(True)
plt.show()

# Print the model summary for diagnostics
print(model_fit.summary())


# Melt the data to transform weeks into a long format
df_long = df.melt(
    id_vars=['track_name', 'artist_name'],  # Keeping track_name and artist_name
    value_vars=[f'Week_{i}' for i in range(1, 17)],  # Weeks 1 to 16
    var_name='Week', 
    value_name='Mentions'
)

# Extract numeric week number for sorting and analysis
df_long['Week_Number'] = df_long['Week'].str.extract('(\d+)').astype(int)

# Sort by Track Name and Week Number
df_long = df_long.sort_values(by=['track_name', 'Week_Number'])

# Plot for a specific song
song_name = 'Timeless (with Playboi Carti)'  # Replace with the desired song name
artist_name = 'The Weeknd'  # Replace with the artist's name

song_data = df_long[(df_long['track_name'] == song_name) & (df_long['artist_name'] == artist_name)]

sns.lineplot(data=song_data, x='Week_Number', y='Mentions', marker='o')
plt.title(f'Trend for {song_name} by {artist_name} Over 16 Weeks')
plt.xlabel('Week Number')
plt.ylabel('Social Media Mentions')
plt.grid(True)
plt.show()

# Select top N songs to compare, for example top 10 based on Weekly Mean
top_songs = df.groupby('track_name')['Weekly_Mean'].mean().nlargest(10).index

# Filter data for top songs
top_songs_data = df_long[df_long['track_name'].isin(top_songs)]

sns.lineplot(data=top_songs_data, x='Week_Number', y='Mentions', hue='track_name', marker='o')
plt.title('Mentions Trends for Top 10 Songs Over 16 Weeks')
plt.xlabel('Week Number')
plt.ylabel('Social Media Mentions')
plt.legend(title='Track Name', loc='upper left')
plt.grid(True)
plt.show()

# Calculate the mean mentions per week across all songs
weekly_trend = df_long.groupby('Week_Number')['Mentions'].mean()

# Plot the overall trend
plt.figure(figsize=(10, 6))
plt.plot(weekly_trend.index, weekly_trend.values, marker='o', linestyle='-', color='b')
plt.title('Average Mentions Across All Songs Over 16 Weeks')
plt.xlabel('Week Number')
plt.ylabel('Average Mentions')
plt.grid(True)
plt.show()

# Select Weeks 1, 4, 8, 12, and 16 for plotting
selected_weeks = ['Week_1', 'Week_4', 'Week_8', 'Week_12', 'Week_16']

# Melt the data to long format for easier plotting
df_long_selected = df.melt(
    id_vars=['track_name', 'artist_name'], 
    value_vars=selected_weeks, 
    var_name='Week', 
    value_name='Mentions'
)

# Extract numeric week numbers
df_long_selected['Week_Number'] = df_long_selected['Week'].str.extract('(\d+)').astype(int)

# Group by week to calculate the mean mentions across songs (or aggregate however desired)
weekly_mentions = df_long_selected.groupby('Week_Number')['Mentions'].mean().reset_index()

# Plotting the trend for Weeks 1, 4, 8, 12, and 16
plt.figure(figsize=(10, 6))
plt.plot(weekly_mentions['Week_Number'], weekly_mentions['Mentions'], marker='o', label='Average Mentions')
plt.title('Mentions Trend for Weeks 1, 4, 8, 12, 16')
plt.xlabel('Week Number')
plt.ylabel('Average Mentions')
plt.xticks(weekly_mentions['Week_Number'])  # Ensure correct week numbers on x-axis
plt.grid(alpha=0.5)
plt.legend()
plt.show()


# Drop columns for Week 1-16 and Weekly Mean
columns_to_drop = ['Week_1', 'Week_2', 'Week_3', 'Week_4', 'Week_5', 'Week_6', 'Week_7', 'Week_8', 'Week_9', 'Week_10', 
                  'Week_11', 'Week_12', 'Week_13', 'Week_14', 'Week_15', 'Week_16', 'Weekly_Mean']

df_filtered = df.drop(columns=columns_to_drop)

# Plot the correlation matrix for the remaining columns
plt.figure(figsize=(10, 8))
sns.heatmap(df_filtered.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Assuming 'df' is your DataFrame and 'social_media_mentions' is the target variable
X = df[['danceability', 'energy', 'loudness', 'stream_count', 'Weekly_Mean', 'popularity']]  # Selected features
y = df['social_media_mentions']  # Target variable

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Check for multicollinearity using VIF (Variance Inflation Factor)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print("Variance Inflation Factors (VIF):")
print(vif_data)

# If any features have a high VIF (>5 or 10), you may consider removing them

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on both training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate on training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Evaluate on test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)


# Calculate and display other evaluation metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")

# Residuals plot
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

# Actual vs Predicted plot
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()


# Pairplot for selected features
sns.pairplot(df[['popularity', 'danceability', 'energy', 'loudness']])
plt.title("Pair Plot of Key Features")
plt.show()

# Create an interactive 3D scatter plot
fig = px.scatter_3d(
    df, 
    x='popularity', 
    y='stream_count', 
    z='social_media_mentions', 
    color='popularity', 
    size='stream_count',  # Optional: size points by stream count
    color_continuous_scale='Viridis',
    title='Interactive 3D Scatter Plot: Popularity vs. Stream Count vs. Song Mentions'
)

# Customize axes and interactivity
fig.update_layout(
    scene=dict(
        xaxis_title='Popularity',
        yaxis_title='Stream Count',
        zaxis_title='Song Mentions'
    ),
    coloraxis_colorbar=dict(title='Popularity'),
)

fig.show()

