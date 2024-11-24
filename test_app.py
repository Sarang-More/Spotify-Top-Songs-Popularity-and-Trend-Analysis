import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Streamlit App Title
st.title("Spotify Top Songs Popularity and Trend Analysis")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Step 2: Filter data for a specific song
    song_options = df['track_name'].unique()
    song_name = st.selectbox("Select Song Name:", song_options)
    song_data = df[df['track_name'] == song_name][['Week_1', 'Week_2', 'Week_3', 'Week_4', 'Week_5', 
                                                   'Week_6', 'Week_7', 'Week_8', 'Week_9', 'Week_10', 
                                                   'Week_11', 'Week_12', 'Week_13', 'Week_14', 'Week_15', 
                                                   'Week_16', 'popularity']].reset_index(drop=True)
    
    if song_data.empty:
        st.warning("Song not found in the dataset.")
    else:
        st.write("Selected Song Data:", song_data)

        # Step 3: Reshape data into time series format
        streaming_counts = song_data.iloc[0, :-1].values  # Get streaming counts
        popularity = song_data['popularity'].iloc[0]  # Get popularity
        weeks = range(1, 17)

        ts_data = pd.DataFrame({
            'week': weeks,
            'stream_count': streaming_counts,
            'popularity': [popularity] * 16
        })
        st.write("Time Series Data:", ts_data)

        # Step 4: Fit SARIMAX model
        st.write("Fitting SARIMAX model...")
        model = SARIMAX(ts_data['stream_count'], exog=ts_data['popularity'], 
                        order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
        model_fit = model.fit(disp=False)

        # Step 5: Forecasting
        steps_ahead = st.slider("Forecast Steps Ahead:", 1, 12, 4)
        future_popularity = [popularity] * steps_ahead
        forecast = model_fit.forecast(steps=steps_ahead, exog=future_popularity)
        
        st.write(f"Predicted stream counts for the next {steps_ahead} weeks:", forecast.tolist())
        
        # Step 6: Plot results
        st.write("Historical and Forecast Data:")
        future_index = range(len(ts_data), len(ts_data) + steps_ahead)

        # Step 7: Plot historical and forecasted data
        st.write("Historical vs. Forecasted Data:")
        
        plt.figure(figsize=(12, 6))
        # Plot historical data
        plt.plot(ts_data.index, ts_data['stream_count'], label='Historical Data', color='blue', linewidth=2)

        # Plot forecasted data
        plt.plot(future_index, forecast, label='Forecasted Data', color='red', linestyle='dashed', marker='o')

        # Add a vertical line to separate historical and forecasted data
        plt.axvline(x=len(ts_data) - 1, color='gray', linestyle='dotted', linewidth=1.5, label='Forecast Start')

        # Formatting the plot
        plt.title('Stream Count: Historical vs. Forecasted Data')
        plt.xlabel('Weeks')
        plt.ylabel('Stream Count')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Step 8: Display SARIMAX model summary
        st.write("SARIMAX Model Summary:")
        st.text(model_fit.summary())
        
        top_songs = df.nlargest(10, 'popularity')[['track_name', 'popularity']]
        st.write("Top 10 Songs by Popularity:", top_songs)
        artist_options = df['artist_name'].unique()
        selected_artist = st.selectbox("Select an Artist:", artist_options)
        artist_data = df[df['artist_name'] == selected_artist]
        st.write(f"Data for {selected_artist}:", artist_data)

        # Step 9: Transform weeks into long format
        st.write("Data Transformation: Weeks to Long Format")
        df_long = df.melt(
            id_vars=['track_name', 'artist_name'], 
            value_vars=[f'Week_{i}' for i in range(1, 17)],
            var_name='Week', 
            value_name='Mentions'
        )

        # Extract numeric week number
        df_long['Week_Number'] = df_long['Week'].str.extract('(\d+)').astype(int)

        # Sort data
        df_long = df_long.sort_values(by=['track_name', 'Week_Number'])
        st.write("Transformed Data (Long Format):", df_long.head())

        # Step 10: Plot trend for a specific song
        st.write("Mentions Trend for Specific Song")
        song_name = st.text_input("Enter Song Name for Mentions Trend:", "Timeless (with Playboi Carti)")
        artist_name = st.text_input("Enter Artist Name:", "The Weeknd")

        song_data = df_long[(df_long['track_name'] == song_name) & (df_long['artist_name'] == artist_name)]

        if song_data.empty:
            st.warning("No data found for the specified song and artist.")
        else:
            st.write(f"Mentions Trend Data for {song_name} by {artist_name}:", song_data)

            # Plotting
            import seaborn as sns
            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=song_data, x='Week_Number', y='Mentions', marker='o')
            plt.title(f'Trend for {song_name} by {artist_name} Over 16 Weeks')
            plt.xlabel('Week Number')
            plt.ylabel('Social Media Mentions')
            plt.grid(True)
            st.pyplot(plt)

        # Step 11: Compare Mentions Trends for Top N Songs
        st.write("Mentions Trends for Top N Songs")

        # Select top N songs based on Weekly Mean
        if 'Weekly_Mean' in df.columns:
            top_n = st.slider("Select Number of Top Songs to Compare:", 1, 20, 10)
            top_songs = df.groupby('track_name')['Weekly_Mean'].mean().nlargest(top_n).index

            # Filter data for top songs
            top_songs_data = df_long[df_long['track_name'].isin(top_songs)]
            st.write(f"Data for Top {top_n} Songs:", top_songs_data)

            # Plot trends for top songs
            sns.set_style("whitegrid")
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=top_songs_data, x='Week_Number', y='Mentions', hue='track_name', marker='o')
            plt.title(f'Mentions Trends for Top {top_n} Songs Over 16 Weeks')
            plt.xlabel('Week Number')
            plt.ylabel('Social Media Mentions')
            plt.legend(title='Track Name', loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.warning("Weekly_Mean column is missing in the dataset.")

        # Step 12: Plot Overall Weekly Trend
        st.write("Average Mentions Trend Across All Songs")

        weekly_trend = df_long.groupby('Week_Number')['Mentions'].mean()
        st.write("Weekly Trend Data:", weekly_trend)

        # Plot overall trend
        plt.figure(figsize=(10, 6))
        plt.plot(weekly_trend.index, weekly_trend.values, marker='o', linestyle='-', color='b')
        plt.title('Average Mentions Across All Songs Over 16 Weeks')
        plt.xlabel('Week Number')
        plt.ylabel('Average Mentions')
        plt.grid(True)
        st.pyplot(plt)

        # Step 13: Mentions Trend for Selected Weeks
        st.write("Mentions Trend for Selected Weeks")

        # Select weeks for analysis
        selected_weeks = ['Week_1', 'Week_4', 'Week_8', 'Week_12', 'Week_16']

        # Transform data for selected weeks
        df_long_selected = df.melt(
            id_vars=['track_name', 'artist_name'], 
            value_vars=selected_weeks, 
            var_name='Week', 
            value_name='Mentions'
        )

        # Extract numeric week numbers
        df_long_selected['Week_Number'] = df_long_selected['Week'].str.extract('(\d+)').astype(int)

        # Calculate mean mentions per selected week
        weekly_mentions = df_long_selected.groupby('Week_Number')['Mentions'].mean().reset_index()
        st.write("Selected Weeks Trend Data:", weekly_mentions)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(weekly_mentions['Week_Number'], weekly_mentions['Mentions'], marker='o', label='Average Mentions')
        plt.title('Mentions Trend for Weeks 1, 4, 8, 12, 16')
        plt.xlabel('Week Number')
        plt.ylabel('Average Mentions')
        plt.xticks(weekly_mentions['Week_Number'])  # Correct week numbers on x-axis
        plt.grid(alpha=0.5)
        plt.legend()
        st.pyplot(plt)


    # Drop unnecessary columns
    columns_to_drop = ['Week_1', 'Week_2', 'Week_3', 'Week_4', 'Week_5', 'Week_6', 'Week_7', 'Week_8', 'Week_9', 'Week_10', 
                      'Week_11', 'Week_12', 'Week_13', 'Week_14', 'Week_15', 'Week_16', 'Weekly_Mean']
    
    df_filtered = df.drop(columns=columns_to_drop)
    df_numeric = df_filtered.select_dtypes(include=[np.number])
    # Plot Correlation Matrix
    st.write("Correlation Matrix:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

    # Prepare data for regression
    X = df[['danceability', 'energy', 'loudness', 'stream_count', 'Weekly_Mean', 'popularity']]
    y = df['social_media_mentions']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check for multicollinearity
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    st.write("Variance Inflation Factors (VIF):", vif_data)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions and Evaluation Metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)

    st.write(f"MAE: {mae:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    st.write(f"MSE: {test_mse:.2f}")

    # Residual Distribution Plot for Streamlit
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='skyblue', bins=20, ax=ax)
    ax.set_title("Residual Distribution", fontsize=14)
    ax.set_xlabel("social_media_mentions", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    st.pyplot(fig)

    # Actual vs Predicted Plot with Line and Scatter Points for Streamlit
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')  # Scatter points
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit')  # Line
    ax.set_title("Actual vs Predicted Song Mentions", fontsize=14)
    ax.set_xlabel("Actual Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.legend()
    st.pyplot(fig)

    # Pairplot for selected features
    st.write("Pair Plot of Key Features:")
    sns.pairplot(df[['popularity', 'danceability', 'energy', 'loudness']])
    plt.title("Pair Plot of Key Features")
    st.pyplot(plt)

    # Interactive 3D Plot
    st.write("Interactive 3D Scatter Plot:")
    fig = px.scatter_3d(
        df, 
        x='popularity', 
        y='stream_count', 
        z='social_media_mentions', 
        color='popularity', 
        size='stream_count',
        color_continuous_scale='Viridis',
        title='Interactive 3D Scatter Plot: Popularity vs. Stream Count vs. Song Mentions'
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='Popularity',
            yaxis_title='Stream Count',
            zaxis_title='Social Media Mentions'
        ),
        coloraxis_colorbar=dict(title='Popularity'),
    )

    st.plotly_chart(fig)
    

    




