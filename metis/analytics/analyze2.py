import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression
import ruptures  # pip install ruptures
from datetime import datetime, timedelta

path = '/Users/btsznh/Downloads/data2.csv'
# read csv file
df = pd.read_csv(path)

# column 1 name
df.columns.values[0] = 'Item'
df.columns.values[1] = 'Speed'
df.columns.values[2] = 'Accuracy'

def advanced_time_series_analysis(df):
    """
    Perform advanced time series analysis using Prophet
    """
    from prophet import Prophet
    import pandas as pd
    import numpy as np
    
    # Convert Accuracy to numeric, handling percentage strings if present
    if df['Accuracy'].dtype == 'object':
        df['Accuracy'] = df['Accuracy'].str.rstrip('%').astype(float) / 100
    elif df['Accuracy'].dtype in ['int64', 'float64']:
        if df['Accuracy'].mean() > 1:
            df['Accuracy'] = df['Accuracy'] / 100
    
    # Create datetime index starting from today
    base_date = datetime.now()
    dates = [base_date + timedelta(days=x) for x in range(len(df))]
    
    # Prepare data for Prophet
    df_speed = pd.DataFrame({
        'ds': dates,
        'y': df['Speed']
    })
    
    df_accuracy = pd.DataFrame({
        'ds': dates,
        'y': df['Accuracy']
    })
    
    # Initialize and fit Prophet models
    speed_model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    
    accuracy_model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    
    speed_model.fit(df_speed)
    accuracy_model.fit(df_accuracy)
    
    # Make future dataframe for predictions
    future_periods = int(len(df) * 0.2)  # Predict 20% more into the future
    future = speed_model.make_future_dataframe(periods=future_periods)
    
    # Get forecasts
    speed_forecast = speed_model.predict(future)
    accuracy_forecast = accuracy_model.predict(future)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Speed Plot
    ax1.scatter(df['Item'], df['Speed'], color='blue', alpha=0.5, label='Actual Speed')
    ax1.plot(speed_forecast['ds'], speed_forecast['yhat'], color='red', 
             label='Predicted Speed', linewidth=2)
    ax1.fill_between(speed_forecast['ds'],
                     speed_forecast['yhat_lower'],
                     speed_forecast['yhat_upper'],
                     color='red', alpha=0.1, label='Confidence Interval')
    ax1.set_title('Speed Analysis and Forecast')
    ax1.set_xlabel('Item (Time)')
    ax1.set_ylabel('Speed')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy Plot
    ax2.scatter(df['Item'], df_accuracy['y'], color='green', alpha=0.5, 
                label='Actual Accuracy')
    ax2.plot(accuracy_forecast['ds'], accuracy_forecast['yhat'], color='red',
             label='Predicted Accuracy', linewidth=2)
    ax2.fill_between(accuracy_forecast['ds'],
                     accuracy_forecast['yhat_lower'],
                     accuracy_forecast['yhat_upper'],
                     color='red', alpha=0.1, label='Confidence Interval')
    ax2.set_title('Accuracy Analysis and Forecast')
    ax2.set_xlabel('Item (Time)')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print Analysis Results
    print("\nTime Series Analysis Results:")
    
    # Speed Analysis
    print("\nSpeed Analysis:")
    print(f"Current Speed Trend: {speed_forecast['trend'].iloc[-1]:.2f}")
    print(f"Forecasted Speed (next period): {speed_forecast['yhat'].iloc[-1]:.2f}")
    print(f"Speed Forecast Range: {speed_forecast['yhat_lower'].iloc[-1]:.2f} to {speed_forecast['yhat_upper'].iloc[-1]:.2f}")
    
    # Accuracy Analysis
    print("\nAccuracy Analysis:")
    print(f"Current Accuracy Trend: {accuracy_forecast['trend'].iloc[-1]:.4f}")
    print(f"Forecasted Accuracy (next period): {accuracy_forecast['yhat'].iloc[-1]:.4f}")
    print(f"Accuracy Forecast Range: {accuracy_forecast['yhat_lower'].iloc[-1]:.4f} to {accuracy_forecast['yhat_upper'].iloc[-1]:.4f}")
    
    # Detect Change Points
    speed_changepoints = speed_model.changepoints
    accuracy_changepoints = accuracy_model.changepoints
    
    print("\nSignificant Change Points:")
    print("\nSpeed Change Points:")
    print(speed_changepoints.tolist())
    print("\nAccuracy Change Points:")
    print(accuracy_changepoints.tolist())
    
    return {
        'speed_forecast': speed_forecast,
        'accuracy_forecast': accuracy_forecast,
        'speed_changepoints': speed_changepoints,
        'accuracy_changepoints': accuracy_changepoints
    }


advanced_time_series_analysis(df)