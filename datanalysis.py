import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression
import ruptures  # pip install ruptures

path = '/Users/btsznh/Downloads/data.csv'
# read csv file
df = pd.read_csv(path)

# column 1 name
df.columns.values[0] = 'Item'
df.columns.values[1] = 'Speed'
df.columns.values[2] = 'Accuracy'

def linear_analysis(df):
    # Convert Accuracy from percentage string to float if needed
    if df['Accuracy'].dtype == 'object':
        df['Accuracy'] = df['Accuracy'].str.rstrip('%').astype(float) / 100

    # analysis - 1 (speed trend)
    slope_speed, intercept_speed, r_value_speed, p_value_speed, std_err_speed = stats.linregress(df['Item'], df['Speed'])
    df['Predicted_Speed'] = slope_speed * df['Item'] + intercept_speed

    # Print Speed Analysis Results
    print("\nSpeed Trend Analysis:")
    print(f"Rate of change in Speed: {slope_speed:.2f} units per item")
    print(f"Starting Speed (intercept): {intercept_speed:.2f}")
    print(f"R-squared value: {r_value_speed**2:.4f}")
    print(f"P-value: {p_value_speed:.4f}")

    # analysis - 2 (accuracy trend)
    slope_accuracy, intercept_accuracy, r_value_accuracy, p_value_accuracy, std_err_accuracy = stats.linregress(df['Item'], df['Accuracy'])
    df['Predicted_Accuracy'] = slope_accuracy * df['Item'] + intercept_accuracy

    # Print Accuracy Analysis Results
    print("\nAccuracy Trend Analysis:")
    print(f"Rate of change in Accuracy: {slope_accuracy:.4f} per item")
    print(f"Starting Accuracy (intercept): {intercept_accuracy:.4f}")
    print(f"R-squared value: {r_value_accuracy**2:.4f}")
    print(f"P-value: {p_value_accuracy:.4f}")

    # Visualize both trends
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Speed trend plot
    ax1.scatter(df['Item'], df['Speed'], color='blue', label='Actual Speed')
    ax1.plot(df['Item'], df['Predicted_Speed'], color='red', label='Speed Trend')
    ax1.set_xlabel('Item (Time)')
    ax1.set_ylabel('Speed')
    ax1.set_title('Speed Trend Over Time')
    ax1.legend()
    ax1.grid(True)

    # Accuracy trend plot
    ax2.scatter(df['Item'], df['Accuracy'], color='green', label='Actual Accuracy')
    ax2.plot(df['Item'], df['Predicted_Accuracy'], color='red', label='Accuracy Trend')
    ax2.set_xlabel('Item (Time)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Trend Over Time')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Save results to file
    df.to_csv('analysis_results.csv', index=False)

def non_linear_analysis(df):
    import numpy as np
    from scipy.optimize import curve_fit
    
    # Define the polynomial fitting function first
    def fit_polynomial(x, y, degree=2):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(x.reshape(-1, 1))
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, y)
        return poly_reg, poly
    
    # 2. Exponential fit function
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    # 3. Logarithmic fit function
    def log_func(x, a, b, c):
        return a * np.log(b * x) + c
    
    # Prepare data
    x = df['Item'].values
    y_speed = df['Speed'].values
    y_accuracy = df['Accuracy'].values if 'Accuracy' in df else None
    
    # Polynomial analysis for Speed
    poly_reg_speed, poly_features = fit_polynomial(x, y_speed, degree=2)
    X_poly_speed = poly_features.fit_transform(x.reshape(-1, 1))
    y_pred_poly_speed = poly_reg_speed.predict(X_poly_speed)
    
    # Exponential fit for Speed
    try:
        popt_exp_speed, _ = curve_fit(exp_func, x, y_speed, p0=[1, 0.1, 1])
        y_pred_exp_speed = exp_func(x, *popt_exp_speed)
    except:
        print("Exponential fit failed for Speed")
        y_pred_exp_speed = None
    
    # Logarithmic fit for Speed
    try:
        popt_log_speed, _ = curve_fit(log_func, x, y_speed, p0=[1, 1, 1])
        y_pred_log_speed = log_func(x, *popt_log_speed)
    except:
        print("Logarithmic fit failed for Speed")
        y_pred_log_speed = None
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot original data
    plt.scatter(x, y_speed, color='blue', label='Actual Speed')
    
    # Plot polynomial fit
    plt.plot(x, y_pred_poly_speed, 'r-', label=f'Polynomial (degree={2})')
    
    # Plot exponential fit
    if y_pred_exp_speed is not None:
        plt.plot(x, y_pred_exp_speed, 'g-', label='Exponential')
    
    # Plot logarithmic fit
    if y_pred_log_speed is not None:
        plt.plot(x, y_pred_log_speed, 'y-', label='Logarithmic')
    
    plt.xlabel('Item (Time)')
    plt.ylabel('Speed')
    plt.title('Non-linear Analysis of Speed Trends')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate R-squared for each model
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    # Print results
    print("\nNon-linear Analysis Results for Speed:")
    print(f"Polynomial R-squared: {r2_score(y_speed, y_pred_poly_speed):.4f}")
    if y_pred_exp_speed is not None:
        print(f"Exponential R-squared: {r2_score(y_speed, y_pred_exp_speed):.4f}")
    if y_pred_log_speed is not None:
        print(f"Logarithmic R-squared: {r2_score(y_speed, y_pred_log_speed):.4f}")
    
    # Add predictions to dataframe
    df['Poly_Speed'] = y_pred_poly_speed
    if y_pred_exp_speed is not None:
        df['Exp_Speed'] = y_pred_exp_speed
    if y_pred_log_speed is not None:
        df['Log_Speed'] = y_pred_log_speed
    
    return df

def detailed_segmented_analysis(df):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    import ruptures
    from scipy import stats
    
    def fit_segment(x, y):
        """Fit a linear regression to a segment and return detailed metrics"""
        model = LinearRegression()
        X = x.reshape(-1, 1)
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        r_squared = model.score(X, y)
        
        # Calculate additional statistics
        mean_speed = np.mean(y)
        std_speed = np.std(y)
        
        # Perform t-test to check if slope is significantly different from 0
        n = len(x)
        slope_se = np.sqrt(np.sum((y - (slope * x + intercept))**2) / (n-2) / np.sum((x - np.mean(x))**2))
        t_stat = slope / slope_se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'mean_speed': mean_speed,
            'std_speed': std_speed,
            'p_value': p_value
        }
    
    # Prepare data
    x = df['Item'].values
    y = df['Speed'].values
    
    # First level segmentation (major phases)
    algo = ruptures.Binseg(model="l2").fit(y.reshape(-1, 1))
    major_change_points = algo.predict(n_bkps=2)
    
    # For the stable phase, perform more detailed analysis
    segments = []
    start_idx = 0
    
    for end_idx in major_change_points:
        segment_x = x[start_idx:end_idx]
        segment_y = y[start_idx:end_idx]
        
        metrics = fit_segment(segment_x, segment_y)
        
        # If this is the stable phase (third segment), do additional analysis
        if len(segments) == 2:  # We're on the third segment
            # Look for sub-segments within the stable phase
            sub_algo = ruptures.Binseg(model="l2").fit(segment_y.reshape(-1, 1))
            sub_change_points = sub_algo.predict(n_bkps=3)  # Try to find sub-phases
            
            # Analyze each sub-segment
            sub_start = 0
            for sub_end in sub_change_points:
                if sub_end > sub_start:
                    sub_x = segment_x[sub_start:sub_end]
                    sub_y = segment_y[sub_start:sub_end]
                    sub_metrics = fit_segment(sub_x, sub_y)
                    
                    segments.append({
                        'start_item': sub_x[0],
                        'end_item': sub_x[-1],
                        'type': 'sub_segment',
                        **sub_metrics
                    })
                sub_start = sub_end
        else:
            segments.append({
                'start_item': segment_x[0],
                'end_item': segment_x[-1],
                'type': 'major_segment',
                **metrics
            })
        
        start_idx = end_idx
    
    # Visualization with enhanced detail
    plt.figure(figsize=(15, 8))
    plt.scatter(x, y, color='blue', alpha=0.5, label='Actual Speed')
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for i, segment in enumerate(segments):
        segment_x = np.array([segment['start_item'], segment['end_item']])
        segment_y = segment['slope'] * segment_x + segment['intercept']
        
        line_style = '--' if segment['type'] == 'major_segment' else ':'
        plt.plot(segment_x, segment_y, line_style, color=colors[i % len(colors)],
                label=f"Segment {i+1} (slope: {segment['slope']:.2f})")
        
    plt.xlabel('Item (Time)')
    plt.ylabel('Speed')
    plt.title('Detailed Segmented Regression Analysis of Speed')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print detailed analysis
    print("\nDetailed Segment Analysis:")
    for i, segment in enumerate(segments):
        print(f"\nSegment {i+1} ({segment['type']}):")
        print(f"Items {segment['start_item']:.0f}-{segment['end_item']:.0f}")
        print(f"Slope: {segment['slope']:.3f} units per item (p={segment['p_value']:.4f})")
        print(f"Mean Speed: {segment['mean_speed']:.2f} ± {segment['std_speed']:.2f}")
        print(f"R-squared: {segment['r_squared']:.4f}")
        
        # Interpret the trend
        if segment['p_value'] < 0.05:
            trend = "significantly increasing" if segment['slope'] > 0 else "significantly decreasing"
        else:
            trend = "stable"
        print(f"Trend: {trend}")
    
    return segments

def speed_accuracy_analysis(df):
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    
    # Convert Accuracy from percentage string to float if needed
    if df['Accuracy'].dtype == 'object':
        df['Accuracy'] = df['Accuracy'].str.rstrip('%').astype(float) / 100
    
    # 1. Correlation Analysis
    correlation, p_value = stats.pearsonr(df['Speed'], df['Accuracy'])
    
    # 2. Linear Regression
    X = df['Speed'].values.reshape(-1, 1)
    y = df['Accuracy'].values
    reg = LinearRegression().fit(X, y)
    r_squared = reg.score(X, y)
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.scatter(df['Speed'], df['Accuracy'], alpha=0.5)
    
    # Regression line
    speed_range = np.linspace(df['Speed'].min(), df['Speed'].max(), 100)
    plt.plot(speed_range, reg.predict(speed_range.reshape(-1, 1)), 'r-', 
             label=f'Regression Line (slope: {reg.coef_[0]:.4f})')
    
    plt.xlabel('Speed')
    plt.ylabel('Accuracy')
    plt.title('Speed-Accuracy Trade-off Analysis')
    plt.legend()
    plt.grid(True)
    
    # Print analysis results
    print("\nSpeed-Accuracy Relationship Analysis:")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Slope: {reg.coef_[0]:.4f}")
    print(f"Intercept: {reg.intercept_:.4f}")
    
    # Optional: Calculate optimal performance points
    optimal_points = df[df['Accuracy'] > df['Accuracy'].mean()][['Speed', 'Accuracy']]
    print("\nOptimal Performance Points (Above Average Accuracy):")
    print(f"Average Speed at high accuracy: {optimal_points['Speed'].mean():.2f}")
    print(f"Average Accuracy at these points: {optimal_points['Accuracy'].mean():.4f}")
    
    # Show the plot
    plt.show()
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'r_squared': r_squared,
        'slope': reg.coef_[0],
        'intercept': reg.intercept_
    }

# Run the analysis
#detailed_segments = detailed_segmented_analysis(df)
relationship_stats = speed_accuracy_analysis(df)