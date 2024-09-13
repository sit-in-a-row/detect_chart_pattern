import numpy as np
import pandas as pd
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler

def create_pattern(n_bars=100, pattern_type='ascending_triangle'):
    # Set up the initial values with variability for the upper resistance and lower support

    if pattern_type in ['ascending_triangle', 'ascending_wedge']:
        upper_start = np.random.uniform(55, 65)  # Randomize starting point of the upper resistance
        lower_start = np.random.uniform(35, 45)  # Randomize starting point of the lower support
        if pattern_type == 'ascending_triangle':
            bias_slope_1 = np.random.uniform(-10, 10)
            bias_slope_2 = np.random.uniform(-10, 10)
            upper_end = np.random.uniform(55 + bias_slope_1, 65 + bias_slope_2)  # Converging upper resistance
            lower_end = upper_end  # Both converge at the same point
        elif pattern_type == 'ascending_wedge':
            upper_end = upper_start  # Horizontal upper resistance
            lower_end = upper_end  # Both converge at the same point
        
        upper_slope = np.linspace(upper_start, upper_end, n_bars)
        lower_slope = np.linspace(lower_start, lower_end, n_bars)
        
    elif pattern_type in ['descending_triangle', 'descending_wedge']:
        upper_start = np.random.uniform(55, 65)  # Randomize starting point of the upper resistance
        lower_start = np.random.uniform(35, 45)  # Randomize starting point of the lower support
        if pattern_type == 'descending_triangle':
            bias_slope_1 = np.random.uniform(-10, 0)  # Decreasing upper slope
            bias_slope_2 = np.random.uniform(0, 5)   # Flat or slightly decreasing lower slope
            lower_end = np.random.uniform(35 + bias_slope_1 , 45 + bias_slope_2 )  # Lower end close to lower start
            upper_end = lower_end  # Ensure upper line is above lower and decreasing
        elif pattern_type == 'descending_wedge':
            lower_end = lower_start + np.random.uniform(-0.1, 0.1)  # Slightly decreasing or flat lower support
            upper_end = lower_end  # Ensure upper line is above lower and decreasing
            
        upper_slope = np.linspace(upper_start, upper_end, n_bars)
        lower_slope = np.linspace(lower_start, lower_end, n_bars)
        
    elif pattern_type == 'double_top':
        peak_price = np.random.uniform(55, 65)
        trough_price = np.random.uniform(35, 45)
        quarter_bars = n_bars // 4
        upper_slope = np.concatenate([
            np.linspace(trough_price, peak_price, quarter_bars),
            np.linspace(peak_price, trough_price, quarter_bars),
            np.linspace(trough_price, peak_price, quarter_bars),
            np.linspace(peak_price, trough_price, quarter_bars)
        ])
        lower_slope = np.full_like(upper_slope, trough_price)  # 길이를 upper_slope에 맞추기
        
    elif pattern_type == 'double_bottom':
        trough_price = np.random.uniform(35, 45)
        peak_price = np.random.uniform(55, 65)
        quarter_bars = n_bars // 4
        lower_slope = np.concatenate([
            np.linspace(peak_price, trough_price, quarter_bars),
            np.linspace(trough_price, peak_price, quarter_bars),
            np.linspace(peak_price, trough_price, quarter_bars),
            np.linspace(trough_price, peak_price, quarter_bars)
        ])
        upper_slope = np.full_like(lower_slope, peak_price)  # 길이를 lower_slope에 맞추기

    if pattern_type not in ['double_top', 'double_bottom']:
        # Adding randomness to make the lines less predictable
        upper_slope += np.random.uniform(-0.5, 0.5, n_bars)
        lower_slope += np.random.uniform(-0.5, 0.5, n_bars)

    # 길이가 n_bars보다 짧은 경우에 맞추기
    if len(upper_slope) < n_bars:
        upper_slope = np.pad(upper_slope, (0, n_bars - len(upper_slope)), 'edge')
        lower_slope = np.pad(lower_slope, (0, n_bars - len(lower_slope)), 'edge')

    # Initialize price data
    prices = [np.random.uniform(lower_start, upper_start) if pattern_type not in ['double_top', 'double_bottom'] else lower_slope[0]]
    direction = 1 if np.random.rand() > 0.5 else -1  # Random initial direction

    # Generate the price data that oscillates between the upper and lower slopes
    for i in range(1, n_bars):
        local_trend = np.random.uniform(-4, 4)  # Small local trend factor
        price = prices[-1] + direction * (np.random.uniform(0.5, 1.5) + local_trend)
        
        # Check for slope boundaries and reverse direction if needed
        if price >= upper_slope[i]:
            price = upper_slope[i]  # Snap to upper line
            direction = -1  # Reverse direction to downward
        elif price <= lower_slope[i]:
            price = lower_slope[i]  # Snap to lower line
            direction = 1  # Reverse direction to upward
            
        prices.append(price)

    # Create the OHLC data
    open_prices = prices[:-1] + np.random.uniform(-0.5, 0.5, n_bars - 1)
    close_prices = prices[1:] + np.random.uniform(-0.5, 0.5, n_bars - 1)
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 1, n_bars - 1)
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 1, n_bars - 1)
    volume = np.random.randint(1000, 5000, n_bars - 1)

    # MinMaxScaler 적용
    scaler = MinMaxScaler()
    scaled_ohlc = scaler.fit_transform(np.column_stack([open_prices, high_prices, low_prices, close_prices]))

    # Prepare the DataFrame
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=n_bars-1),
        'Open': scaled_ohlc[:, 0],
        'High': scaled_ohlc[:, 1],
        'Low': scaled_ohlc[:, 2],
        'Close': scaled_ohlc[:, 3],
        'Volume': volume
    }
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)

    return df, upper_slope[1:], lower_slope[1:]

def get_pattern_graph(target_df, upper_slope, lower_slope):
    '''
    Visualize Created 
    '''
    apds = [
        mpf.make_addplot(upper_slope, color='red', linestyle='--', width=1),
        mpf.make_addplot(lower_slope, color='green', linestyle='--', width=1)
    ]
    mpf.plot(target_df, type='candle', volume=True, title='Triangle Pattern with Noise Among Trends', style='charles', addplot=apds)



# def get_pattern_graph(pattern_type):
#     '''
#     Visualize Created 
#     '''
#     target_df, upper_slope, lower_slope = generate_pattern(n_bars=100, pattern_type=pattern_type)

#     # Plot the generated stock data with mplfinance and add the upper and lower slopes
#     apds = [
#         mpf.make_addplot(upper_slope, color='red', linestyle='--', width=1),
#         mpf.make_addplot(lower_slope, color='green', linestyle='--', width=1)
#     ]

#     mpf.plot(target_df, type='candle', volume=True, title='Triangle Pattern with Noise Among Trends', style='charles', addplot=apds)