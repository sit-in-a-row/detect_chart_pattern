from generate_pattern import generate_pattern, get_pattern_graph

import mplfinance as mpf
import numpy as np
import os

def store_datasets(pattern, generation_count, show_graph):
    '''
    Generate directory to store datasets

    ========<Pattern List>========

    - ascending_trinagle
    - ascending_wedge
    - descending_trinagle
    - descending_wedge
    - double_top
    - double_bottom

    ==============================
    '''
    save_path_parent = f'../store_dataset/{pattern}'

    for i in range(1, generation_count+1):
    
        # Initialize save path
        save_path = f'{save_path_parent}/{pattern}_{i}.csv'
        # Create directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Randomly choose the number of candles in chart pattern
        n_bars = int(np.random.uniform(80, 120))

        # Call generate_pattern() function
        df, upper_slope, lower_slope = generate_pattern(n_bars, pattern)

        # Save data as csv file
        df.to_csv(save_path)

        print(f'{i}/{generation_count} Done')

        if show_graph:
            get_pattern_graph(df, upper_slope, lower_slope)