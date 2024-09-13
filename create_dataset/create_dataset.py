from .create_pattern import create_pattern, get_pattern_graph
from .save_datasets import save_datasets

pattern_list = ['ascending_triangle', 'ascending_wedge', 'descending_triangle', 'descending_wedge', 'double_top', 'double_bottom']

def create_dataset(generation_count, n_min, n_max):
    for pattern in pattern_list:
        print(f'===== Creating dataset: {pattern} =====')
        save_datasets(pattern, generation_count, n_min, n_max)