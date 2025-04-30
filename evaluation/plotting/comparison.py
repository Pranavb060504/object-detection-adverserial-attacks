import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import re

map_faster_rcnn = {
    'baseline': 0.373,
    'fgsm_epsilon_0.01': 0.252,
    'fgsm_epsilon_0.05': 0.175,
    'fgsm_epsilon_0.1': 0.115,
    'dpattack_epsilon_0.3_iterations_10': 0.106,
    'dpattack_epsilon_0.3_iterations_20': 0.100,
    'dpattack_epsilon_1.0_iterations_10': 0.091,
    'dpattack_epsilon_1.0_iterations_20': 0.085,
    'pgd_epsilon_0.031_alpha_0.0078_iterations_10': 0.331,
    'pgd_epsilon_0.031_alpha_0.0078_iterations_20': 0.307,
    'pgd_epsilon_0.062_alpha_0.0156_iterations_10': 0.180,
    'pgd_epsilon_0.062_alpha_0.0156_iterations_20': 0.147,
    'edge_attack_epsilon_0.3_iterations_10': 0.249,
    'edge_attack_epsilon_0.3_iterations_20': 0.247,
    'edge_attack_epsilon_1.0_iterations_10': 0.273,
    'edge_attack_epsilon_1.0_iterations_20': 0.260,
}

mar_faster_rcnn = {
    'baseline': 0.445,
    'fgsm_epsilon_0.01': 0.300,
    'fgsm_epsilon_0.05': 0.221,
    'fgsm_epsilon_0.1': 0.146,
    'dpattack_epsilon_0.3_iterations_10': 0.111,
    'dpattack_epsilon_0.3_iterations_20': 0.106,
    'dpattack_epsilon_1.0_iterations_10': 0.098,
    'dpattack_epsilon_1.0_iterations_20': 0.097,
    'pgd_epsilon_0.031_alpha_0.0078_iterations_10': 0.437,
    'pgd_epsilon_0.031_alpha_0.0078_iterations_20': 0.404,
    'pgd_epsilon_0.062_alpha_0.0156_iterations_10': 0.322,
    'pgd_epsilon_0.062_alpha_0.0156_iterations_20': 0.291,
    'edge_attack_epsilon_0.3_iterations_10': 0.270,
    'edge_attack_epsilon_0.3_iterations_20': 0.280,
    'edge_attack_epsilon_1.0_iterations_10': 0.313,
    'edge_attack_epsilon_1.0_iterations_20': 0.291,
}

map_yolo = {
    'baseline': 0.392,
    'fgsm_epsilon_0.01': 0.015,
    'fgsm_epsilon_0.05': 0.009,
    'fgsm_epsilon_0.1': 0.004,
    'dpattack_epsilon_0.3_iterations_10': 0.031,
    'dpattack_epsilon_0.3_iterations_20': 0.032,
    'dpattack_epsilon_1.0_iterations_10': 0.022,
    'dpattack_epsilon_1.0_iterations_20': 0.030,
    'pgd_epsilon_0.031_alpha_0.0078_iterations_10': 0.014,
    'pgd_epsilon_0.031_alpha_0.0078_iterations_20': 0.014,
    'pgd_epsilon_0.062_alpha_0.0156_iterations_10': 0.010,
    'pgd_epsilon_0.062_alpha_0.0156_iterations_20': 0.009,
    'edge_attack_epsilon_0.3_iterations_10': 0.042,
    'edge_attack_epsilon_0.3_iterations_20': 0.044,
    'edge_attack_epsilon_1.0_iterations_10': 0.040,
    'edge_attack_epsilon_1.0_iterations_20': 0.037,
}

mar_yolo = {
    'baseline': 0.444,
    'fgsm_epsilon_0.01': 0.040,
    'fgsm_epsilon_0.05': 0.020,
    'fgsm_epsilon_0.1': 0.009,
    'dpattack_epsilon_0.3_iterations_10': 0.035,
    'dpattack_epsilon_0.3_iterations_20': 0.036,
    'dpattack_epsilon_1.0_iterations_10': 0.025,
    'dpattack_epsilon_1.0_iterations_20': 0.036,
    'pgd_epsilon_0.031_alpha_0.0078_iterations_10': 0.038,
    'pgd_epsilon_0.031_alpha_0.0078_iterations_20': 0.038,
    'pgd_epsilon_0.062_alpha_0.0156_iterations_10': 0.026,
    'pgd_epsilon_0.062_alpha_0.0156_iterations_20': 0.022,
    'edge_attack_epsilon_0.3_iterations_10': 0.051,
    'edge_attack_epsilon_0.3_iterations_20': 0.052,
    'edge_attack_epsilon_1.0_iterations_10': 0.049,
    'edge_attack_epsilon_1.0_iterations_20': 0.046,
}


name_map = {
    'baseline': 'Baseline',
    'fgsm': 'FGSM',
    'dpattack': 'DPAttack',
    'pgd': 'PGD',
    'edge_attack': 'Edge-DPAttack'
}

def xtick_map(label):
    epsilon_values = ['0.01', '0.05', '0.1', '0.3', '0.31', '0.62', '1.0']
    alpha_values = ['0.0078', '0.0156']
    iteration_values = ['10', '20']

    epsilon_label = ''
    for epsilon in epsilon_values:
        if epsilon in label:
            epsilon_label += f'$\epsilon={epsilon}$'
            break

    alpha_label = ''
    for alpha in alpha_values:
        if alpha in label:
            alpha_label += f'$\\alpha={alpha}$'
            break

    t_label = ''
    for t in iteration_values:
        if t in label:
            t_label += f'$T={t}$'
            break
    
    if epsilon_label and t_label:
        if alpha_label:
            xtick = epsilon_label + ', ' + alpha_label + ', ' + t_label
        else:
            xtick = epsilon_label + ', ' + t_label
            if 'attack' not in label:
                xtick += '.'
    else:
        xtick = epsilon_label + alpha_label + t_label
    return xtick

attack_palette = {
    'Baseline': '#000000',
    'FGSM': '#1f77b4',
    'DPAttack': '#ff7f0e',
    'PGD': '#d62728',
    'Edge-DPAttack': '#2ca02c',
}

result_dicts = [map_faster_rcnn, mar_faster_rcnn, map_yolo, mar_yolo]
titles = [
    'Mean Average Precision (mAP) on Faster RCNN',
    'Mean Average Recall (mAR) on Faster RCNN',
    'Mean Average Precision (mAP) on YOLO v5s',
    'Mean Average Recall (mAR) on YOLO v5s',
]
filenames = [
    'map_faster_rcnn.png',
    'mar_faster_rcnn.png',
    'map_yolo.png',
    'mar_yolo.png',
]

for result_dict, title, filename in zip(result_dicts, titles, filenames):
    df = pd.DataFrame([
        {
            'Attack': name_map[k.split('_epsilon')[0].split('_iterations')[0]] if k != 'baseline' else 'Baseline',
            'Attack Configuration': xtick_map(k.replace(f"{k.split('_')[0]}_", '')) if k != 'baseline' else 'Baseline',
            'mAP': v
        }
        for k, v in result_dict.items()
    ])
    plt.figure(layout='constrained')
    sns.barplot(
        df,
        x='Attack Configuration',
        y='mAP',
        hue='Attack',
        palette=attack_palette
    )
    plt.axhline(y=result_dict['baseline'], linestyle='--', c='black')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc=(1.01, 0.45), title='Attack')
    plt.title(title)
    plt.savefig(f'plots/{filename}')
    # plt.show()