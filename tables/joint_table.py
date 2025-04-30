import os
import json

# open all files in ./results/fgsm/epsilon_0.1

M0 = 'baseline'
M1 = 'dpattack @ epsilon = 0.3 iters = 20 gridesize = 3'
M2 = 'dpattack @ epsilon = 0.3 iters = 10 gridesize = 3'
M3 = 'dpattack @ epsilon = 1.0 iters = 10 gridesize = 3' 
M4 = 'dpattack @ epsilon = 1.0 iters = 20 gridesize = 3'
M5 = 'fgsm @ epsilon = 0.01'
M6 = 'fgsm @ epsilon = 0.05'
M7 = 'fgsm @ epsilon = 0.1'
M8 = 'edge_attack @ epsilon = 0.3 iters = 20 gridesize = 3'
M9 = 'edge_attack @ epsilon = 0.3 iters = 10 gridesize = 3'
M10 = 'edge_attack @ epsilon = 1.0 iters = 20 gridesize = 3'
M11 = 'edge_attack @ epsilon = 1.0 iters = 10 gridesize = 3'
M12 = 'pgdattack @ epsilon = 0.031 alpha = 0.007843 iters = 10'
M13 = 'pgdattack @ epsilon = 0.031 alpha = 0.007843 iters = 20'
M14 = 'pgdattack @ epsilon = 0.062 alpha = 0.015686 iters = 10'
M15 = 'pgdattack @ epsilon = 0.062 alpha = 0.015686 iters = 20'

M = [M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11]

def get_files(path):
    """
    Get all files in a directory
    """
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

files = get_files('../results')

# exclude files in the directory ../results/time
files = [file for file in files if 'time' not in file]

def get_table(model, all_data):
    data = all_data[model]
    table = f'''
\\begin{{table*}}[h]
    \\centering
    \\resizebox{{1\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{lccc|ccccccccccc}}
        \\toprule
        Metric & IoU & Area & Baseline & 
        \\makecell{{dpattack \\\\ $\\epsilon=0.3$, iters=20, \\\\ grid=3}} & 
        \\makecell{{dpattack \\\\ $\\epsilon=0.3$, iters=10, \\\\ grid=3}} & 
        \\makecell{{dpattack \\\\ $\\epsilon=1.0$, iters=10, \\\\ grid=3}} & 
        \\makecell{{dpattack \\\\ $\\epsilon=1.0$, iters=20, \\\\ grid=3}} & 
        \\makecell{{fgsm \\\\ $\\epsilon=0.05$}} & 
        \\makecell{{fgsm \\\\ $\\epsilon=0.1$}} & 
        \\makecell{{fgsm \\\\ $\\epsilon=0.01$}} &
        \\makecell{{edge attack \\\\ $\\epsilon=0.3$, iters=20, \\\\ grid=3}} & 
        \\makecell{{edge attack \\\\ $\\epsilon=0.3$, iters=10, \\\\ grid=3}} & 
        \\makecell{{edge attack \\\\ $\\epsilon=1.0$, iters=20, \\\\ grid=3}} & 
        \\makecell{{edge attack \\\\ $\\epsilon=1.0$, iters=10, \\\\ grid=3}} &
        \\makecell{{pgdattack \\\\ $\\epsilon=0.31$, \\\\ alpha=0.007843, \\\\ iters=10}} &
        \\makecell{{pgdattack \\\\ $\\epsilon=0.31$, \\\\ alpha=0.007843, \\\\ iters=20}} &
        \\makecell{{pgdattack \\\\ $\\epsilon=0.62$, \\\\ alpha=0.015686, \\\\ iters=10}} &
        \\makecell{{pgdattack \\\\ $\\epsilon=0.62$, \\\\ alpha=0.015686, \\\\ iters=20}} \\\\
        \\midrule
        \\textbf{{Average Precision (AP)}} & 0.50:0.95 & all & {data[M0][0]} & {data[M1][0]} & {data[M2][0]} & {data[M3][0]} & {data[M4][0]} & {data[M5][0]} & {data[M6][0]} & {data[M7][0]} & {data[M8][0]} & {data[M9][0]} & {data[M10][0]} & {data[M11][0]} & {data[M12][0]} & {data[M13][0]} & {data[M14][0]} & {data[M15][0]} \\\\
        & 0.50 & all & {data[M0][1]} & {data[M1][1]} & {data[M2][1]} & {data[M3][1]} & {data[M4][1]} & {data[M5][1]} & {data[M6][1]} & {data[M7][1]} & {data[M8][1]} & {data[M9][1]} & {data[M10][1]} & {data[M11][1]} & {data[M12][1]} & {data[M13][1]} & {data[M14][1]} & {data[M15][1]} \\\\
        & 0.75 & all & {data[M0][2]} & {data[M1][2]} & {data[M2][2]} & {data[M3][2]} & {data[M4][2]} & {data[M5][2]} & {data[M6][2]} & {data[M7][2]} & {data[M8][2]} & {data[M9][2]} & {data[M10][2]} & {data[M11][2]} & {data[M12][2]} & {data[M13][2]} & {data[M14][2]} & {data[M15][2]} \\\\
        & 0.50:0.95 & small & {data[M0][3]} & {data[M1][3]} & {data[M2][3]} & {data[M3][3]} & {data[M4][3]} & {data[M5][3]} & {data[M6][3]} & {data[M7][3]} & {data[M8][3]} & {data[M9][3]} & {data[M10][3]} & {data[M11][3]} & {data[M12][3]} & {data[M13][3]} & {data[M14][3]} & {data[M15][3]} \\\\
        & 0.50:0.95 & medium & {data[M0][4]} & {data[M1][4]} & {data[M2][4]} & {data[M3][4]} & {data[M4][4]} & {data[M5][4]} & {data[M6][4]} & {data[M7][4]} & {data[M8][4]} & {data[M9][4]} & {data[M10][4]} & {data[M11][4]} & {data[M12][4]} & {data[M13][4]} & {data[M14][4]} & {data[M15][4]} \\\\
        & 0.50:0.95 & large & {data[M0][5]} & {data[M1][5]} & {data[M2][5]} & {data[M3][5]} & {data[M4][5]} & {data[M5][5]} & {data[M6][5]} & {data[M7][5]} & {data[M8][5]} & {data[M9][5]} & {data[M10][5]} & {data[M11][5]} & {data[M12][5]} & {data[M13][5]} & {data[M14][5]} & {data[M15][5]} \\\\
        \\midrule
        \\textbf{{Average Recall (AR)}} & 0.50:0.95 & all (maxDets=1) & {data[M0][6]} & {data[M1][6]} & {data[M2][6]} & {data[M3][6]} & {data[M4][6]} & {data[M5][6]} & {data[M6][6]} & {data[M7][6]} & {data[M8][6]} & {data[M9][6]} & {data[M10][6]} & {data[M11][6]} & {data[M12][6]} & {data[M13][6]} & {data[M14][6]} & {data[M15][6]} \\\\
        & 0.50:0.95 & all (maxDets=10) & {data[M0][7]} & {data[M1][7]} & {data[M2][7]} & {data[M3][7]} & {data[M4][7]} & {data[M5][7]} & {data[M6][7]} & {data[M7][7]} & {data[M8][7]} & {data[M9][7]} & {data[M10][7]} & {data[M11][7]} & {data[M12][7]} & {data[M13][7]} & {data[M14][7]} & {data[M15][7]} \\\\
        & 0.50:0.95 & all (maxDets=100) & {data[M0][8]} & {data[M1][8]} & {data[M2][8]} & {data[M3][8]} & {data[M4][8]} & {data[M5][8]} & {data[M6][8]} & {data[M7][8]} & {data[M8][8]} & {data[M9][8]} & {data[M10][8]} & {data[M11][8]} & {data[M12][8]} & {data[M13][8]} & {data[M14][8]} & {data[M15][8]} \\\\
        & 0.50:0.95 & small & {data[M0][9]} & {data[M1][9]} & {data[M2][9]} & {data[M3][9]} & {data[M4][9]} & {data[M5][9]} & {data[M6][9]} & {data[M7][9]} & {data[M8][9]} & {data[M9][9]} & {data[M10][9]} & {data[M11][9]} & {data[M12][9]} & {data[M13][9]} & {data[M14][9]} & {data[M15][9]} \\\\
        & 0.50:0.95 & medium & {data[M0][10]} & {data[M1][10]} & {data[M2][10]} & {data[M3][10]} & {data[M4][10]} & {data[M5][10]} & {data[M6][10]} & {data[M7][10]} & {data[M8][10]} & {data[M9][10]} & {data[M10][10]} & {data[M11][10]} & {data[M12][10]} & {data[M13][10]} & {data[M14][10]} & {data[M15][10]} \\\\
        & 0.50:0.95 & large & {data[M0][11]} & {data[M1][11]} & {data[M2][11]} & {data[M3][11]} & {data[M4][11]} & {data[M5][11]} & {data[M6][11]} & {data[M7][11]} & {data[M8][11]} & {data[M9][11]} & {data[M10][11]} & {data[M11][11]} & {data[M12][11]} & {data[M13][11]} & {data[M14][11]} & {data[M15][11]} \\\\
        \\bottomrule
    \\end{{tabular}}%
    }}
    \\caption{{COCO evaluation results for {model}}}
    \\label{{tab:coco_{model}}}
\\end{{table*}}

'''
    return table

all_data = {}

for file in files:
    model_name = file.split('_results_')[1].split('.')[0]
    folder_parts = file.split('/')
    
    if folder_parts[2] == 'dpattack':
        attack_name = folder_parts[2] + ' @ epsilon = ' + folder_parts[3].split('_')[1] + ' iters = ' + folder_parts[3].split('_')[3] + ' gridesize = ' + folder_parts[3].split('_')[-1]
    elif folder_parts[2] == 'fgsm':
        attack_name = folder_parts[2] + ' @ epsilon = ' + folder_parts[3].split('_')[1]
    elif folder_parts[2] == 'edge_attack':
        attack_name = folder_parts[2] + ' @ epsilon = ' + folder_parts[3].split('_')[1] + ' iters = ' + folder_parts[3].split('_')[3] + ' gridesize = ' + folder_parts[3].split('_')[-1]
    elif folder_parts[2] == 'baseline':
        attack_name = folder_parts[2]
    elif folder_parts[2] == 'pgdattack':
        attack_name = folder_parts[2] + ' @ epsilon = ' + folder_parts[3].split('_')[1] + ' alpha = ' + folder_parts[3].split('_')[3] + ' iters = ' + folder_parts[3].split('_')[5]
    else:
        attack_name = "Unknown Attack"

    if model_name not in all_data:
        all_data[model_name] = {}

    x = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            num = line.strip().split('maxDets=')[1].split('=')[1].strip()
            x.append(num)

    for m in M:
        if m not in all_data[model_name]:
            all_data[model_name][m] = [''] * 12

    # if attack_name not in all_data[model_name]:
    all_data[model_name][attack_name] = x
    # else:
    #     all_data[model_name][attack_name] += x

# print(models['faster_rcnn'])
# print(json.dumps(all_data, indent=4))

print(get_table('yolov5s', all_data))
print(get_table('faster_rcnn', all_data))