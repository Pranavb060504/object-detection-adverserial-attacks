import os

def get_files(path):
    """
    Get all files in a directory
    """
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

attacks = ['dpattack', 'fgsm', 'edge_attack', 'baseline', 'pgdattack']


def get_table(x, model, attack):
    table = f'''
\\begin{{table}}[h!]
    \\centering
    \\resizebox{{0.5\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{lccc}}
        \\toprule
        Metric & IoU & Area & Value \\\\
        \\midrule
        \\textbf{{Average Precision (AP)}} & 0.50:0.95 & all & {x[0]} \\\\
        & 0.50 & all & {x[1]} \\\\
        & 0.75 & all & {x[2]} \\\\
        & 0.50:0.95 & small & {x[3]} \\\\
        & 0.50:0.95 & medium & {x[4]} \\\\
        & 0.50:0.95 & large & {x[5]} \\\\
        \\midrule
        \\textbf{{Average Recall (AR)}} & 0.50:0.95 & all (maxDets=1) & {x[6]} \\\\
        & 0.50:0.95 & all (maxDets=10) & {x[7]} \\\\
        & 0.50:0.95 & all (maxDets=100) & {x[8]} \\\\
        & 0.50:0.95 & small & {x[9]} \\\\
        & 0.50:0.95 & medium & {x[10]} \\\\
        & 0.50:0.95 & large & {x[11]} \\\\
        \\bottomrule
    \\end{{tabular}}%
    }}
    \\caption{{COCO evaluation results for {model} on {attack}}}
    \\label{{tab:coco_{model}_{attack}}}
\\end{{table}}

    '''
    return table

for attack in attacks:
    files = get_files(f'./results/{attack}/')
    for file in files:
        model_name = file.split('_results_')[1].split('.')[0]

        if model_name == 'faster_rcnn':
            model_name = 'Faster R-CNN'
        else:
            model_name = 'YOLOv5'

        folder_parts = file.split('/')
        
        if folder_parts[2] == 'dpattack':
            attack_name = folder_parts[2] + ' @ epsilon = ' + folder_parts[3].split('_')[1] + ' iters = ' + folder_parts[3].split('_')[3] + ' gridesize = ' + folder_parts[3].split('_')[-1]
        elif folder_parts[2] == 'fgsm':
            attack_name = folder_parts[2] + ' @ epsilon = ' + folder_parts[3].split('_')[1]
        elif folder_parts[2] == 'edge_attack':
            attack_name = 'Edge Attack' + ' @ epsilon = ' + folder_parts[3].split('_')[1] + ' iters = ' + folder_parts[3].split('_')[3] + ' gridesize = ' + folder_parts[3].split('_')[-1]
        elif folder_parts[2] == 'baseline':
            attack_name = folder_parts[2]
        elif folder_parts[2] == 'pgdattack':
            attack_name = 'PGD Attack' + ' @ epsilon = ' + folder_parts[3].split('_')[1] + ' $\\alpha =$ ' + folder_parts[3].split('_')[3] + ' iters = ' + folder_parts[3].split('_')[5]
        else:
            attack_name = "Unknown Attack"

        with open(file, 'r') as f:
            lines = f.readlines()
            x = []
            for line in lines:
                num = line.strip().split('maxDets=')[1].split('=')[1].strip()
                x.append(num)

            print(get_table(x, model_name, attack_name))
