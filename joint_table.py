import os
import json

# open all files in ./results/fgsm/epsilon_0.1

def get_files(path):
    """
    Get all files in a directory
    """
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

files = get_files('./results')

def get_table(x, model, attack):
    table = f'''
\\begin{{table}}[h]
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

models = {}

for file in files:
    model_name = file.split('_results_')[1].split('.')[0]
    folder_parts = file.split('/')
    
    if folder_parts[2] == 'dpattack':
        attack_name = folder_parts[2] + ' @ epsilon = ' + folder_parts[3].split('_')[1] + ' iters = ' + folder_parts[3].split('_')[3] + ' gridesize = ' + folder_parts[3].split('_')[-1]
    elif folder_parts[2] == 'fgsm':
        attack_name = folder_parts[2] + ' @ epsilon = ' + folder_parts[3].split('_')[1]
    elif folder_parts[2] == 'baseline':
        attack_name = folder_parts[2]
    else:
        print(folder_parts)
        attack_name = "Unknown Attack"

    print(model_name, attack_name)

    # add model name to models dict
    if model_name not in models:
        models[model_name] = {}

    # add attack name to models dict
    
    # print(model_name, attack_name)
    x = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            num = line.strip().split('maxDets=')[1].split('=')[1].strip()
            x.append(num)

    if attack_name not in models[model_name]:
        models[model_name][attack_name] = x
    else:
        models[model_name][attack_name] += x
        # print(get_table(x, model_name, attack_name))

# print(models['faster_rcnn'])
print(json.dumps(models, indent=4))