import urllib.request
import json

url = 'https://www.openml.org/api/v1/json/dataset/list/status/active/limit/3000'
req = urllib.request.Request(url)
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())

datasets = data['data']['dataset']
existing_names = {'iris', 'wine', 'breast_cancer', 'glass', 'seeds', 'ionosphere', 'vehicle', 'sonar', 'zoo', 'ecoli', 'vertebral', 'dermatology', 'haberman', 'balance_scale', 'blood_transfusion', 'liver', 'hayes_roth', 'teaching', 'user_knowledge', 'planning_relax', 'diabetes', 'splice', 'sick', 'colic', 'banknote', 'car', 'segment', 'satimage', 'optdigits', 'pendigits', 'waveform', 'page_blocks', 'mfeat_factors', 'steel_plates', 'yeast', 'abalone', 'credit_german', 'vowel', 'wall_robot', 'kr_vs_kp', 'mfeat_pixel', 'phishing', 'eeg', 'letter', 'magic', 'shuttle', 'nursery', 'mushroom', 'electricity', 'nomao', 'heart', 'titanic', 'adult', 'spambase', 'credit_australian', 'mfeat_morphological', 'tic_tac_toe', 'cylinder_bands', 'climate_model', 'monks1'}

selected = []
for d in datasets:
    name = d['name'].lower().replace('-', '_')
    if name in existing_names: continue
    
    # filter criteria: classification, not too many classes, not too many features
    try:
        n_classes = int(d.get('NumberOfClasses', 0))
        n_features = int(d.get('NumberOfFeatures', 0))
        n_instances = int(d.get('NumberOfInstances', 0))
    except (ValueError, TypeError):
        continue
    
    if n_classes >= 2 and n_classes <= 50 and n_features <= 500 and n_features >= 2 and n_instances >= 50 and n_instances <= 50000:
        selected.append((name, d['did'], n_instances))
        existing_names.add(name)  # ensure unique names
    
    if len(selected) == 100:
        break

print('Fetched', len(selected))
with open('openml_new_100.txt', 'w') as f:
    for s in selected:
        f.write(f'    "{s[0]}": {s[1]},\n')
    f.write('\n--- NAMES ONLY ---\n')
    names_str = '", "'.join([s[0] for s in selected])
    f.write(f'"{names_str}"\n')
    
    f.write('\n--- SMALL (<=1000) ---\n')
    small_str = '", "'.join([s[0] for s in selected if s[2] <= 1000])
    f.write(f'"{small_str}"\n')
    
    f.write('\n--- MEDIUM (1000-10000) ---\n')
    med_str = '", "'.join([s[0] for s in selected if 1000 < s[2] <= 10000])
    f.write(f'"{med_str}"\n')
    
    f.write('\n--- LARGE (>10000) ---\n')
    large_str = '", "'.join([s[0] for s in selected if s[2] > 10000])
    f.write(f'"{large_str}"\n')
