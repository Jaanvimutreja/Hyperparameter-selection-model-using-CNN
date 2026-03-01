try:
    import openml
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openml', 'pandas'])
    import openml

import pandas as pd

print("Fetching active openml datasets...", flush=True)
datasets = openml.datasets.list_datasets(status='active', output_format='dataframe')

# Filter tabular classification datasets
df = datasets[
    (datasets['NumberOfClasses'] >= 2) &
    (datasets['NumberOfClasses'] <= 50) &
    (datasets['NumberOfInstances'] >= 50) &
    (datasets['NumberOfInstances'] <= 50000) &
    (datasets['NumberOfFeatures'] >= 2) &
    (datasets['NumberOfFeatures'] <= 500)
]
df = df.dropna(subset=['NumberOfClasses', 'NumberOfInstances', 'NumberOfFeatures'])

existing_names = {'iris', 'wine', 'breast_cancer', 'glass', 'seeds', 'ionosphere', 'vehicle', 'sonar', 'zoo', 'ecoli', 'vertebral', 'dermatology', 'haberman', 'balance_scale', 'blood_transfusion', 'liver', 'hayes_roth', 'teaching', 'user_knowledge', 'planning_relax', 'diabetes', 'splice', 'sick', 'colic', 'banknote', 'car', 'segment', 'satimage', 'optdigits', 'pendigits', 'waveform', 'page_blocks', 'mfeat_factors', 'steel_plates', 'yeast', 'abalone', 'credit_german', 'vowel', 'wall_robot', 'kr_vs_kp', 'mfeat_pixel', 'phishing', 'eeg', 'letter', 'magic', 'shuttle', 'nursery', 'mushroom', 'electricity', 'nomao', 'heart', 'titanic', 'adult', 'spambase', 'credit_australian', 'mfeat_morphological', 'tic_tac_toe', 'cylinder_bands', 'climate_model', 'monks1'}

selected = []
for idx, row in df.iterrows():
    name = str(row['name']).lower().replace('-', '_').replace(' ', '_').replace('.', '_')
    if name in existing_names: continue
    
    selected.append({
        'name': name,
        'did': int(row['did']),
        'instances': int(row['NumberOfInstances'])
    })
    existing_names.add(name)
    if len(selected) >= 100:
        break

with open('openml_new_100.txt', 'w') as f:
    for s in selected:
        f.write(f'    "{s["name"]}": {s["did"]},\n')
    
    names_str = '", "'.join([s['name'] for s in selected])
    f.write(f'\n--- NAMES ONLY ---\n    "{names_str}"\n')
    
    small = [s['name'] for s in selected if s['instances'] <= 1000]
    small_str = '",\n    "'.join(small)
    f.write(f'\n--- SMALL (<=1000) ---\n    "{small_str}"\n')
    
    med = [s['name'] for s in selected if 1000 < s['instances'] <= 10000]
    med_str = '",\n    "'.join(med)
    f.write(f'\n--- MEDIUM (1000-10000) ---\n    "{med_str}"\n')
    
    lg = [s['name'] for s in selected if s['instances'] > 10000]
    large_str = '",\n    "'.join(lg)
    f.write(f'\n--- LARGE (>10000) ---\n    "{large_str}"\n')

print(f"Done writing {len(selected)} datasets to openml_new_100.txt", flush=True)
