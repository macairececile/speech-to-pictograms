import yaml

# Fonction pour lire le fichier YAML et extraire les durations
def extract_durations(file_path):
    durations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        # Parcourir chaque ligne pour extraire la valeur de 'duration'
        for item in data:
            if 'duration' in item:
                durations.append(item['duration'])
    return durations

# Fonction pour calculer la moyenne
def calculate_mean(durations):
    return sum(durations) / len(durations) if durations else 0

# Chemin vers votre fichier YAML
file_path = '/gpfswork/rech/czj/uef37or/ConST/orfeo_process/train_s2p_orfeo.yaml'

# Extraction des durations et calcul de la moyenne
durations = extract_durations(file_path)
mean_duration = calculate_mean(durations)

print(f"Moyenne des durations: {mean_duration:.2f} secondes")
