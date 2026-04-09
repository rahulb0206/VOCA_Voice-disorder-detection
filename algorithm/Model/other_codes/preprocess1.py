# Original dataset organisation script — renames PVQD audio files using
# metadata from the Excel sheet. See src/preprocess.py for the cleaned version.

import os
import shutil
import pandas as pd

df = pd.read_excel(r'/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/dataset/dataset.xlsx')

file_names = df['file_id']
labels = df['disorder']
genders = df['gender']
ages = df['age']
category_counts = labels.value_counts()

audio_path = r'/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/Perceptual Voice Qualities Database (PVQD)/Audio_Files/'  
destination_path = r'/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/Perceptual Voice Qualities Database (PVQD)/new_files/'  
os.makedirs(destination_path, exist_ok=True)

# D1: Spasmodic Dysphonia
# D2: Vocal Nodules 
# D3: Vocal Fold Paralysis 

# Counters for each category and gender
counters = {'N_M': 1, 'N_F': 1, 'D1_M': 1, 'D1_F': 1, 'D2_M': 1, 'D2_F': 1, 'D3_M': 1, 'D3_F': 1}

# introducing a list to store details including old and new file names, age, gender, and disorder
renamed_files = []

for i, file_name in enumerate(file_names):
    label = labels[i]
    gender = genders[i] 
    age = ages[i] 
    
    # Search for the actual file in the folder by checking if the partial name is in the full name
    matching_files = [f for f in os.listdir(audio_path) if file_name.lower() in f.lower()]
    
    # If a matching file is found, rename and copy it
    if matching_files:
        original_file = matching_files[0]  # Get the first match
        
        # Generate new file name based on label, gender, and counter
        new_file_name = f"{label}_{gender}_{counters[f'{label}_{gender}']}.wav"
        
        # Construct full path to the old and new file
        old_file_path = os.path.join(audio_path, original_file)
        new_file_path = os.path.join(destination_path, new_file_name)
        
        # Copy and rename the file to the destination folder
        shutil.copy2(old_file_path, new_file_path)
        
        # Increment the counter for that label and gender
        counters[f'{label}_{gender}'] += 1
        
        # Append the details (old name, new name, age, gender, disorder) to the list
        renamed_files.append([original_file, new_file_name, age, gender, label])

# generate a DataFrame with the details
renamed_df = pd.DataFrame(renamed_files, columns=['OldFileName', 'NewFileName', 'Age', 'Gender', 'Disorder'])

renamed_df.to_excel(r'/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/dataset/renamed_files_log.xlsx', index=False)
renamed_df.to_csv(r'/Users/rahulbalasubramani/Desktop/VOCA/model/VOCA-Health/algorithm/Model/dataset/renamed_files_log.csv', index=False)

print("Number of samples in each category:")
for label, count in category_counts.items():
    print(f"{label}: {count}")

print("Doneeeee!")
