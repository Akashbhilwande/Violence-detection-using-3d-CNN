import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to the dataset
dataset_path = r'C:\Users\bhilw\OneDrive\Documents\Contentshield\ACCURATE MODEL\Real Life Violence Dataset'
output_path = r'ACCURATE MODEL/Real Life Violence Dataset/Test_new'

# Subfolders
violence_folder = os.path.join(dataset_path, 'Violence')
nonviolence_folder = os.path.join(dataset_path, 'NonViolence')

# Output folders
train_violence_folder = os.path.join(output_path, 'Train', 'Violence')
train_nonviolence_folder = os.path.join(output_path, 'Train', 'NonViolence')
test_violence_folder = os.path.join(output_path, 'Test', 'Violence')
test_nonviolence_folder = os.path.join(output_path, 'Test', 'NonViolence')

# Create output folders
os.makedirs(train_violence_folder, exist_ok=True)
os.makedirs(train_nonviolence_folder, exist_ok=True)
os.makedirs(test_violence_folder, exist_ok=True)
os.makedirs(test_nonviolence_folder, exist_ok=True)

def split_and_copy_files(input_folder, train_folder, test_folder, test_size=0.2):
    # Get all files in the folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    
    # Split into train and test
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
    
    # Copy train files
    for file in train_files:
        shutil.copy(os.path.join(input_folder, file), os.path.join(train_folder, file))
    
    # Copy test files
    for file in test_files:
        shutil.copy(os.path.join(input_folder, file), os.path.join(test_folder, file))

# Split violence videos
split_and_copy_files(violence_folder, train_violence_folder, test_violence_folder)

# Split non-violence videos
split_and_copy_files(nonviolence_folder, train_nonviolence_folder, test_nonviolence_folder)

print("Dataset successfully split into training and testing sets.")
