import os
import shutil
import random

dataset_folder='D:\Sim Level 3\Semester 1\Advanced Multimedia\Sign Language Project\Arabic'

def split_dataset(dataset_folder, train_ratio=0.8):
    # Create train and test folders
    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Iterate through each alphabet folder
    for alphabet_folder in os.listdir(dataset_folder):
        alphabet_path = os.path.join(dataset_folder, alphabet_folder)
        
        # Check if it's a directory
        if os.path.isdir(alphabet_path):
            # List all images in the alphabet folder
            images = [f for f in os.listdir(alphabet_path) if f.endswith('.jpg')]
            
            # Calculate the number of images for training
            num_train = int(len(images) * train_ratio)
            
            # Randomly shuffle the images
            random.shuffle(images)
            
            # Split the images into training and testing sets
            train_images = images[:num_train]
            test_images = images[num_train:]
            
            # Move the images to the appropriate folders
            for image in train_images:
                src_path = os.path.join(alphabet_path, image)
                dest_path = os.path.join(train_folder, alphabet_folder, image)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(src_path, dest_path)
            
            for image in test_images:
                src_path = os.path.join(alphabet_path, image)
                dest_path = os.path.join(test_folder, alphabet_folder, image)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(src_path, dest_path)



split_dataset(dataset_folder, train_ratio=0.8)
