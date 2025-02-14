import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle  # Use pickle instead of joblib

def load_images_from_folders(positive_folder, negative_folder):
    images = []
    labels = []
    # Load positive samples (your face)
    for filename in os.listdir(positive_folder):
        img_path = os.path.join(positive_folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            images.append(img)
            labels.append("Kazi")
    # Load negative samples (other faces)
    for filename in os.listdir(negative_folder):
        img_path = os.path.join(negative_folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            images.append(img)
            labels.append("Not Kazi")
    return images, labels

def train_model(images, labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Preprocess images and extract embeddings, ensuring they align with the labels.
    embeddings = []
    valid_labels = []
    for i, img in enumerate(images):
        img_cropped = mtcnn(img)
        if img_cropped is not None:
            img_embedding = model(img_cropped.to(device)).detach().cpu().numpy()
            # Use the mean embedding for images with multiple faces (if any)
            embeddings.append(img_embedding.mean(axis=0))
            valid_labels.append(labels[i])
    # Normalize embeddings
    embeddings = Normalizer().fit_transform(embeddings)

    # Train an SVC classifier since we now have two classes
    clf = SVC(kernel='linear', probability=True)
    clf.fit(embeddings, valid_labels)

    # Save the classifier and its classes using pickle
    model_data = {'clf': clf, 'classes': clf.classes_}
    with open('models/face_recognition_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

def main():
    positive_folder = 'data/raw'
    negative_folder = 'data/negative'
    images, labels = load_images_from_folders(positive_folder, negative_folder)
    train_model(images, labels)

if __name__ == "__main__":
    main()