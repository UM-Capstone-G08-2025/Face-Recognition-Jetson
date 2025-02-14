# Face Recognition Jetson

This project implements a personalized face recognition system optimized for Jetson Nano. The system is designed to distinguish between authorized users and potential intruders using a webcam. The project leverages PyTorch's facial recognition models (MTCNN + InceptionResNetV1) for effective face detection and recognition.

## Project Structure

```
face-recognition-jetson
├── src
│   ├── main.py                     # Entry point for the application
│   ├── train.py                    # Script for training the face recognition model
│   ├── detect.py                   # Real-time face detection and recognition
│   └── utils.py                    # Utility functions for image processing
├── data
│   ├── raw                         # Directory for storing raw training images (positive samples)
│   ├── processed                   # Directory for storing processed images
│   └── negative                    # Directory for storing negative images (images of other people)
├── models
│   └── face_recognition_model.pth  # Trained SVC model saved via joblib
├── configs
│   └── config.yaml                 # Configuration settings for the project
├── LICENSE.txt                     # License file for the project
├── requirements.txt                # List of Python package dependencies
└── README.md                       # Project documentation
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd face-recognition-jetson
   ```

2. **Install Dependencies**
   Make sure you have Python 3.x installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**
   Place your raw images in the `data/raw` directory. Ensure that the images are clear and well-lit for optimal training results.

4. **Train the Model**
   Run the training script to train the face recognition model:
   ```bash
   python src/train.py
   ```

5. **Run the Application**
   After training, you can run the main application to start detecting and recognizing faces:
   ```bash
   python src/main.py
   ```

## Usage Guidelines

- The application will initialize the webcam and start detecting faces in real-time.
- Authorized users will be recognized, while potential intruders will be flagged.
- You can modify the configuration settings in `configs/config.yaml` to adjust model parameters and paths.

## Acknowledgments

This project utilizes the MTCNN and InceptionResNetV1 models for face detection and recognition. Special thanks to the contributors of the respective libraries and models.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

Updated README Instructions
```bash
   git clone <repository-url>
   cd face-recognition-jetson
   git checkout new-feature
   ```


Prepare your Jetson Nano
sudo apt-get update && sudo apt-get upgrade

Make sure your Jetson Nano is updated:
Ensure Python 3.8 is active.
Install PyTorch and TorchVision

Because the Jetson Nano (Jetpack 4.6, Ubuntu 18.04) requires ARM/CUDA–compatible versions, follow NVIDIA’s instructions. For example:
# Remove any previous torch installations
   ```bash
   sudo apt-get remove python3-torch
   ```


# Install torch and torchvision (check NVIDIA’s official forum posts or GitHub for the current wheel URLs)
# Example command (wheel URL may vary):
   ```bash
   sudo apt-get install libopenblas-base libopenmpi-dev
   pip3 install https://developer.download.nvidia.com/compute/redist/jp/v46/torch-1.10.0%2Bnv20.12-cp38-cp38-linux_aarch64.whl
   pip3 install https://developer.download.nvidia.com/compute/redist/jp/v46/torchvision-0.11.1%2Bnv20.12-cp38-cp38-linux_aarch64.whl
   ```


Make sure to verify the proper URLs for your Jetpack 4.6 installation from NVIDIA’s official resources.

Install remaining dependencies

Navigate to the project directory and run:
   ```bash
   pip3 install -r requirements.txt
   ```


Download / Transfer the pretrained model

Ensure that the pretrained classifier file face_recognition_model.pkl (which you obtained by training on your desktop) is placed under the models folder.

Run Detection

To start the real‑time face detection, simply run:
   ```bash
   python3 src/detect.py
   ```


This will open your webcam and begin detection as implemented in detect.py.
