# Face Recognition Jetson

This project implements a personalized face recognition system optimized for Jetson Nano. The system is designed to distinguish between authorized users and potential intruders using a webcam. The project leverages PyTorch's facial recognition models (MTCNN + InceptionResNetV1) for effective face detection and recognition.

## Project Structure

```
face-recognition-jetson
├── src
│   ├── main.py          # Entry point for the application
│   ├── train.py         # Logic for training the face recognition model
│   ├── detect.py        # Real-time face detection and recognition
│   └── utils.py         # Utility functions for image processing
├── data
│   ├── raw              # Directory for storing raw training images
│   └── processed        # Directory for storing processed images
├── models
│   └── face_recognition_model.pth  # Trained face recognition model
├── configs
│   └── config.yaml      # Configuration settings for the project
├── requirements.txt      # Required Python packages
└── README.md            # Project documentation
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