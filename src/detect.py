import cv2
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
from sklearn.preprocessing import Normalizer
import numpy as np

# Load the face recognition model (feature extractor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the MTCNN face detector
mtcnn = MTCNN(keep_all=True, device=device)

# Load the SVC classifier using joblib
clf = joblib.load('models/face_recognition_model.pth')

# Instantiate the Normalizer (L2 normalization)
normalizer = Normalizer()

def recognize_face(frame):
    # Detect faces in the frame
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Validate the bounding box coordinates
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid boxes

            face = frame[y1:y2, x1:x2]
            # Check if the face region is non-empty
            if face.size == 0:
                continue

            # Resize safely
            try:
                face = cv2.resize(face, (160, 160))
            except Exception as e:
                print("Error resizing face region:", e)
                continue

            face_tensor = transforms.ToTensor()(face).unsqueeze(0).to(device)

            # Get the face embedding
            with torch.no_grad():
                embedding = model(face_tensor).detach().cpu().numpy()

            # Normalize the embedding
            embedding_norm = normalizer.transform(embedding)

            # Get prediction from the SVC classifier using probabilities
            probs = clf.predict_proba(embedding_norm)[0]
            max_prob = max(probs)
            label = clf.classes_[np.argmax(probs)]
            # Apply a confidence threshold (60%)
            if max_prob < 0.6:
                label = "Not Kazi"

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def run_detection():
    cap = cv2.VideoCapture(0)
    
    # Set the desired resolution (e.g., 1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        raise Exception("Could not open video device")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process frame for face recognition
        frame = recognize_face(frame)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()