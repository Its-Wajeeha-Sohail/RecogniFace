import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pyttsx3
import time

# Load a pre-trained MTCNN face detector
mtcnn = MTCNN(keep_all=True)

# Load a pre-trained FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Function to get face embeddings from an image
def get_face_embeddings(image):
    faces = mtcnn(image)
    embeddings_list = []
    if faces is not None:
        resized_faces = []
        for face in faces:
            resized_face = cv2.resize(face.permute(1, 2, 0).numpy(), (160, 160))
            resized_faces.append(resized_face)

        resized_faces = np.array(resized_faces)
        resized_faces = np.transpose(resized_faces, (0, 3, 1, 2))
        resized_faces = torch.tensor(resized_faces).float()

        embeddings = facenet_model(resized_faces)
        embeddings_list = embeddings.detach().numpy()

    return embeddings_list

# Function to load images and their respective labels (folder names)
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            image = cv2.imread(img_path)
            images.append(image)
            labels.append(folder.split('/')[-1])  # Assuming folder structure is 'folder_name/image.jpg'
    return images, labels

# Load images and labels from folders
known_faces = []
known_labels = []

# Path to folders containing images of known people
folders_path = "ps"

for folder_name in os.listdir(folders_path):
    folder_path = os.path.join(folders_path, folder_name)
    if os.path.isdir(folder_path):
        images, labels = load_images_from_folder(folder_path)
        known_faces.extend(images)
        known_labels.extend([folder_name] * len(images))  # Use folder name as label for all images

# Get embeddings and labels for known faces
known_embeddings = []

for face_image in known_faces:
    face_embeddings = get_face_embeddings(face_image)
    if face_embeddings is not None:
        known_embeddings.append(face_embeddings)


# Capture video from webcam or provide a video file path
video_capture = cv2.VideoCapture(0)  # Use 0 for webcam
camera_ready = False
person_present = False

confidence_threshold = 0.8  # Set a threshold for face recognition confidence

prev_recognized_labels = set()

# Function to get the current time as a string
def get_current_time():
    return time.strftime("%H:%M:%S")

while True:
    ret, frame = video_capture.read()

    # Get face embeddings from the current frame
    frame_embeddings = get_face_embeddings(frame)

    # Compare the embeddings with known embeddings and calculate distances
    if frame_embeddings is not None:
        recognized_labels = set()
        for i, frame_embedding in enumerate(frame_embeddings):
            distances = []
            for known_emb in known_embeddings:
                dist = np.linalg.norm(known_emb - frame_embedding, axis=1)
                distances.append(np.mean(dist))

            # Choose the label with the minimum distance as the recognized face
            min_distance = np.min(distances)
            recognized_label = known_labels[np.argmin(distances)] if min_distance < confidence_threshold else "Unknown"

            recognized_labels.add(recognized_label)

            # Display the recognized label and distance on the frame for each face
            cv2.putText(frame, f"Face {i+1}: {recognized_label} - {min_distance:.2f}", (50, 50 + i*50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the current time on the frame
        current_time = get_current_time()
        cv2.putText(frame, f"Time: {current_time}", (frame.shape[1] - 250, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        # Speak only if the recognized label changes and the person is present
        new_labels = recognized_labels - prev_recognized_labels
        if person_present and new_labels and camera_ready:
            engine.say(f"Recognized: {', '.join(new_labels)}")
            engine.runAndWait()
            prev_recognized_labels = recognized_labels

        # Update person presence based on the number of recognized faces
        person_present = len(frame_embeddings) > 0

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Check if the camera is ready
    if not camera_ready and ret:
        print("Camera is ready!")
        camera_ready = True

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
video_capture.release()
cv2.destroyAllWindows()
