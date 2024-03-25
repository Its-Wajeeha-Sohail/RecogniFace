import os
import sys
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pyttsx3
import time
import mysql.connector

app = Flask(__name__)

# Flag to indicate if recognition is in progress
recognition_in_progress = False

# Load a pre-trained MTCNN face detector
mtcnn = MTCNN(keep_all=True)

# Load a pre-trained FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'flaskapp',
}

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
folders_path = r"C:\Python_projects\face-rec\facenet\frontend\ps"

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

# Initialize pyttsx3 engine
engine = pyttsx3.init()

def generate_frames():
    global camera_ready, person_present, prev_recognized_labels

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
                engine.say(f"Welcome to Falcon: {', '.join(new_labels)}")
                engine.runAndWait()

                # Save the recognition data to the database and exit the program
                save_recognition_data_and_exit(new_labels)

            # Update person presence based on the number of recognized faces
            person_present = len(frame_embeddings) > 0

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

def save_recognition_data_and_exit(labels):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    recognized_time = get_current_time()
    try:
        for label in labels:
            cursor.execute('''
                INSERT INTO recognition_data (name, recognized_time) VALUES (%s, %s)
            ''', (label, recognized_time))
        connection.commit()
    except Exception as e:
        print(f"Error saving data to database: {e}")
    finally:
        connection.close()

    if labels:
        message = f"Face(s) recognized: {', '.join(labels)} - Time: {recognized_time}"
        print(message)  # Print the message to the console.
    print(f"Executing SQL query: {cursor.statement}")
    sys.exit(0)  # Terminate the program

@app.route('/start_recognition', methods=['POST'])
def recognize():
    global camera_ready
    if camera_ready:
        # Get face embeddings from the current frame
        frame = cv2.VideoCapture(0).read()[1]  # Capture a frame from the webcam
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

            # Save the recognition data to the database
            if recognized_labels:
                save_recognition_data_and_exit(recognized_labels)

    return {'status': 'Recognition data saved in the database'}

# New routes for starting and stopping the camera
@app.route('/start_recognition', methods=['GET'])
def start_recognition():
    global camera_ready
    camera_ready = True
    return {'status': 'Camera started for recognition'}

@app.route('/')
def index():
    return render_template('recognition_app.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    name = request.form['name']
    recognized_time = request.form['recognized_time']

    # Connect to MySQL database
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    # Insert data into the database
    sql = "INSERT INTO user_data (name, recognized_time) VALUES (%s, %s)"
    values = (name, recognized_time)
    cursor.execute(sql, values)

    # Commit the changes
    connection.commit()

    # Close the connection
    cursor.close()
    connection.close()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)




