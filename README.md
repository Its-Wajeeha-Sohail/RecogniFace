# Face Recognition System

This is a Python-based face recognition system that recognizes faces using the FaceNet model and checks if the recognized person exists in the database. It also has a registration form for enter new entity.

## Setup

### 1. Database Configuration
- Install xampp if not already installed.
- Run the `recognition_app.py` script to create the necessary database and table for the system. This script will automatically create a MySQL database named `face_recognition` and a table named `recognized_person` and start processing.

### 2. Dependencies
- Install the required Python packages by running:
- pip install -r requirements.txt


### 3. Running the Program
- Ensure that your webcam is connected and accessible.
- Run the `recognition_app.py` script to start the face recognition system.
- When a face is recognized, the system checks if the person exists in the database. If not, it adds a new entry with the current date and time.


### 4. New Entity
-If a person is new and their data is not store in database than you just need to run `registration_form.py` script , it display a form that help you to enter the new person data in your database.
-when you capture the image from database it store only the location of image in database and image into your local device you just need to give the path where you want to store the images/dataset.

## File Descriptions

### 1. `recognition_app.py`
- Main script for face recognition.
- Initializes the FaceNet model and other necessary components.
- Detects faces using the webcam and recognizes them.


### 2. `registration_form.py`
-Script for new entry registration.
-Run the script ,form open put the all information and click the clear images 3-5 and submit the form all information automatically store in the database.
