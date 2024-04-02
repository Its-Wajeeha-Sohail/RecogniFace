# dynamic path
from flask import Flask, render_template, request
import mysql.connector
import os

UPLOAD_FOLDER = r'C:/Python_projects/face-rec/facenet/frontend/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the root path dynamically
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'flaskapp',
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_user_folder(user_folder_path):
    if not os.path.exists(user_folder_path):
        os.makedirs(user_folder_path)

@app.route('/')
def index():
    return render_template('registration_foam.html')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    name = request.form['name']
    name = name.strip()  # Add this line to remove leading and trailing spaces
    email = request.form['email']
    phone = request.form['phone']
    cnic = request.form['CNIC']

    # Handle the uploaded image
    uploaded_image = request.files['capturedImage']
    if uploaded_image.filename != '' and allowed_file(uploaded_image.filename):
        # Create user folder path
        user_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], name)
        create_user_folder(user_folder_path)

        # Concatenate relative path (folder name and image name)
        relative_path = os.path.join(name, uploaded_image.filename)
        image_path = os.path.join(ROOT_PATH, app.config['UPLOAD_FOLDER'], relative_path)
        uploaded_image.save(image_path)
    else:
        image_path = None

    # Connect to MySQL database
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    # Insert data into the database
    sql = "INSERT INTO user_data (name, email, phone, cnic, image_path) VALUES (%s, %s, %s, %s, %s)"
    values = (name, email, phone, cnic, relative_path)  # Save relative path in the database
    cursor.execute(sql, values)

    # Commit the changes
    connection.commit()

    # Close the connection
    cursor.close()
    connection.close()

    return 'Form submitted successfully!'

if __name__ == '__main__':
    app.run(debug=True)


