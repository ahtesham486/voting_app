from flask import Flask, logging, render_template, request, session, flash, redirect, url_for, make_response
import pymysql
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import pandas as pd
from datetime import datetime
from functools import wraps
import base64
import pdfkit
import time
import torch
import pickle
from io import BytesIO
from PIL import Image
from bs4 import BeautifulSoup
import dlib
from scipy.spatial import distance
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
import smtplib
import json
from flask_mail import Mail, Message
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import logging
from flask_cors import CORS 

# Initialize the Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ajsihh98rw3fyes8o3e9ey3w5dc'

# Enable CORS for all routes
CORS(app)

app.config['MAIL_SERVER'] = 'sandbox.smtp.mailtrap.io'
app.config['MAIL_PORT'] = 2525
app.config['MAIL_USERNAME'] = '48fd56811e318b'
app.config['MAIL_PASSWORD'] = '5a22a51459efe6'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

# Initialize database connection
mydb = pymysql.connect(
    host='localhost',
    user='root',
    password='ahtsham',
    port=3306,
    database='voting_system'
)

## Ensure the upload directory exists
upload_folder = 'code/static/student_images'
# Face detection using MTCNN
mtcnn = MTCNN()

# Face recognition using FaceNet
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Decorator to ensure that the route is only accessible to admins
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('IsAdmin'):
            flash('Admin access required.', 'danger')
            return redirect(url_for('admin'))
        return f(*args, **kwargs)
    return decorated_function

@app.before_request
def initialize():
    session.setdefault('IsAdmin', False)
    session.setdefault('User', None)

@app.route('/')
@app.route('/home')
def home():
    sql = "SELECT end_time FROM election_schedule ORDER BY id DESC LIMIT 1"
    cur = mydb.cursor()
    cur.execute(sql)
    result = cur.fetchone()
    cur.close()

    remaining_seconds = None
    if result:
        end_time = result[0]
        current_time = datetime.now()
        if current_time < end_time:
            remaining_time = end_time - current_time
            remaining_seconds = int(remaining_time.total_seconds())

    # Detect if the request is for Android (API)
    if request.headers.get('Accept') == 'application/json':
        return jsonify({"remaining_seconds": remaining_seconds})
    
    # Web response
    return render_template('index.html', remaining_seconds=remaining_seconds)

@app.route('/admin_dashboard')
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/admin', methods=['POST', 'GET'])
def admin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if (email == 'admin@voting.com') and (password == 'admin'):
            session['IsAdmin'] = True
            session['User'] = 'admin'
            flash('Admin login successful', 'success')
            return render_template('admin_dashboard.html', admin=session.get('IsAdmin', False))
    return render_template('admin.html', admin=session.get('IsAdmin', False))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/set_schedule', methods=['POST', 'GET'])
@admin_required
def set_schedule():
    if request.method == 'POST':
        action = request.form['action']
        start_time_str = request.form['start_time']
        end_time_str = request.form['end_time']
        election_title = request.form['election_title']  # Get the election title from the form

        # Convert the start_time and end_time strings to datetime objects
        try:
            start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M")
            end_time = datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M")
            current_time = datetime.now()
        except ValueError:
            flash('Invalid date/time format. Please use the correct format.', 'danger')
            return redirect(url_for('set_schedule'))

        # Validate the start_time and end_time
        if start_time < current_time:
            flash('Start time cannot be in the past. Please choose a future time.', 'danger')
            return redirect(url_for('set_schedule'))
        
        if end_time < current_time:
            flash('End time cannot be in the past. Please choose a future time.', 'danger')
            return redirect(url_for('set_schedule'))

        if end_time < start_time:
            flash('End time cannot be earlier than the start time. Please choose a valid time range.', 'danger')
            return redirect(url_for('set_schedule'))

        if action == "schedule_new":
            clear_previous_results()
            sql = "INSERT INTO election_schedule (start_time, end_time, election_title) VALUES (%s, %s, %s)"
            cur = mydb.cursor()
            cur.execute(sql, (start_time_str, end_time_str, election_title))
            mydb.commit()
            cur.close()
            flash('New election schedule set successfully! Previous results cleared.', 'success')

        elif action == "extend":
            sql = "UPDATE election_schedule SET start_time=%s, end_time=%s, election_title=%s ORDER BY id DESC LIMIT 1"
            cur = mydb.cursor()
            cur.execute(sql, (start_time_str, end_time_str, election_title))
            mydb.commit()
            cur.close()
            flash('Election time extended successfully!', 'success')

    return render_template('set_schedule.html')

def clear_previous_results():
    cur = mydb.cursor()
    cur.execute("DELETE FROM vote")
    mydb.commit()
    cur.close()

@app.route('/add_candidate', methods=['GET', 'POST'])
@admin_required
def add_candidate():
    if request.method == 'POST':
        try:
            position = request.form.get('position')
            member_name = request.form.get('member_name')
            party_name = request.form.get('party_name')
            symbol_name = request.form.get('symbol_name')
            custom_symbol = request.files.get('custom_symbol')
            cnic = request.form.get('cnic')

            # Check if all required fields are filled out, including CNIC
            if not position or not member_name or not party_name or not symbol_name or not cnic:
                flash('All required fields must be filled out.', 'danger')
                return redirect(url_for('add_candidate'))

            # Handle custom symbol upload if selected
            if symbol_name == 'custom' and custom_symbol:
                filename = custom_symbol.filename
                file_path = os.path.join(upload_folder, filename)
                custom_symbol.save(file_path)
                symbol_name = filename

            # Insert candidate information into the database including CNIC
            sql = "INSERT INTO candidates (position, member_name, party_name, symbol_name, cnic) VALUES (%s, %s, %s, %s, %s)"
            cur = mydb.cursor()
            cur.execute(sql, (position, member_name, party_name, symbol_name, cnic))
            mydb.commit()
            cur.close()
            flash('Candidate added successfully!', 'success')
            return redirect(url_for('add_candidate'))

        except KeyError as e:
            flash(f'Missing required form field: {e}', 'danger')
            return redirect(url_for('add_candidate'))

    return render_template('add_candidate.html')


@app.route('/delete_candidate', methods=['POST'])
@admin_required
def delete_candidate():
    try:
        cnic = request.form.get('cnic')

        # Check if CNIC is provided
        if not cnic:
            flash('CNIC is required to delete a candidate.', 'danger')
            return redirect(url_for('add_candidate'))  # Adjusted redirect to correct form page

        # Delete the candidate by CNIC
        sql = "DELETE FROM candidates WHERE cnic = %s"
        cur = mydb.cursor()
        cur.execute(sql, (cnic,))
        mydb.commit()

        if cur.rowcount == 0:
            flash(f'No candidate found with CNIC {cnic}', 'warning')
        else:
            flash('Candidate deleted successfully!', 'success')

        cur.close()
    except Exception as e:
        flash(f'Error deleting candidate: {e}', 'danger')

    return redirect(url_for('add_candidate'))  # Corrected redirect URL



@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        first_name = request.form['first_name']
        middle_name = request.form.get('middle_name', '')
        last_name = request.form['last_name']
        cnic = request.form['cnic']
        email = request.form['email']
        phone_number = request.form['phone_number']
        voter_id = request.form['voter_id']
        department = request.form['department']
        semester = request.form['semester']
        photo = request.files['photo']

        # Save the photo
        photo_filename = f"{cnic}.jpg"
        photo_path = os.path.join(upload_folder, photo_filename)
        photo.save(photo_path)

        # Extract face embeddings from the photo
        face_embedding = extract_face_embedding(photo_path)

        # Insert student details into the database
        sql_insert = """
        INSERT INTO student 
        (first_name, middle_name, last_name, cnic, email, phone_number, voter_id, department, semester, photo, face_embedding) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur = mydb.cursor()
        cur.execute(sql_insert, (
            first_name, middle_name, last_name, cnic, email, phone_number, voter_id, 
            department, semester, photo_filename, face_embedding
        ))
        mydb.commit()
        cur.close()

        flash('Student added successfully!', 'success')
        return redirect(url_for('add_student'))

    return render_template('add_student.html')

import pickle

def extract_face_embedding(image_path):
    img = cv2.imread(image_path)
    face = mtcnn(img)
    if face is not None:
        face_embedding = resnet(face.unsqueeze(0)).detach().numpy()
        return pickle.dumps(face_embedding)  # Serialize the numpy array into binary
    else:
        return None

@app.route('/registration', methods=['POST', 'GET'])
def registration():
    # Check if the client expects a JSON response
    expects_json = request.headers.get('Accept') == 'application/json'

    if request.method == 'POST':
        # Get CNIC from form or JSON
        cnic = request.form.get('cnic') or (request.json.get('cnic') if request.is_json else None)

        if not cnic:
            # Handle missing CNIC
            if expects_json:
                return jsonify({"error": "CNIC is required."}), 400
            flash("CNIC is required.", "danger")
            return redirect(url_for('registration'))

        try:
            # Fetch student details based on CNIC
            sql = "SELECT * FROM student WHERE cnic = %s"
            cur = mydb.cursor(pymysql.cursors.DictCursor)
            cur.execute(sql, (cnic,))
            student = cur.fetchone()
            cur.close()

        except Exception as e:
            if expects_json:
                return jsonify({"error": "Database error.", "details": str(e)}), 500
            flash("Database error. Please try again later.", "danger")
            return redirect(url_for('registration'))

        if not student:
            if expects_json:
                return jsonify({"error": "No student found with the provided CNIC."}), 404
            flash("No student found with the provided CNIC.", "danger")
            return redirect(url_for('registration'))

        # Process the photo field if it exists
        if 'photo' in student and student['photo']:
            upload_folder = r'E:\fyp\e-voting-app\code\static\student_images'
            student_photo = student['photo']
            file_path = os.path.join(upload_folder, student_photo)

            if not os.path.exists(file_path):
                if expects_json:
                    return jsonify({"error": "Student photo not found."}), 404
                flash(f"Image file not found: {file_path}", "danger")
                return redirect(url_for('registration'))

            student['photo_path'] = url_for('static', filename=f'student_images/{student_photo}')

        # Handle binary data in the student object
        for key, value in student.items():
            if isinstance(value, bytes):
                try:
                    student[key] = value.decode('utf-8')
                except UnicodeDecodeError:
                    student[key] = base64.b64encode(value).decode('utf-8')

        # Store CNIC in session for further usage
        session['voter_cnic'] = cnic

        # Return JSON response for API clients (Postman/Android)
        if expects_json:
            return jsonify(student)

        # Render the HTML template for web users
        return render_template('confirm_voter.html', student=student)

    # For GET requests
    if expects_json:
        # For Postman or Android clients, return a JSON response even on GET requests
        return jsonify({"message": "GET method not supported for JSON."}), 405

    # Render the registration form for web users
    return render_template('registration.html')



# Initialize dlib for blink detection
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(r'C:\Users\MS\Desktop\e-voting-app\code\shape_predictor_68_face_landmarks.dat')  
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# Download from dlib's website

EYE_AR_THRESHOLD = 0.25  # Threshold for eye aspect ratio (EAR)
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames for which EAR < threshold to consider a blink
blink_counter = 0

# Require the user to blink 5 times for liveness detection
blinks_required = 2
blink_total = 0

# Lower the resolution for performance improvement
CAMERA_WIDTH = 400  # Define the width of the camera frame
CAMERA_HEIGHT = 400  # Define the height of the camera frame

# Skip frames for performance
FRAME_SKIP_RATE = 2  # Only process every 5th frame

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_blinks(frame, gray_frame):
    global blink_counter, blink_total

    faces = detector(gray_frame, 0)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_AR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                blink_total += 1
                blink_counter = 0

        # Draw rectangle around eyes
        cv2.rectangle(frame, (left_eye[0][0], left_eye[0][1]), (left_eye[3][0], left_eye[3][1]), (0, 255, 0), 1)
        cv2.rectangle(frame, (right_eye[0][0], right_eye[0][1]), (right_eye[3][0], right_eye[3][1]), (0, 255, 0), 1)

    return blink_total >= blinks_required

# Verify face with blink detection
def verify_face(cnic):
    global blink_total
    blink_total = 0  # Reset blink counter

    # Load stored image path and embedding from the database
    sql = "SELECT photo, face_embedding FROM student WHERE cnic = %s"
    cur = mydb.cursor(pymysql.cursors.DictCursor)
    cur.execute(sql, (cnic,))
    result = cur.fetchone()
    cur.close()

    if not result:
        flash("No student found with the provided CNIC.", "danger")
        return False

    stored_image_path = os.path.join(upload_folder, result['photo'])
    stored_embedding = pickle.loads(result['face_embedding'])

    # Initialize webcam for real-time video capture
    cam = cv2.VideoCapture(0)

    # Set lower resolution for faster performance
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    consistent_matches = 0
    required_consistent_matches = 5  # Require 5 consistent matches for face verification 

    face_verified = False
    blink_check_passed = False
    frame_count = 0  # Initialize frame counter

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames to improve performance
        if frame_count % FRAME_SKIP_RATE != 0:
            continue

        # Convert the captured frame to grayscale and RGB
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the current frame
        boxes, _ = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                face = rgb_frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                # Resize the face and prepare it for embedding extraction
                face_resized = cv2.resize(face, (160, 160))
                face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float()
                face_tensor.div_(255.0)

                # Extract face embedding using InceptionResnetV1
                with torch.no_grad():
                    face_embedding = resnet(face_tensor).numpy()

                # Compute cosine similarity between the embeddings
                similarity = np.dot(stored_embedding, face_embedding.T) / (
                    np.linalg.norm(stored_embedding) * np.linalg.norm(face_embedding)
                )

                if similarity > 0.7:  # Adjust the threshold based on testing
                    consistent_matches += 1
                    cv2.putText(frame, f"Matching... ({consistent_matches}/{required_consistent_matches})",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    if consistent_matches >= required_consistent_matches:
                        face_verified = True
                        break
                else:
                    consistent_matches = 0
                    cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Draw rectangle around detected face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if face_verified:
            # After face is verified, check for blinks (liveness detection)
            blink_check_passed = detect_blinks(frame, gray)

            if blink_check_passed:
                break
            else:
                cv2.putText(frame, f"Blink {blinks_required - blink_total} more times",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the video feed with face and blink detection status
        cv2.imshow('Face Verification and Blink Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if face_verified and blink_check_passed:
        flash("Face and liveness verification passed. You are allowed to vote.", "success")
        return True
    else:
        flash("Face verification or liveness detection failed. Please try again.", "danger")
        return False

# Flask route to confirm voter identity using face verification and blink detection
@app.route('/confirm_voter', methods=['POST'])
def confirm_voter():
    cnic = session.get('voter_cnic')

    # Verify the voter's face and check for blinks
    if not verify_face(cnic):
        if request.headers.get('Accept') == 'application/json':
            return jsonify({"error": "Face verification failed"}), 400
        return redirect(url_for('registration'))  # Show the error message set in `verify_face()`

    # Check if the election is still active
    sql_schedule = "SELECT end_time FROM election_schedule ORDER BY id DESC LIMIT 1"
    cur = mydb.cursor()
    cur.execute(sql_schedule)
    end_time = cur.fetchone()
    cur.close()

    if not end_time or datetime.now() > end_time[0]:
        if request.headers.get('Accept') == 'application/json':
            return jsonify({"error": "Election time is over or not active."}), 400
        flash("Election time is over or not active.", "danger")
        return redirect(url_for('home'))

    if request.headers.get('Accept') == 'application/json':
        return jsonify({"message": "Voter confirmed. Proceed to candidate selection."})

    return redirect(url_for('select_candidate'))


@app.route('/select_candidate', methods=['POST', 'GET'])
def select_candidate():
    # Log the incoming headers for debugging
    logging.debug(f"Request headers: {request.headers}")

    # Check if CNIC is in session (for web requests)
    cnic = session.get('voter_cnic')

    # If CNIC is not in session, check the request headers for API requests
    if not cnic:
        cnic = request.headers.get('Authorization')  # Use Authorization header to pass CNIC

    logging.debug(f"CNIC: {cnic}")  # Log the CNIC value

    if not cnic:
        # If CNIC is still missing, return an error
        if request.headers.get('Accept') == 'application/json':
            return jsonify({"error": "No CNIC found in session or request. Please complete the registration process."}), 400
        flash("No CNIC found in session. Please complete the registration process.", "error")
        return redirect(url_for('registration'))

    # Skip face verification for API requests (Android/Postman)
    if request.headers.get('Accept') != 'application/json':
        # For web requests, proceed with face verification
        if not verify_face(cnic):
            flash("Face verification failed. Please try again.", "error")
            return redirect(url_for('home'))
    
    try:
        # Fetch candidates from the database
        df_nom = pd.read_sql_query('SELECT * FROM candidates', mydb)

        if df_nom.empty:
            if request.headers.get('Accept') == 'application/json':
                return jsonify({"error": "No candidates found."}), 404
            flash('No candidates found.', 'warning')
            return redirect(url_for('home'))

        # Get all unique positions for which candidates are running
        all_positions = df_nom['position'].unique()

        # Create a dictionary of positions and candidates
        position_nominees = {
            pos: df_nom[df_nom['position'] == pos][['symbol_name', 'member_name']].values.tolist()
            for pos in all_positions
        }

        if request.method == 'POST':
            # Collect votes from the form or API request
            votes = {}
            if request.headers.get('Accept') == 'application/json':
                # For API (Android/Postman), expect JSON input
                data = request.json
                if not data:
                    return jsonify({"error": "No data received."}), 400

                for position in all_positions:
                    vote = data.get(position)
                    if not vote:
                        return jsonify({"error": f"Missing vote for position: {position}"}), 400
                    votes[position] = vote
            else:
                # For web form input
                for position in all_positions:
                    vote = request.form.get(position)
                    if not vote:
                        flash(f"Missing vote for position: {position}", "error")
                        return redirect(url_for('select_candidate'))
                    votes[position] = vote

            # Insert votes into the database
            try:
                cur = mydb.cursor()
                for position, vote in votes.items():
                    sql = "INSERT INTO vote (position, vote, cnic) VALUES (%s, %s, %s)"
                    cur.execute(sql, (position, vote, cnic))
                mydb.commit()
            except Exception as db_error:
                logging.error(f"Error inserting votes: {db_error}")
                return jsonify({"error": "Database error while inserting votes"}), 500
            finally:
                cur.close()

            # Fetch voter email from the database using the cnic (for web requests only)
            if request.headers.get('Accept') != 'application/json':
                try:
                    cur = mydb.cursor()
                    sql = "SELECT email FROM student WHERE cnic = %s"
                    cur.execute(sql, (cnic,))
                    email_result = cur.fetchone()
                    cur.close()

                    if email_result is None:
                        return jsonify({"error": "No email found for this CNIC."}), 404

                    email = email_result[0]
                    logging.debug(f"Email found for CNIC: {email}")

                    # Get the current time for the vote confirmation email
                    vote_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Send the confirmation email with candidate images (for web requests)
                    send_vote_confirmation_email(email, votes, df_nom, vote_time)
                except Exception as email_error:
                    logging.error(f"Error sending confirmation email: {email_error}")
                    flash("An error occurred while sending confirmation email.", "error")
                    return redirect(url_for('home'))

            if request.headers.get('Accept') == 'application/json':
                # Return success response for API (Android/Postman)
                return jsonify({"message": "Vote submitted successfully."})

            # Redirect for website users
            return redirect(url_for('home'))

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if request.headers.get('Accept') == 'application/json':
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
        flash(f"An unexpected error occurred: {str(e)}", 'error')
        return redirect(url_for('home'))

    # Return JSON response for Android/API for a GET request
    if request.method == 'GET' and request.headers.get('Accept') == 'application/json':
        return jsonify({
            "position_nominees": position_nominees
        })

    # Render the HTML page to select candidates for each position (Web)
    return render_template('select_candidate.html', position_nominees=position_nominees)
# Function to attach candidate images to the email
def attach_candidate_images(votes, df_nom, msg):
    """
    Attaches candidate images to the email message.
    
    Args:
        votes (dict): The dictionary containing positions and selected symbols.
        df_nom (DataFrame): DataFrame containing the candidates' data.
        msg (MIMEMultipart): The email message object where the images will be attached.
    """
    # Define the correct directory where the images are located
    image_dir = os.path.join(os.getcwd(), 'code', 'static', 'student_images')  # Added 'code' subdirectory
    
    for position, symbol_name in votes.items():
        # Fetch the candidate's details based on their symbol_name
        candidate_row = df_nom[(df_nom['position'] == position) & (df_nom['symbol_name'] == symbol_name)].iloc[0]
        member_name = candidate_row['member_name']
        
        # Construct the full path to the image in the "static/student_images" directory
        image_path = os.path.join(image_dir, symbol_name)

        # Attach candidate image
        try:
            print(f"Trying to attach image for {member_name} from {image_path}")  # Debug line to check path
            with open(image_path, 'rb') as f:
                img_data = f.read()
                image = MIMEImage(img_data)
                image.add_header('Content-ID', f"<{symbol_name}>")
                msg.attach(image)
        except Exception as e:
            # Flash an error message if there's an issue attaching the image
            flash(f"Error attaching image for {member_name}: {str(e)}", "error")
            print(f"Error: {str(e)}")  # Debug the error

# Function to create the HTML email body with embedded images
def create_vote_email_html(votes, df_nom, vote_time):
    """
    Creates the HTML email body with candidate images embedded.

    Args:
        votes (dict): The dictionary containing positions and selected symbols.
        df_nom (DataFrame): DataFrame containing the candidates' data.
        vote_time (str): The timestamp when the vote was cast.

    Returns:
        str: The HTML content of the email.
    """
    vote_details_html = ""
    # Track the positions that have been added to avoid duplicates
    added_positions = set()

    for position, symbol_name in votes.items():
        if position in added_positions:
            continue  # Skip this position if it has already been added
        added_positions.add(position)

        # Fetch the candidate's details based on the symbol_name
        candidate_row = df_nom[(df_nom['position'] == position) & (df_nom['symbol_name'] == symbol_name)].iloc[0]
        member_name = candidate_row['member_name']

        # Create a block for each candidate with their name and embedded image
        vote_details_html += f"""
        <p><strong>{position}</strong>: {member_name} (Symbol: {symbol_name})<br>
        <img src="cid:{symbol_name}" style="width: 100px; height: auto;"></p>
        """

    # Build the full HTML content for the email
    message_body_html = f"""
    <html>
    <head></head>
    <body>
        <p>Dear voter,</p>
        <p>You have successfully cast your vote.</p>
        <p>Details:</p>
        {vote_details_html}
        <p>Time: {vote_time}</p>
        <p>Thank you for participating in the election!</p>
    </body>
    </html>
    """
    
    return message_body_html

# Revised function to send the confirmation email with candidate images
def send_vote_confirmation_email(email, votes, df_nom, vote_time):
    """
    Sends the voting confirmation email with attached candidate images.
    
    Args:
        email (str): The recipient's email address.
        votes (dict): The dictionary containing positions and selected symbols.
        df_nom (DataFrame): DataFrame containing the candidates' data.
        vote_time (str): The timestamp when the vote was cast.
    """
    # Email subject
    subject = "Voting Confirmation for UOBS E-Voting"
    
    # Create the email message object
    msg = MIMEMultipart()
    msg['From'] = "UOBS E-Voting-System <info@scholarlink.biz>"
    msg['To'] = email
    msg['Subject'] = subject

    # Attach the HTML version with embedded images
    message_body_html = create_vote_email_html(votes, df_nom, vote_time)
    msg.attach(MIMEText(message_body_html, 'html'))

    # Attach candidate images
    attach_candidate_images(votes, df_nom, msg)

    # Send the email using SMTP
    try:
        sender_address = "info@scholarlink.biz"
        password = "B@ltist@n941"  # Be cautious with storing sensitive information like this!
        with smtplib.SMTP_SSL("smtp.stackmail.com", 465) as server:
            server.login(sender_address, password)
            server.sendmail(sender_address, email, msg.as_string())
        
        flash('Voted successfully! Confirmation email sent.', 'success')
    except smtplib.SMTPException as e:
        flash(f"An error occurred while sending the confirmation email: {e}", "error")

        
@app.route('/chart')
def chart():
    # Fetch the live vote count from the database
    sql = "SELECT * FROM vote"
    cur = mydb.cursor(pymysql.cursors.DictCursor)
    cur.execute(sql)
    votes = cur.fetchall()
    cur.close()

    # Prepare labels and data for the chart
    labels = []
    data = []

    for vote in votes:
        labels.append(vote['vote'])  # Append the 'vote' value
        data.append(vote.get('count', 0))  # Safely get the 'count', default to 0 if missing

    # Render the chart page
    return render_template('chart.html', labels=labels, data=data)


from flask import jsonify
@app.route('/chart_data')
def chart_data():
    # Fetch the vote data from the database
    sql = '''
    SELECT vote, COUNT(vote) as count 
    FROM vote 
    GROUP BY vote
    '''
    cur = mydb.cursor(pymysql.cursors.DictCursor)
    cur.execute(sql)
    votes = cur.fetchall()
    cur.close()

    # Prepare data for the frontend
    labels = []
    data = []

    # Ensure that 'count' is always available, even if it's 0
    for vote in votes:
        labels.append(vote['vote'])  # Candidate names/symbols
        data.append(vote.get('count', 0))  # Safely get 'count', default to 0 if missing

    # Return the data in JSON format for the chart
    return jsonify({'labels': labels, 'data': data})

import numpy as np

@app.route('/voting_res')
def voting_res():
    # Get the election end time and current time
    sql_schedule = "SELECT end_time, election_title FROM election_schedule ORDER BY id DESC LIMIT 1"
    schedule = pd.read_sql_query(sql_schedule, mydb)

    if schedule.empty:
        if request.headers.get('Accept') == 'application/json':
            return jsonify({"error": "Election schedule is not set."}), 400
        flash('Election schedule is not set.', 'warning')
        return redirect(url_for('home'))

    end_time = schedule['end_time'].iloc[0]
    election_title = schedule['election_title'].iloc[0]
    current_time = datetime.now()

    # Check if the election is still ongoing
    if current_time < end_time:
        if request.headers.get('Accept') == 'application/json':
            return jsonify({"message": "Election is still ongoing. Redirecting to live chart."})
        return redirect(url_for('chart'))

    # Election has ended, show the results
    total_voters_sql = "SELECT COUNT(*) FROM student"
    total_voters = pd.read_sql_query(total_voters_sql, mydb).iloc[0, 0]

    total_participants_sql = "SELECT COUNT(DISTINCT cnic) FROM vote"
    total_participants = pd.read_sql_query(total_participants_sql, mydb).iloc[0, 0]

    sql = '''
    SELECT c.position, c.symbol_name, c.member_name, IFNULL(vote_count.count, 0) as count
    FROM candidates c
    LEFT JOIN (
        SELECT vote, COUNT(vote) as count 
        FROM vote 
        GROUP BY vote
    ) as vote_count ON c.symbol_name = vote_count.vote
    ORDER BY c.position, count DESC
    '''
    votes = pd.read_sql_query(sql, mydb)

    vote_results = {}
    for _, row in votes.iterrows():
        position = row['position']
        symbol = row['symbol_name']
        name = row['member_name']
        count = row['count']

        # Convert numpy.int64 to Python int
        if isinstance(count, (np.integer, np.int64)):
            count = int(count)

        if position not in vote_results:
            vote_results[position] = []
        vote_results[position].append((symbol, name, count))

    # Store election results
    election_record = {
        'election_date': current_time,
        'election_title': election_title,
        'total_voters': int(total_voters),  # Ensure conversion to int
        'total_participants': int(total_participants),  # Ensure conversion to int
        'results': vote_results  # This should already be serializable
    }

    # For API/JSON response
    if request.headers.get('Accept') == 'application/json':
        return jsonify(election_record)

    # Render results page for web
    return render_template('voting_res.html', 
                           vote_results=vote_results, 
                           total_voters=total_voters, 
                           total_participants=total_participants,
                           election_title=election_title)


@app.route('/pdf_results')
def pdf_results():
    # Fetch election schedule with title and start/end time
    sql_schedule = "SELECT election_title, start_time, end_time FROM election_schedule ORDER BY id DESC LIMIT 1"
    schedule = pd.read_sql_query(sql_schedule, mydb)

    if schedule.empty:
        flash('Election schedule is not set.', 'warning')
        return redirect(url_for('home'))

    election_title = schedule['election_title'].iloc[0]
    start_time = schedule['start_time'].iloc[0]
    end_time = schedule['end_time'].iloc[0]
    current_time = datetime.now()

    if current_time < end_time:
        flash('Election results are not available until the election ends.', 'warning')
        return redirect(url_for('home'))

    # Get total voters and participants
    total_voters_sql = "SELECT COUNT(*) FROM student"
    total_voters = pd.read_sql_query(total_voters_sql, mydb).iloc[0, 0]

    total_participants_sql = "SELECT COUNT(DISTINCT cnic) FROM vote"
    total_participants = pd.read_sql_query(total_participants_sql, mydb).iloc[0, 0]

    # Get voting results, using symbol_name as the image and symbol
    sql = '''
    SELECT c.position, c.symbol_name, c.member_name, IFNULL(vote_count.count, 0) as count
    FROM candidates c
    LEFT JOIN (
        SELECT vote, COUNT(vote) as count 
        FROM vote 
        GROUP BY vote
    ) as vote_count ON c.symbol_name = vote_count.vote
    ORDER BY c.position, count DESC
    '''
    votes = pd.read_sql_query(sql, mydb)

    # Organize the results by position
    vote_results = {}
    for _, row in votes.iterrows():
        position = row['position']
        symbol = row['symbol_name']  # Candidate image and symbol
        name = row['member_name']
        count = row['count']

        # Convert numpy.int64 to Python int
        if isinstance(count, (np.integer, np.int64)):
            count = int(count)

        # Build the path to the candidate's image using symbol_name
        photo_path = os.path.join('static/student_images', symbol) if symbol else None

        if position not in vote_results:
            vote_results[position] = []
        vote_results[position].append((symbol, name, count, photo_path))  # Include photo_path in the data

    # Render the HTML template with election details and candidate images
    rendered_html = render_template('pdf_results.html', 
                                    vote_results=vote_results, 
                                    total_voters=total_voters, 
                                    total_participants=total_participants,
                                    election_title=election_title,
                                    start_time=start_time,
                                    end_time=end_time)

    # Generate PDF using pdfkit, allowing local file access
    options = {
        'enable-local-file-access': ''
    }
    pdf = pdfkit.from_string(rendered_html, False, options=options)

    # Return the PDF as a downloadable response
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={election_title}_results.pdf'

    return response


@app.route('/pdf_voters')
@admin_required
def pdf_voters():
    sql = 'SELECT * FROM student'
    voters = pd.read_sql_query(sql, mydb)
    
    rendered = render_template('pdf_voters.html', voters=voters)
    pdf = pdfkit.from_string(rendered, False)
    
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=voters.pdf'
    return response

@app.route('/voters/<symbol_name>')
@admin_required
def voters_for_candidate(symbol_name):
    # Fetch candidate details including position
    sql_candidate = '''
    SELECT c.member_name, c.symbol_name, c.position, COUNT(v.vote) as total_votes
    FROM candidates c
    LEFT JOIN vote v ON c.symbol_name = v.vote
    WHERE c.symbol_name = %s
    GROUP BY c.symbol_name, c.member_name, c.position
    '''
    cur = mydb.cursor(pymysql.cursors.DictCursor)
    cur.execute(sql_candidate, (symbol_name,))
    candidate = cur.fetchone()
    
    # Fetch voter details for the given candidate symbol
    sql_voters = '''
    SELECT s.first_name, s.last_name, s.cnic 
    FROM vote v 
    JOIN student s ON v.cnic = s.cnic 
    WHERE v.vote = %s
    '''
    cur.execute(sql_voters, (symbol_name,))
    voters = cur.fetchall()
    cur.close()

    return render_template('voters_for_candidate.html', 
                           candidate=candidate, 
                           voters=voters)

# Define the image_to_base64 function
def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

# to end the election any time   
@app.route('/admin/end_election', methods=['POST'])
@admin_required
def end_election():
    sql = "UPDATE election_schedule SET end_time = NOW() ORDER BY id DESC LIMIT 1"
    cur = mydb.cursor()
    cur.execute(sql)
    mydb.commit()
    cur.close()

    # Update election results
    sql = "SELECT * FROM vote"
    cur = mydb.cursor(pymysql.cursors.DictCursor)
    cur.execute(sql)
    votes = cur.fetchall()
    cur.close()

    # Get the total number of voters (students) and those who voted
    total_voters_sql = "SELECT COUNT(*) FROM student"
    total_voters = pd.read_sql_query(total_voters_sql, mydb).iloc[0, 0]

    total_participants_sql = "SELECT COUNT(DISTINCT cnic) FROM vote"
    total_participants = pd.read_sql_query(total_participants_sql, mydb).iloc[0, 0]

    # Get voting results, including candidates who received zero votes
    sql = '''
    SELECT c.position, c.symbol_name, c.member_name, IFNULL(vote_count.count, 0) as count
    FROM candidates c
    LEFT JOIN (
        SELECT vote, COUNT(vote) as count 
        FROM vote 
        GROUP BY vote
    ) as vote_count ON c.symbol_name = vote_count.vote
    ORDER BY c.position, count DESC
    '''
    votes = pd.read_sql_query(sql, mydb)

    vote_results = {}
    result_data = ""  # To store result details for the database
    for _, row in votes.iterrows():
        position = row['position']
        symbol = row['symbol_name']
        name = row['member_name']
        count = row['count']

        if position not in vote_results:
            vote_results[position] = []
        vote_results[position].append((symbol, name, count))

        # Append to result_data for storage
        result_data += f"Position: {position}, Candidate: {name} ({symbol}), Votes: {count}\n"
    
    # Insert the results into election_records table
    sql_insert = "INSERT INTO election_records (election_date, election_title, total_voters, total_participants, results) VALUES (%s, %s, %s, %s, %s)"
    cur = mydb.cursor()
    cur.execute(sql_insert, (datetime.now(), "Election Title", total_voters, total_participants, result_data))
    mydb.commit()
    cur.close()

    flash('Election ended successfully!', 'success')
    return redirect(url_for('admin_dashboard'))


# Too keep election records
@app.route('/election_records')
@admin_required
def election_records():
    sql = "SELECT * FROM election_records"
    records = pd.read_sql_query(sql, mydb)

    # Convert the DataFrame to a list of dictionaries
    records_list = records.to_dict(orient='records')

    return render_template('election_records.html', records=records_list)

@app.route('/election_record_details/<int:record_id>')
@admin_required
def election_record_details(record_id):
    sql = "SELECT * FROM election_records WHERE id = %s"
    record = pd.read_sql_query(sql, mydb, params=(record_id,))

    if record.empty:
        flash("No record found with the provided ID.", "danger")
        return redirect(url_for('election_records'))

    # Get voting results, including candidates who received zero votes
    sql = '''
    SELECT c.position, c.symbol_name, c.member_name, IFNULL(vote_count.count, 0) as count
    FROM candidates c
    LEFT JOIN (
        SELECT vote, COUNT(vote) as count 
        FROM vote 
        GROUP BY vote
    ) as vote_count ON c.symbol_name = vote_count.vote
    ORDER BY c.position, count DESC
    '''
    votes = pd.read_sql_query(sql, mydb)

    vote_results = {}
    for _, row in votes.iterrows():
        position = row['position']
        symbol = row['symbol_name']
        name = row['member_name']
        count = row['count']

        if position not in vote_results:
            vote_results[position] = []
        vote_results[position].append((symbol, name, count))

    return render_template('election_record_details.html', 
                           record=record.iloc[0],  # Access the first row of the DataFrame
                           vote_results=vote_results)

# To delete the student from Database
@app.route('/delete_student', methods=['GET', 'POST'])
def delete_student():
    if request.method == 'POST':
        cnic = request.form['cnic']

        # Check if student exists
        sql_check = "SELECT * FROM student WHERE cnic = %s"
        cur = mydb.cursor(pymysql.cursors.DictCursor)
        cur.execute(sql_check, (cnic,))
        student = cur.fetchone()

        if not student:
            flash("No student found with the provided CNIC.", "danger")
            cur.close()
            return redirect(url_for('delete_student'))

        # Delete student record
        sql_delete = "DELETE FROM student WHERE cnic = %s"
        cur.execute(sql_delete, (cnic,))
        mydb.commit()
        cur.close()

        flash(f'Student with CNIC {cnic} has been deleted successfully!', 'success')
        return redirect(url_for('delete_student'))

    return render_template('delete_student.html')

# To uodate the student details
@app.route('/update_student', methods=['GET', 'POST'])
def update_student():
    if request.method == 'POST':
        cnic = request.form['cnic']

        # Fetch student details
        sql_check = "SELECT * FROM student WHERE cnic = %s"
        cur = mydb.cursor(pymysql.cursors.DictCursor)
        cur.execute(sql_check, (cnic,))
        student = cur.fetchone()

        if not student:
            flash("No student found with the provided CNIC.", "danger")
            cur.close()
            return redirect(url_for('update_student'))

        # Process form data and update fields only if they are provided
        fields_to_update = {}
        if request.form['email']:
            fields_to_update['email'] = request.form['email']
        if request.form['phone_number']:
            fields_to_update['phone_number'] = request.form['phone_number']

        if 'photo' in request.files and request.files['photo'].filename != '':
            photo = request.files['photo']
            photo_filename = f"{cnic}.jpg"
            photo_path = os.path.join(upload_folder, photo_filename)
            photo.save(photo_path)
            fields_to_update['photo'] = photo_filename

        # Update student details in the database
        if fields_to_update:
            sql_update = "UPDATE student SET " + ", ".join(f"{key} = %s" for key in fields_to_update.keys()) + " WHERE cnic = %s"
            values = list(fields_to_update.values()) + [cnic]
            cur.execute(sql_update, values)
            mydb.commit()
            flash('Student details updated successfully!', 'success')
        else:
            flash('No fields provided to update.', 'warning')

        cur.close()
        return redirect(url_for('update_student'))

    return render_template('update_student.html')



app.config['TEMPLATES_AUTO_RELOAD'] = True

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
 
