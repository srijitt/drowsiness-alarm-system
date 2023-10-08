import cv2
import numpy as np
from imutils import face_utils
import dlib
import streamlit as st
import scipy.io.wavfile as wav
import streamlit as st
from plyer import notification
import time
import pygame
import os
from tempfile import NamedTemporaryFile




# Initialization pygame mixer
pygame.mixer.init()


# Set up Streamlit app title and description
st.set_page_config(page_title=":violet[DROWSINESS_IDENTIFIER]", page_icon = "😴")
st.title(":orange[Sleep Detection Prototype😴]")
st.caption("_Sleep Well in Bed, Not in the Car,while driving!_")

# Create placeholders for buttons and video frame
start_button = st.button(":green[START STREAMING]")
b = st.button(":red[STOP THE MUSIC]")


video_placeholder = st.empty()

# Flag to track if the person is drowsy
drowsy_flag = False
drowsy_start_time = None


# Computes distance between 2 points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


# Checks whether blinked or not
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

uploaded_file = st.file_uploader("Upload an audio file to be used as an alarm", type=["mp3","wav","mp4"])

# Create a placeholder for the audio player
audio_placeholder = st.empty()

# Define the temporary audio file path
audio_file_path = None

# Function to play audio
def play_audio(audio_file_path):
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Function to notify
def notify():
    for _ in range(5):  # Play the notification 5 times
        notification.notify(
            title="WAKE UP! WAKE UP!",
            message="STOP THE CAR, STOP THE CAR",
            timeout=10
        )
        play_audio(audio_file_path)  # Play the audio notification
        time.sleep(3600)  # Wait for 1 hour before showing the notification again

# Check if an audio file is uploaded
if uploaded_file:
    # Save the uploaded audio file as a temporary file
    with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.write(uploaded_file.read())
        audio_file_path = temp_audio_file.name

    # Display the audio player
    audio_placeholder.audio(open(audio_file_path, "rb").read(), format="audio/mp3")


# Function to check for drowsiness
def check_drowsiness(status):
    global drowsy_flag
    global drowsy_start_time

    if status == "Drowsy" or status == "Sleeping":
        if not drowsy_flag:
            drowsy_flag = True
            drowsy_start_time = time.time()
        else:
            if time.time() - drowsy_start_time >= 10:  # Check if drowsy for more than 1 minute
                notify()
    else:
        drowsy_flag = False

# Function to read the camera frame and process it
def detect_drowsiness():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Initializing the face detector and landmark detector
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    sleep = 0
    drowsy = 0
    active = 0
    status = ""  # Initialize 'status' to an empty string
    color = (0, 0, 0)

    face_frame = []

    # Loop to continuously capture and process frames
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label_position = (x1, y1 - 10)  # ensures that the text always stays on top of the frame

            landmarks = predict(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # setting up LANDMARKS
            left_blink = blinked(landmarks[36], landmarks[37],
                                 landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43],
                                  landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            # Setting up BASE CONDITIONS for prediction
            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status = "Sleeping"
                    color = (255, 255, 0)

            elif left_blink == 1 or right_blink == 1:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "Drowsy"
                    color = (0, 255, 255)

            else:
                drowsy = 0
                sleep = 0
                active += 1
                if active > 6:
                    status = "Active"
                    color = (0, 255, 0)

            cv2.putText(frame, status, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            check_drowsiness(status)  # Check for drowsiness

        # Convert the frame to RGB format for displaying in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in the Streamlit app
        video_placeholder.image(frame_rgb, channels="RGB")

        # Check if the Stop button is pressed
        if stop_button:
            cap.release()
            cv2.destroyAllWindows()
            

# Start streaming when the Start button is pressed
if start_button:
    stop_button = st.button(":red[STOP STREAMING]")
    if stop_button:
        pygame.mixer.music.stop()
    detect_drowsiness()
if b:
    pygame.mixer.music.stop()
else:
    pass

