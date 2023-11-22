import cv2  # OpenCV library for computer vision tasks

import numpy as np  # NumPy for numerical computations and working with arrays

from imutils import face_utils  # Imutils, a set of convenience functions for OpenCV

import dlib  # Dlib, a toolkit for machine learning and computer vision

import streamlit as st  # Streamlit, a Python library for creating web applications

from plyer import notification  # Plyer, a cross-platform API for various features like notifications

import time  # Time module for time-related operations

import pygame  # Pygame, a library for multimedia applications like audio playback

import os  # OS module for interacting with the operating system

from tempfile import NamedTemporaryFile  # module for creating and handling temporary files

# Initialization of pygame mixer
pygame.mixer.init()

# Setting up the Streamlit app title and description
st.set_page_config(page_title="Drowsiness Detection", page_icon="😴")
st.title(":orange[Driver Drowsiness Detection.]")
st.caption("_This is a prototype model, which detects drowsiness, through analyzing eye movements._")


t1, t2 = st.tabs([":violet[HOW IT WORKS ?]",":orange[Let's detect drowsiness]"])

with t1:
    st.title(":blue[How it Works?]")
    
    # You can add various information and content here.
    st.header("**:red[Welcome to the Drowsiness Detection Prototype!]**")
    st.write(":green[This prototype uses computer vision to detect drowsiness in a person's eyes.]")
    st.write(":green[Here's how it works:]")
    st.write(":grey[- It captures video from your camera.]")
    st.write(":grey[- It detects your face and eyes.]")
    st.write(":grey[- It monitors your blinking patterns.]")
    st.write(":grey[- If it detects drowsiness, it will notify you and play an alarm.]")
    st.write(":green[You can upload an audio file to be used as the alarm.]")
    st.write(":green[Click on 'START STREAMING' to begin monitoring.]")
    st.write(":grey[You can stop the monitoring at any time by clicking 'STOP STREAMING'.]")
    st.markdown(":green[*Remember that this is just a prototype for demonstration purposes.*]")
    time.sleep(18)
    st.toast("*If you are done reading, you can move to the next section for testing*")
    time.sleep(3)
if t1:
    time.sleep(2)
    st.toast(":orange[__HELLO 👋__]")



with t2:
    video_placeholder = st.empty()

    # Flag to track if the person is drowsy
    drowsy_flag = False
    drowsy_start_time = None


    # Computing the distance between 2 points
    def compute(ptA, ptB):
        dist = np.linalg.norm(ptA - ptB)
        return dist


    # Checking whether the eye is blinked or not
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


    uploaded_file = st.file_uploader("Upload an audio file to be used as an alarm", type=["mp3", "wav", "mp4"])

    start_button = st.button(":green[START STREAMING]")
    stop_button = st.button(":red[STOP STREAMING]")
    stop_music = st.button(":red[STOP MUSIC]")

    # Creating a placeholder for the audio player
    audio_placeholder = st.empty()

    # Defining the temporary audio file path where our audio will be saved
    audio_file_path = None


    # Function to play the audio
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
                title="Drowsiness Detected",
                message="Driver seems drowsy. Please wake up.",
                timeout=10
            )
            play_audio(audio_file_path)  # Play the audio notification
            time.sleep(3600)  # Wait for 1 hour before showing the notification again


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
    def detect_drowsiness(cap):
        # cap = cv2.VideoCapture(1)
        # checking  Initialization for the camera
        if not cap.isOpened():
            st.write(":blue[CLICK ON START STREAMING]")
            exit()

        # Initializing the face detector and landmark detector
        detect = dlib.get_frontal_face_detector()
        predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # Initializing the variables with 0
        sleep = 0
        drowsy = 0
        active = 0
        status = ""  # Initialize 'status' to an empty string
        color = (0, 0, 0)

        face_frame = []

        # Loop to continuously capture and process the frames
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
                label_position = (x1, y1 - 10)  # ensures that the text always stays on top of your face

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

                check_drowsiness(status)  # Checking for drowsiness

            # Converting the frame to RGB format for displaying in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in the Streamlit app
            video_placeholder.image(frame_rgb, channels="RGB")


    if stop_music:
        pygame.mixer.music.stop()
    else:
        pass

    # Checking if an audio file is uploaded
    if uploaded_file:
        # Save the uploaded audio file as a temporary file
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(uploaded_file.read())
            audio_file_path = temp_audio_file.name

        # Displaying the audio player
        audio_placeholder.audio(open(audio_file_path, "rb").read(), format="audio/mp3")


    if uploaded_file is None and t1:
    
        p = st.warning("UPLOAD AN AUDIO FILE TO BE USED AS AN ALARM")
        while uploaded_file is None:
            time.sleep(30)
            st.toast(":orange[Welcome to the prototype! Upload an audio file to continue]")
    else:
        st.success("AUDIO FILE UPLOADED SUCCESSFULLY")
        time.sleep(4)
    
        # st.toast(":orange[now click on START STREAMING]")
    if not start_button:
        cap = cv2.VideoCapture(1)
        while not start_button and not t2:
            time.sleep(5)
            st.toast(":blue[Great! Now click on _START STREAMING_]")
            time.sleep(5)
        # detect_drowsiness(cap)
    else:
        cap = cv2.VideoCapture(0)
        detect_drowsiness(cap)

    if stop_button:
        cap.release()
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture(1)
        # detect_drowsiness(cap)
    

