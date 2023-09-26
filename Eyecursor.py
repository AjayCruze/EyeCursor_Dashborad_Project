import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Initialize MediaPipe's FaceMesh solution
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get the screen width and height
screen_w, screen_h = pyautogui.size()

# Initialize timer variables for automatic clicks
click_interval = 3 # Time interval in seconds for automatic clicks
last_click_time = time.time()

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a more intuitive experience
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using FaceMesh to obtain facial landmarks
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        # Calculate cursor position based on eye landmarks
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            if id == 1:
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
                pyautogui.moveTo(screen_x, screen_y)

    # Check if it's time for an automatic click
    current_time = time.time()
    if current_time - last_click_time >= click_interval:
        pyautogui.click()
        last_click_time = current_time

    # Display the processed frame
    cv2.imshow('Eye Controlled Mouse', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the OpenCV window
cam.release()
cv2.destroyAllWindows()