import cv2
import mediapipe as mp
import pickle
import numpy as np

# Download the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Declare the 3 objects from mediapipe module used to detect landmarks on hand
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'ا' , 1:'ب' ,2:'ج'}
# Open camera code
cap = cv2.VideoCapture(0)

while True:
    data_aux = []
    ret, frame = cap.read()
    
    # Change from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Make process on every photo of hand landmarks and save it in results as a class
    results = hands.process(frame_rgb)
    
    # Test print the landmarks values for each photo have 21 landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Populate data_aux with x and y coordinates
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.extend([x, y])
    
            # Check if data_aux has the correct number of features
            if len(data_aux) == 42:
                prediction=model.predict([np.asarray(data_aux)])
            else:
                print("Error: Incorrect number of features in data_aux")
            prediction_character=labels_dict[int(prediction[0])]
            print(prediction_character)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
