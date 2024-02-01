import cv2
import mediapipe as mp
import pickle
import numpy as np
# Download the model
model_dict=pickle.load(open('./model.p','rb'))
model=model_dict['model']


#Declare The 3 objects from mediapipe module used to detect landmarks on hand
mp_hands=mp.solutions. hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles= mp.solutions.drawing_styles
hands=mp_hands.Hands (static_image_mode=True, min_detection_confidence=0.3)

# Open camera code
cap =cv2.VideoCapture(0)

while True:
    data_aux = []
    ret, frame= cap.read()
    #change from bgr to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Make process on every photo of hand landmarks and save it in results as a class
    results = hands.process (frame_rgb)
    #Test print the landmarks values for ecah photo have 21 landmarks.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks (
            frame, # image to draw
            hand_landmarks, # model output
            mp_hands.HAND_CONNECTIONS, # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
                # First loop on 3 results classes
        for hand_landmarks in results.multi_hand_landmarks:
            # second loop bring 21 landmarks values for each result photo 
            # Each landmark has x,y,z values 
            # we will use only x,y
            # i from 0 to 20
            for i in range (len (hand_landmarks.landmark)):
                # print(hand_landmarks.landmark[i])
                # print("loop",i)
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
        
        model.predict([np.asarray(data_aux)])
                    
    cv2.imshow('frame',frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

