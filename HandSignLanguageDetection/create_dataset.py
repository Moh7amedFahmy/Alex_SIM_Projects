import os
import mediapipe as mp
import pickle
import cv2
import matplotlib.pyplot as plt 

#Declare The 3 objects from mediapipe module used to detect landmarks on hand
mp_hands=mp.solutions. hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles= mp.solutions.drawing_styles
hands=mp_hands.Hands (static_image_mode=True, min_detection_confidence=0.3)
DATA_DIR = 'Dataset'

# produce classfication
data=[]
# 
labels=[]

# Bring each folder in Dataset Directory 0,1,2
for dir_ in os.listdir(DATA_DIR):
    # Second loop on each photo in 3 Directories
    #To use one photo from each dir                     [:1]
    for img_path in os.listdir(os.path.join(DATA_DIR,dir_)):
        data_aux=[]
        # Read the image from directory
        img = cv2.imread(os.path.join(DATA_DIR,dir_,img_path))
        #change from bgr to rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Make process on every photo of hand landmarks and save it in results as a class
        results = hands.process (img_rgb)
        #Test print the landmarks values for ecah photo have 21 landmarks.
        if results.multi_hand_landmarks:
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
                    

            data.append(data_aux)
            labels.append(dir_)
            # dir_=[0,1,2]
            # dataaux_=[store x,y for photos 21 landmarks]
            # print(dir_)
            # print(data_aux)
        # Store in data array[ array for every photo contain x,y 21 ]
        # store in label for every photo from which dir number 0,1,2  
        # print(data)
# print(labels)
f = open('data.pickle', 'wb')
pickle.dump({ 'data': data, 'labels': labels}, f)
f.close()

        
        # Check the mediapipe library drwaing landmarks on photos.
        # for hand_landmarks in results.multi_hand_landmarks:
        #     mp_drawing.draw_landmarks (
        #     img_rgb, # image to draw
        #     hand_landmarks, # model output
        #     mp_hands.HAND_CONNECTIONS, # hand connections
        #     mp_drawing_styles.get_default_hand_landmarks_style(),
        #     mp_drawing_styles.get_default_hand_connections_style())


        #show image in plt 
#         plt.figure()
#         plt.imshow(img_rgb)


# plt.show()