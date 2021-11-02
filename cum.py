'''
Created by: Yogi/Vegeta2007k
'''

import numpy as np
import pywhatkit
import face_recognition as fr
import pyautogui
import webbrowser
import cv2



#It's funny because it's cum haha lmfaoo
CumCapture = cv2.VideoCapture(0)

# Feeding data about the people
pitaji_image = fr.load_image_file("pitaji/pic#1.png")
pitaji_face_encoding = fr.face_encodings(pitaji_image)[0]

mataji_image = fr.load_image_file("Mummy.png")
mataji_face_encoding = fr.face_encodings(mataji_image)[0]

kislay_image = fr.load_image_file("kislay.png")
kislay_face_encoding = fr.face_encodings(kislay_image)[0]

known_face_encondings = [pitaji_face_encoding, kislay_face_encoding, mataji_face_encoding]
known_face_names = ["Pitaji", "Kislay", "Mummy"]

while True: 

    # grabbing a single frame from the camera
    ret, frame = CumCapture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        #Compares the faces with the data provided

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            #Closes the present tab and opens up a maths video

            if name == "Pitaji":
                pyautogui.hotkey("ctrl", "w")                
                webbrowser.open("https://www.youtube.com/watch?v=t_K95XuXkSw")

            
                

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Bruh', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

CumCapture.release()
cv2.destroyAllWindows()
