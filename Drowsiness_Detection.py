import cv2
from scipy.spatial import distance
import dlib
import numpy as np
from playsound import playsound
import sounddevice as sd
import soundfile as sf
import time

def eyeaspectratio(eye) :
    ver1 = distance.euclidean(eye[1] , eye[5])
    ver2 = distance.euclidean(eye[2] , eye[4])
    hor = distance.euclidean(eye[0] , eye[3])

    EAR = (ver1+ver2)/(2.0*hor)
    return EAR

if __name__ == "__main__" :
    threshold = 0.3
    filename = "loud_alarm.wav"
    data, fs = sf.read(filename, dtype='float32')  
    frame_limit_for_warning = 30
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/mactavish/Computer Vision Projects/Drowsiness Detection/shape_predictor_68_face_landmarks.dat")
    count = 0
    cap = cv2.VideoCapture(0)
    while True :
        ret , frame = cap.read()
        frame = cv2.resize(frame , (512,512))
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray , 0)
        for face in faces :
            shape = predictor(gray , face)
            leye = str([shape.part(36) , shape.part(37) , shape.part(38) , shape.part(39) , shape.part(40) , shape.part(41)])
            reye = str([shape.part(42) , shape.part(43) , shape.part(44) , shape.part(45) , shape.part(46) , shape.part(47)])
            leye = np.array(leye.replace('[','').replace('(','').replace(')','').replace(']','').replace('point','').split(','),dtype=int).reshape(-1,2)
            reye = np.array(reye.replace('[','').replace('(','').replace(')','').replace(']','').replace('point','').split(','),dtype=int).reshape(-1,2)
            lefteyehull = cv2.convexHull(leye)
            righteyehull = cv2.convexHull(reye)
            cv2.drawContours(frame , [lefteyehull], -1 ,(0,0,255) , 1)
            cv2.drawContours(frame , [righteyehull], -1 ,(0,0,255) , 1)
            lEAR = eyeaspectratio(leye)
            rEAR = eyeaspectratio(reye)
            EAR = (lEAR + rEAR) / 2.0
            print(EAR)
            #print(count)
            if (EAR <= threshold) :
                count+=1
                print(count)
                if(count>=frame_limit_for_warning) :
                    #print('Yes')
                    cv2.putText(frame , "ALERT" , (10,30) , cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (0,255,0),2)
                    time.sleep(1)
                    sd.play(data, fs)
                    #status = sd.wait() 
                    count = 0
            else :
                count=0
        
            cv2.putText(frame, "EAR: {:.2f}".format(EAR), (400, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        cv2.imshow("Window",frame)
        if(cv2.waitKey(1)&0xFF==27) :
            cv2.destroyAllWindows()
            cap.release()
            break