from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import argparse
import numpy as np
import time
import cv2
import tensorflow as tf
from operator import itemgetter


ap = argparse.ArgumentParser() 
args = vars(ap.parse_args())

print("[INFO] start video...")
#if ap=='false':
 #   cap=cv2.VideoCapture(0)
#else:    
#    cap.cv2.VideoCapture(args["video"])
cap = cv2.VideoCapture("sample.mp4")
#cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'XVID'), 25, (400,300))


print("[INFO] loading network...")
model = load_model("./avcnet_best_5.hdf5",custom_objects={"tf": tf})

alfa=.1
dist=300
dist_old=300

while True:
    _,frame = cap.read()
    frame = imutils.resize(frame, width=400)
   
    orig  = frame.copy()

       
    frame = cv2.resize(frame, (64,64))
    frame = frame.astype("float")/255.0
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)


    (stop, left,right,fly) = model.predict(frame)[0]
                  
    
    my_dict = {'stop':stop, 'left':left, 'right':right,'fly':fly}
    print my_dict
    maxPair = max(my_dict.iteritems(), key=itemgetter(1))
    label=maxPair[0]
    proba=maxPair[1]
    label = "{} {:.1f}%".format(label, proba * 100)

    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
    
    cv2.imshow("Frame", output)
    key = cv2.waitKey(10) & 0xFF

    out.write(output)
    

    if key == 120:
        break



cv2.destroyAllWindows()
