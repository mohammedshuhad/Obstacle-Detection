
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import argparse
import numpy as np
import time
import cv2
import tensorflow as tf
from operator import itemgetter
import picamera
from picamera import PiCamera
from picamera.array import PiRGBArray
ap = argparse.ArgumentParser() 
args = vars(ap.parse_args())
VIDEO_BASE_FOLDER = '/home/pi/Videos/'
def get_video_filename():
    return VIDEO_BASE_FOLDER+'video_out'+'_video.h264'


print("[INFO] start video...")
#with picamera.PiCamera() as camera:
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)
camera.start_recording(get_video_filename())
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'XVID'), 25, (400,300))
print("[INFO] loading network...")
model = load_model("./avcnet_best_5.hdf5",custom_objects={"tf": tf.cast})

alfa=.1
dist=300
dist_old=300
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
	image = frame.array   
	orig = image.copy()
	image = cv2.resize(image, (64,64))
	image = image.astype("float")/255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	(stop, left,right,fly) = model.predict(image)[0]
	my_dict = {'stop':stop, 'left':left, 'right':right,'fly':fly}
	print (my_dict)
	maxPair = max(my_dict.items(), key=itemgetter(1))
	label=maxPair[0]
	proba=maxPair[1]
	label = "{} {:.1f}%".format(label, proba * 100)

	output = imutils.resize(orig, width=400)
	cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

	#cv2.imshow("Frame", output)
	key = cv2.waitKey(10) & 0xFF
	out.write(output)
	rawCapture.truncate(0)
	if key == ord("q"):
		break
camera.stop_recording()
cv2.destroyAllWindows()
