from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False,
	help="path to Caffe 'deploy' prototxt file", default='/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/facedetection/caffemodel/deploy.prototxt.txt')
ap.add_argument("-m", "--model", required=False,
	help="path to Caffe pre-trained model", default='/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/facedetection/caffemodel/res10_300x300_ssd_iter_140000.caffemodel')
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)


while True:

	rval, frame = vs.read()
	print(frame.shape)
	if frame is None:
		print("non")
	frame = imutils.resize(frame, width=1000)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
