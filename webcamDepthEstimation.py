import cv2
import numpy as np
from MidasDepthEstimation.midasDepthEstimator import midasDepthEstimator

# Initialize depth estimation model
depthEstimator = midasDepthEstimator()

# Initialize webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL) 	

while True:

	# Read frame from the webcam
	ret, img = camera.read()	

	# Estimate depth
	colorDepth = depthEstimator.estimateDepth(img)

	# Inference FPS
	cv2.putText(img,
			"FPS : " + '{}'.format(depthEstimator.fps) + 
			" & Elapsed time : " + '{:.2f}'.format(depthEstimator.elapsed) + 'ms', 
			(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
			cv2.LINE_AA)

	# Add the depth image over the color image:
	combinedImg = cv2.addWeighted(img,0.7,colorDepth,0.6,0)

	# Join the input image, the estiamted depth and the combined image
	img_out = np.hstack((img, colorDepth, combinedImg))

	cv2.imshow("Depth Image", img_out)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

camera.release()
cv2.destroyAllWindows()