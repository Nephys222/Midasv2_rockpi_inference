import time
import cv2
import numpy as np
# from rknn.api import RKNN


class midasRknnEstimator():

	def __init__(self):
		self.fps = 0
		self.timeLastPrediction = time.time()
		self.frameCounter = 0
		self.elapsed = 0
		self.inputWidth = 256
		self.inputHeight = 256


	def estimateDepth(self, image, rknn):

		input_tensor = self.prepareInputForInference(image)

		# Perform inference on the image
		# rawDisparity = rknn.inference(input_tensor)
		outputs = rknn.inference(inputs=[input_tensor])
		# print("Outputs: {}".format(outputs))
		rawDisparity = outputs[0][0]
		# print("RawDisp: {}".format(rawDisparity))

		# Normalize and resize raw disparity
		processedDisparity = self.processRawDisparity(rawDisparity, image.shape)

		# Draw depth image
		colorizedDisparity = self.drawDepth(processedDisparity)

		# Update fps calculator
		self.updateFps()

		return colorizedDisparity



	def prepareInputForInference(self, image):
		img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		self.img_height, self.img_width, self.img_channels = img.shape

		# Input values should be from -1 to 1 with a size of 128 x 128 pixels for the fornt model
		# and 256 x 256 pixels for the back model
		img_input = cv2.resize(img, (self.inputWidth,self.inputHeight),interpolation = cv2.INTER_CUBIC).astype(np.float32)

		# Scale input pixel values to -1 to 1
		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		# reshape_img = img_input.reshape(1, self.inputHeight, self.inputWidth,3)
		# img_input = ((img_input/ 255.0 - mean) / std).astype(np.float32)
		# print(img_input.shape)
		# img_input = img_input.astype(np.float32)
		img_input = img_input[np.newaxis,:,:,:]

		return img_input

	def processRawDisparity(self, rawDisparity, img_shape):

		# Normalize estimated depth to have values between 0 and 255
		depth_min = rawDisparity.min()
		depth_max = rawDisparity.max()
		normalizedDisparity = (255 * (rawDisparity - depth_min) / (depth_max - depth_min)).astype("uint8")

		# Resize disparity map to the sam size as the image inference
		estimatedDepth = cv2.resize(normalizedDisparity, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)

		return estimatedDepth

	def drawDepth(self, processedDisparity):
		return cv2.applyColorMap(processedDisparity, cv2.COLORMAP_JET)

	def updateFps(self):
		updateRate = 1
		self.frameCounter += 1

		# Every updateRate frames calculate the fps based on the ellapsed time
		if self.frameCounter == updateRate:
			timeNow = time.time()
			ellapsedTime = timeNow - self.timeLastPrediction
			self.elapsed = ellapsedTime

			self.fps = int(updateRate/ellapsedTime)
			self.frameCounter = 0
			self.timeLastPrediction = timeNow


if __name__ == '__main__':

	# Initialize depth estimation model
	rknnEstimator = midasRknnEstimator()

	# Initialize webcam
	camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)

	while True:

		# Read frame from the webcam
		ret, img = camera.read()

		# Estimate depth
		colorDepth = rknnEstimator.estimateDepth(img)

		# Combine RGB image and Depth image
		combinedImg = np.hstack((img, colorDepth))

		cv2.imshow("Depth Image", combinedImg)

		# Press key q to stop
		if cv2.waitKey(1) == ord('q'):
			break

	camera.release()
	cv2.destroyAllWindows()
