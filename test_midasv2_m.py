import numpy as np
import cv2
from rknn.api import RKNN

def prepareInputForInference(image):
	img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img_height, img_width, img_channels = img.shape

	# Input values should be from -1 to 1 with a size of 128 x 128 pixels for the fornt model
	# and 256 x 256 pixels for the back model
	img_input = cv2.resize(img, (256, 256),interpolation = cv2.INTER_CUBIC).astype(np.float32)
	
	# Scale input pixel values to -1 to 1
	mean=[0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]
	reshape_img = img_input.reshape(1, 256, 256, 3)
	img_input = ((img_input/ 255.0 - mean) / std).astype(np.float32)
	img_input = img_input[np.newaxis,:,:,:]        

	return img_input


def processRawDisparity(rawDisparity, img_shape):

	# Normalize estimated depth to have values between 0 and 255
	depth_min = rawDisparity.min()
	depth_max = rawDisparity.max()
	normalizedDisparity = (255 * (rawDisparity - depth_min) / (depth_max - depth_min)).astype("uint8")

	# Resize disparity map to the sam size as the image inference
	estimatedDepth = cv2.resize(normalizedDisparity, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)

	return estimatedDepth


def drawDepth(processedDisparity):
	return cv2.applyColorMap(processedDisparity, cv2.COLORMAP_MAGMA)


def estimateDepth(image):

	input_tensor = prepareInputForInference(image)

	# Perform inference on the image
	outputs = rknn.inference(inputs=[input_tensor])
	rawDisparity = outputs[0][0]
	print(rawDisparity)

	# Normalize and resize raw disparity
	processedDisparity = processRawDisparity(rawDisparity, image.shape)

	# Draw depth image
	colorizedDisparity = drawDepth(processedDisparity)

	return colorizedDisparity


def save_depth(img):

	# Estimate depth
	colorDepth = estimateDepth(img)

	# Add the depth image over the color image:
	combinedImg = cv2.addWeighted(img, 0.7, colorDepth, 0.6, 0)

	# Join the input image, the estiamted depth and the combined image
	img_out = np.hstack((img, colorDepth, combinedImg))
	cv2.imwrite("output3d-2.jpg",img_out)


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    
    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], 
                reorder_channel='0 1 2', target_platform='rk3399pro')
    
    print('done')

    # Load TFLite model
    print('--> Loading model')
    ret = rknn.load_tflite(model='./models/midasv2_float32.tflite')
    if ret != 0:
        print('Load midas2_full_integer_quant failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='./dataset.txt', pre_compile=False)
    if ret != 0:
        print('Build midas2_full_integer_quant failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./midas2_quant.rknn')
    if ret != 0:
        print('Export midas2_quant.rknn failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('./img/motorcycle_741x497.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    save_depth(img)

    print('done')

    # perf
    print('--> Evaluate model performance')
    # perf_results = rknn.eval_perf(inputs=[img])
    print('done')

    rknn.release()

