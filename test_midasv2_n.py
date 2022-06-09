import numpy as np
import cv2
from rknn.api import RKNN
from MidasDepthEstimation.midasRknnEstimator import midasRknnEstimator


def save_depth(img, rknn):

	# Initialize depth estimation model
	rknnEstimator = midasRknnEstimator()

	# Estimate depth
	colorDepth = rknnEstimator.estimateDepth(img, rknn)

	# Inference time
	cv2.putText(img,
		"Elapsed time : " + '{:.2f}'.format(rknnEstimator.elapsed) + 'ms', 
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
		cv2.LINE_AA)

	# Add the depth image over the color image:
	combinedImg = cv2.addWeighted(img, 0.7, colorDepth, 0.6, 0)

	# Join the input image, the estiamted depth and the combined image
	img_out = np.hstack((img, colorDepth, combinedImg))
	cv2.imwrite("./output/output5d.jpg",img_out)


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
    ret = rknn.load_tflite(model='./models/midasv2_float16_quant.tflite')
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
    ret = rknn.export_rknn('./midas2_float16_quant.rknn')
    if ret != 0:
        print('Export midas2_quant.rknn failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('./img/motorcycle_741x497.png')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    # outputs = rknn.inference(img)
    # print(outputs[0][0])
    save_depth(img, rknn)

    print('done')

    # perf
    print('--> Evaluate model performance')
    # perf_results = rknn.eval_perf(inputs=[img])
    print('done')

    rknn.release()

