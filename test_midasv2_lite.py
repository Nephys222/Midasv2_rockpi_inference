import platform
import cv2
import numpy as np
from rknnlite.api import RKNNLite
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
	cv2.imwrite("./result_depth1.jpg",img_out)

def show_top5(result):
    output = result[0].reshape(-1)
    # softmax
    output = np.exp(output)/sum(np.exp(output))
    output_sorted = sorted(output, reverse=True)
    top5_str = 'resnet18\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


if __name__ == '__main__':
    rknn_lite = RKNNLite()

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn('./midasv2_float16_quant.rknn')
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    img = cv2.imread('./img/motorcycle_741x497.png')
    # img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    # run on RK3399Pro/RK1808 with Debian OS, do not need specify target.
    if platform.machine() == 'aarch64':
        target = None
    else:
        target = 'rk3399pro'
    ret = rknn_lite.init_runtime(target=target)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    # outputs = rknn_lite.inference(inputs=[img])
    # print(outputs)
    # show_top5(outputs)
    save_depth(img, rknn_lite)
    print('done')

    rknn_lite.release()
