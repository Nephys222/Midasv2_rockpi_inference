import platform
import cv2
# import pafy
import numpy as np
from rknnlite.api import RKNNLite
from MidasDepthEstimation.midasRknnEstimator import midasRknnEstimator

# videoUrl = 'https://youtu.be//TGadVbd-C-E'
# videPafy = pafy.new(videoUrl)

def save_depth_video(rknn):

    # Initialize depth estimation model
    rknnEstimator = midasRknnEstimator()

    # Initialize video
    cap = cv2.VideoCapture("./img/Guinea.mp4")
    # print(videoPafy.streams)
    # cap = cv2.VideoCapture(videoPafy.streams[-1].url)
    # cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)

    # We need to set resolutions so, convert them from float to integer.
    frame_width = 1920*2
    frame_height = 1080
    size = (frame_width, frame_height)

    # Below VideoWriter object will create a frame of above defined The output
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter('./result_depthR.mp4',
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            30, size)
    print(cap.isOpened())

    while cap.isOpened():

        # Read frame from the video
        ret, img = cap.read()

        if ret:

            # Estimate depth
            colorDepth = rknnEstimator.estimateDepth(img, rknn)

            # Inference FPS
            cv2.putText(img,
                "FPS : " + '{}'.format(rknnEstimator.fps) + 
                " & Elapsed time : " + '{:.2f}'.format(rknnEstimator.elapsed) + 'ms', 
                (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3,
                cv2.LINE_AA)


            # Add the depth image over the color image:
            # combinedImg = cv2.addWeighted(img,0.7,colorDepth,0.6,0)

            # Join the input image, the estiamted depth and the combined image
            img_out = np.hstack((img, colorDepth))

            # Write the frame into the file 'filename.avi'
            result.write(img_out)

            # cv2.imshow("Depth Image", img_out)

        # Press key q to stop
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    result.release()

if __name__ == '__main__':
    rknn_lite = RKNNLite()

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn('./midasv2_float16_quant.rknn')
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

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
    save_depth_video(rknn_lite)
    print('done')

    rknn_lite.release()
