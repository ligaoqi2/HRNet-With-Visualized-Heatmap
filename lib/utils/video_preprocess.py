import numpy as np
import cv2 as cv
import subprocess
from matplotlib import pyplot as plt
from tqdm import tqdm


def video_preprocessing(video_path):
    cap = cv.VideoCapture(video_path)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)

    print("待检测视频为{}".format(video_path))
    print("待检测视频高度为{}".format(height))
    print("待视频宽度为{}".format(width))
    print("待视频总帧数为{}".format(count))

    assert '.mp4' in video_path, "Not support other video format except .mp4!"

    video_out_path = video_path.split('.mp4')[0] + '_pre' + '.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    if height >= width:
        out = cv.VideoWriter(video_out_path, fourcc, fps, (height, width))
    else:
        out = cv.VideoWriter(video_out_path, fourcc, fps, (width, height))
    for _ in tqdm(range(int(count)), desc='video preprocessing'):
        ret, frame = cap.read()
        if not ret:
            print("video preprocess finished")
            break
        if height >= width:
            frame = np.rot90(frame, 1)
        out.write(frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()

    return video_out_path


if __name__ == "__main__":
    video_preprocessing('/home/ligaoqi/projects/python_projects/openpose-liyi00-tpami_384_insize_368/video/video_path/2022_3_19/01203.mp4')

