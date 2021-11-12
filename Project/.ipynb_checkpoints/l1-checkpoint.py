import dlib
print('dlib cuda:',dlib.cuda.get_num_devices())
import face_recognition
from PIL import Image
import os
import cv2
import numpy as np
import face_recognition
import cv2
from multiprocessing import Process, Manager, cpu_count, set_start_method
import time
import numpy
import threading
import platform
import numpy as np
import pinyin.cedict

# This is a little bit complicated (but fast) example of running face recognition on live video from your webcam.
# This example is using multiprocess.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


def add_alpha_channel(img):
    """ 为jpg图像添加alpha通道 """
 
    b_channel, g_channel, r_channel = cv2.split(img) # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # 创建Alpha通道
 
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # 融合通道
    return img_new


def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    """ 将png透明图像与jpg图像叠加 
        y1,y2,x1,x2为叠加位置坐标值
    """
    
    # 判断jpg图像是否已经为4通道
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)
    else:
        raise Exception('jpg图像不为3通道')
        
    # 判断png图像是否已经为4通道
    if png_img.shape[2] != 4:
        raise Exception('png图像不为4通道')
    
    '''
    当叠加图像时，可能因为叠加位置设置不当，导致png图像的边界超过背景jpg图像，而程序报错
    这里设定一系列叠加位置的限制，可以满足png图像超出jpg图像范围时，依然可以正常叠加
    '''
    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]
 
    if x1 < 0:
        xx1 = -x1
        x1 = 0
    if y1 < 0:
        yy1 = - y1
        y1 = 0
    if x2 > jpg_img.shape[1]:
        xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
        x2 = jpg_img.shape[1]
    if y2 > jpg_img.shape[0]:
        yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
        y2 = jpg_img.shape[0]
 
    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = png_img[yy1:yy2,xx1:xx2,3] / 255.0
    alpha_jpg = 1 - alpha_png
    
    # 开始叠加
    for c in range(0,3):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg*jpg_img[y1:y2,x1:x2,c]) + (alpha_png*png_img[yy1:yy2,xx1:xx2,c]))
    # 4通道转换成3通道
    jpg_img = jpg_img[:,:,:3]
 
    return jpg_img


# Get next worker's id
def next_id(current_id, worker_num):
    if current_id == worker_num:
        return 1
    else:
        return current_id + 1


# Get previous worker's id
def prev_id(current_id, worker_num):
    if current_id == 1:
        return worker_num
    else:
        return current_id - 1


# A subprocess use to capture frames.
def capture(read_frame_list, Global, worker_num):
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    # video_capture.set(3, 640)  # Width of the frames in the video stream.
    # video_capture.set(4, 480)  # Height of the frames in the video stream.
    # video_capture.set(5, 30) # Frame rate.
    print("Width: %d, Height: %d, FPS: %d" % (video_capture.get(3), video_capture.get(4), video_capture.get(5)))

    while not Global.is_exit:
        # If it's time to read a frame
        if Global.buff_num != next_id(Global.read_num, worker_num):
            # Grab a single frame of video
            ret, frame = video_capture.read()
            read_frame_list[Global.buff_num] = frame
            Global.buff_num = next_id(Global.buff_num, worker_num)
        else:
            time.sleep(0.01)

    # Release webcam
    video_capture.release()


# Many subprocess use to process frames.
def process(worker_id, read_frame_list, write_frame_list, Global, worker_num):
    known_face_encodings = Global.known_face_encodings
    known_face_names = Global.known_face_names
    # 边框(透明png)
    bod_png = Global.bod
    while not Global.is_exit:

        # Wait to read
        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num, worker_num):
            # If the user has requested to end the app, then stop waiting for webcam frames
            if Global.is_exit:
                break

            time.sleep(0.01)

        # Delay to make the video look smoother
        time.sleep(Global.frame_delay)

        # 载入边框(保持清晰度)
        bod = bod_png
        
        # Read a single frame from frame list
        frame_process = read_frame_list[worker_id]

        # Expect next worker to read frame
        Global.read_num = next_id(Global.read_num, worker_num)
        
        fh,fw,fs = frame_process.shape

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame_process[:, :, ::-1]

        # Find all the faces and face encodings in the frame of video, cost most time
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
#             # Or instead, use the known face with the smallest distance to the new face
#             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#             best_match_index = np.argmin(face_distances)
#             if matches[best_match_index]:
#                 name = known_face_names[best_match_index]

            # Draw a box around the face
#             cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 0, 255), 2)
        
            # 添加 png(bod边框)
            if top-40 >= 0:
                fac_t = top-40
            else:
                fac_t = 0
            if bottom+10 <= fh:
                fac_b = bottom+10
            else:
                fac_b = fh
            if left-10 >= 0:
                fac_l = left-10
            else:
                fac_l = 0
            if right+10 <= fw:
                fac_r = right+10
            else:
                fac_r = fw
            fac = frame_process[fac_t: fac_b, fac_l: fac_r]  # 第top行到第bottom行，第left列到第right列
            h,w,s=fac.shape
            bod=cv2.resize(bod,(w,h),interpolation=cv2.INTER_LINEAR)#两图像补充为一样的大小
            fac=merge_img(fac, bod, 0, h, 0, w)
            frame_process[fac_t: fac_b, fac_l: fac_r] = fac

            # Draw a label with a name below the face
            cv2.rectangle(frame_process, (left, bottom - 35), (right, bottom), (255,255,255), 1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_process, name, (left + int(w*0.15), bottom - int(h*0.1)), font, h/fh, (255,191,0), 1)
            cv2.putText(frame_process, str(face_distances[first_match_index])[0:6], (left + int(w*0.2), bottom + int(h*0.1)), font, 0.5, (255,191,0), 1)

        # Wait to write
        while Global.write_num != worker_id:
            time.sleep(0.01)

        # Send frame to global
        write_frame_list[worker_id] = frame_process

        # Expect next worker to write frame
        Global.write_num = next_id(Global.write_num, worker_num)


if __name__ == '__main__':
    # 边框(透明png)
    bod = cv2.imread(r'C:\Users\14640\Pictures\bodimg\ds.png',-1)
    
    # 加载人脸数据库
    known_images = []
    known_face_encodings = []
    known_face_names = []
    known_path = "C:/Users/14640/Pictures/known_pictures/"
    imgnames = os.listdir(known_path)
    # print(imgnames)
    for i in range(len(imgnames)):
        print(i)
        print(known_path+imgnames[i])

        # Load a sample picture and learn how to recognize it.
        known_img = face_recognition.load_image_file(known_path+imgnames[i])
        known_face_encoding = face_recognition.face_encodings(known_img)

        # 如果不能检测到脸
        if known_face_encoding == []:
            print(known_path+imgnames[i]+' ---->>> 没有检测到人脸')

        known_face_encodings.append(known_face_encoding[0])
        known_images.append(known_img)
        # 获取成中文名
        china_name = os.path.splitext(imgnames[i])[0]
        print(china_name)
        # 获取拼音
        pinyin_name = pinyin.get(china_name, format="strip", delimiter=" ")
        print(pinyin_name)
        known_face_names.append(pinyin_name)

    # Fix Bug on MacOS
    if platform.system() == 'Darwin':
        set_start_method('forkserver')

    # Global variables
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.frame_delay = 0
    Global.is_exit = False
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()

    # Number of workers (subprocess use to process frames)
    print('cpu_count():', cpu_count())
    if cpu_count() > 2:
        worker_num = cpu_count() - 1  # 1 for capturing frames
    else:
        worker_num = 2

    # Subprocess list
    p = []

    # Create a thread to capture frames (if uses subprocess, it will crash on Mac)
    p.append(threading.Thread(target=capture, args=(read_frame_list, Global, worker_num,)))
    p[0].start()

#     # Load a sample picture and learn how to recognize it.
#     obama_image = face_recognition.load_image_file("obama.jpg")
#     obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

#     # Load a second sample picture and learn how to recognize it.
#     biden_image = face_recognition.load_image_file("biden.jpg")
#     biden_face_encoding = face_recognition.face_encodings(biden_image)[0]


    # Create arrays of known face encodings and their names
    Global.known_face_encodings = known_face_encodings
    Global.known_face_names = known_face_names
    
    Global.bod = bod
    
#     # Create arrays of known face encodings and their names
#     Global.known_face_encodings = [
#         obama_face_encoding,
#         biden_face_encoding
#     ]
#     Global.known_face_names = [
#         "Barack Obama",
#         "Joe Biden"
#     ]

    # Create workers
    for worker_id in range(1, worker_num + 1):
        p.append(Process(target=process, args=(worker_id, read_frame_list, write_frame_list, Global, worker_num,)))
        p[worker_id].start()

    # Start to show video
    last_num = 1
    fps_list = []
    tmp_time = time.time()
    while not Global.is_exit:
        while Global.write_num != last_num:
            last_num = int(Global.write_num)

            # Calculate fps
            delay = time.time() - tmp_time
            tmp_time = time.time()
            fps_list.append(delay)
            if len(fps_list) > 5 * worker_num:
                fps_list.pop(0)
            fps = len(fps_list) / numpy.sum(fps_list)
            print("fps: %.2f" % fps)

            # Calculate frame delay, in order to make the video look smoother.
            # When fps is higher, should use a smaller ratio, or fps will be limited in a lower value.
            # Larger ratio can make the video look smoother, but fps will hard to become higher.
            # Smaller ratio can make fps higher, but the video looks not too smoother.
            # The ratios below are tested many times.
            if fps < 6:
                Global.frame_delay = (1 / fps) * 0.75
            elif fps < 20:
                Global.frame_delay = (1 / fps) * 0.5
            elif fps < 30:
                Global.frame_delay = (1 / fps) * 0.25
            else:
                Global.frame_delay = 0

            # Display the resulting image
            cv2.imshow('Video', write_frame_list[prev_id(Global.write_num, worker_num)])

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Global.is_exit = True
            break

        time.sleep(0.01)

    # Quit
    cv2.destroyAllWindows()
