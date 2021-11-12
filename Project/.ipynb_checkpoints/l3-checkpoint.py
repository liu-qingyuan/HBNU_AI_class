import dlib
print('dlib.DLIB_USE_CUDA:',dlib.DLIB_USE_CUDA)
print('dlib cuda:',dlib.cuda.get_num_devices())
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import numpy as np
import face_recognition
import cv2
from multiprocessing import Process, Manager, cpu_count, set_start_method, Lock, Queue
import time
import numpy
import threading
import platform
import numpy as np
import pinyin.cedict



from flask import Flask,render_template, request, flash, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import datetime

# 目标：判断规定时间是不是签到
# 获取   签到时间 (当前时间)，判断   签到时间  是不是  早于  上课时间(规定时间)

app = Flask(__name__)

# 配置数据库地址
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:yydsjkhs@localhost/flask_class'

# 跟踪数据库修改(false)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 每次请求结束后都会自动提交数据库中的变动
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN']=True

# 配置密钥
app.secret_key = 'qingyuan'

# 数据库对象
db = SQLAlchemy(app)

db.make_connector( app, bind=None )
Session = db.sessionmaker(bind=db.engine)


'''
1. 配置数据库
    a. 导入SQLAlchemy扩展
    b. 创建db对象, 配置参数
    c. 终端创建数据库 
2. 添加角色，学生，签到模型
'''

# 数据库的模型，需要继承db.Model
class Role(db.Model):
    # 定义表名
    __tablename__ = 'roles'
    
    # 定义字段
    # db.Column表示是一个字段
    role_id = db.Column(db.Integer, primary_key=True)
    role_class = db.Column(db.String(8), unique=True)
    
    def __repr__(self):
        return ('Role: %s' %(role_class))
    
class Student(db.Model):
    # 定义表名
    __tablename__ = 'students'
    
    # 定义字段
    # db.Column表示是一个字段
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(16), unique=False)
    # db.ForeignKey('roles.id')  --> '外键'.'表名'
    role_id = db.Column(db.Integer, db.ForeignKey('roles.role_id'))
    # 128组face_encoding
    names = locals()
    for i in range(128):
        names['face_encoding%s' % i] = db.Column(db.DECIMAL(24, 22))
    
    # 关系引用
    roles = db.relationship('Role')
    
    def __repr__(self):
        return ('Student: %s %s' %(name, role_id))

class Sign(db.Model):
    __tablename__ = 'signs'
    sign_id =db.Column(db.Integer, primary_key=True)
    class_start_time = db.Column(db.DateTime)
    sign_in_time = db.Column(db.DateTime)
    is_sign_in = db.Column(db.Boolean)
    # db.ForeignKey('roles.id')  --> '外键'.'表名'
    sno = db.Column(db.Integer, db.ForeignKey('students.sno'))
    
    # 关系引用
    students = db.relationship('Student')

    def __repr__(self):
        return ('Book: %s %s %s %s' %(class_start_time, sign_in_time, is_sign_in, sno))

def is_sign(class_start_time, sign_in_time):
    '''
    格式: class_start_time, sign_in_time为datetime.datetime
    返回: True or False
    '''
    return class_start_time>sign_in_time
    
@app.route('/', methods=['POST','GET'])
def index():
    
#     # 规定上课时间
#     class_start_time = datetime.datetime.now().replace(hour=22, minute=30, second=0,microsecond=0)
        
#     # 查询所有学生信息
#     stus = Student.query.all()
    
    
#     now = datetime.datetime.now()
#     sign1 = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=2019012604)
#     now = datetime.datetime.now()
#     sign2 = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=2017011837)
#     now = datetime.datetime.now()
#     sign3 = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=2019012621)
#     now = datetime.datetime.now()
#     sign4 = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=2019012606)
#     now = datetime.datetime.now()
#     sign5 = Sign(class_start_time=class_start_time+datetime.timedelta(hours=1), sign_in_time=now,is_sign_in =is_sign(class_start_time+datetime.timedelta(hours=1), now),sno=2019012604)
#     # 把数据提交给用户会话
#     db.session.add_all([sign1,sign2,sign3,sign4,sign5])
#     # 提交会话
#     db.session.commit()
    
    # 查询所有签到信息
    signs = Sign.query.all()
    
    # 传递信息给网页模板
    return render_template('index.html', signs=signs)



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
def process(worker_id, read_frame_list, write_frame_list, Global, worker_num,lock):
    known_face_encodings = Global.known_face_encodings
    known_face_names = Global.known_face_names
    known_face_snos = Global.known_face_snos
    # 上课时间
    class_start_time = Global.class_start_time
    # 开始签到时间
    start_sign_time = Global.start_sign_time
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
            # 阈值tolerance：0.4 
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)

            name = "Unknown"

#             # If a match was found in known_face_encodings, just use the first one.
#             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#             if True in matches:
#                 first_match_index = matches.index(True)
#                 name = known_face_names[first_match_index]
#                 sno = known_face_snos[first_match_index]

            
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                sno = known_face_snos[best_match_index]
                
                # 生成签到数据
                # 获取对应人, 及识别时间
                # 获取对应学生数据对象
                stu = Student.query.filter_by(sno=sno).first()
                sno = stu.sno
                
                
                
                def get_sign_time():
                    # 获取识别时间
                    now = datetime.datetime.now()
                    # 如果数据库中有数据(从start_sign_time到class_start_time之后2小时之间)      不写入
                    end = class_start_time + datetime.timedelta(hours=2)
                    if now >=start_sign_time and now<=end:
                        print('在可签到区间')
                        
                        try:
                            # 数据库对象
                            db = SQLAlchemy(app) #每个线程都可以直接使用数据库模块定义的Session(保证进程安全！！！)
                            db.session.flush()
                            sign_t = db.session.query(Sign).filter(Sign.sno==sno, Sign.class_start_time==class_start_time, Sign.sign_in_time>=start_sign_time, Sign.sign_in_time<=end).all()
                            
                            # 如果数据库中没有数据(从start_sign_time到class_start_time之后2小时之间)      写入
                            if sign_t == []:
                                sign = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=sno)
                                try:
                                    # 把数据提交给用户会话
                                    db.session.add(sign)
                                    # 提交会话
                                    db.session.commit()
                                    print('{}签到成功!!!'.format(stu.name))
                                except Exception as e:
                                    print(e)
                                    db.session.rollback()
                            
                            # 如果数据库中有数据(从start_sign_time到class_start_time之后2小时之间)      不写入
                            elif len(sign_t)>1:
                                for i in range(1,len(sign_t)):
                                    try:
                                        # 把数据提交给用户会话
                                        db.session.delete(sign_t[i])
                                        # 提交会话
                                        db.session.commit()
                                    except Exception as e:
                                        print(e)
                                        db.session.rollback()
                                print('{}同步数据成功'.format(stu.name))
                                print('{}已签到过'.format(stu.name))
                            
                            else:
                                print('{}已签到过'.format(stu.name))
                                
                        finally:
                            db.session.remove()
                    else:
                        print('{}识别成功，但不在可签到区间'.format(stu.name))
            
                lock.acquire()
                try:
                    get_sign_time()
                finally:
                    lock.release()

            

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
            # first_match_index
#             cv2.putText(frame_process, str(face_distances[first_match_index])[0:6], (left + int(w*0.2), bottom + int(h*0.1)), font, 0.5, (255,191,0), 1)
            # best_match_index
            cv2.putText(frame_process, str(face_distances[best_match_index])[0:6], (left + int(w*0.2), bottom + int(h*0.1)), font, 0.5, (255,191,0), 1)

        # Wait to write
        while Global.write_num != worker_id:
            time.sleep(0.01)

        # Send frame to global
        write_frame_list[worker_id] = frame_process

        # Expect next worker to write frame
        Global.write_num = next_id(Global.write_num, worker_num)

        
def load_face_encodings2MYSQL():
    '''
    加载人脸数据库(sno)到MYSQL
    注意格式：
    known_face_encodings => [ndarray(128,) * data_len]
    known_face_names => [str(name) * data_len]
    known_face_snos => [int(sno) * data_len]
    return known_images,known_face_encodings,known_face_names,known_face_snos
    '''
    known_images = []
    known_face_encodings = []
    known_face_names = []
    known_face_snos = []
    known_path = "C:/Users/14640/Pictures/known_sno_pictures/"
    imgnames = os.listdir(known_path)
    # print(imgnames)
    for i in range(len(imgnames)):
        print(i)
        print(known_path+imgnames[i])
        # Load a sample picture and learn how to recognize it.
        known_img = face_recognition.load_image_file(known_path+imgnames[i])
        known_face_encoding = face_recognition.face_encodings(known_img)
#         print('known_face_encoding[0].shape:',known_face_encoding[0].shape)
        
        # 如果不能检测到脸
        if known_face_encoding == []:
            print(known_path+imgnames[i]+' ---->>> 没有检测到人脸')
            raise Exception('此照片没有检测到人脸')
        known_face_encodings.append(known_face_encoding[0])
        known_images.append(known_img)
        # 获取学号sno
        sno = os.path.splitext(imgnames[i])[0]
        print('学号sno: ', sno)
        known_face_snos.append(int(sno))
        # 获取学生数据对象
        stu = Student.query.filter_by(sno=int(sno)).first()
        # 获取姓名
        china_name = stu.name
        print(china_name)
        # 获取拼音
        pinyin_name = pinyin.get(china_name, format="strip", delimiter=" ")
        print(pinyin_name)
        known_face_names.append(pinyin_name)
        # 将人脸特征编码存储在MYSQL中
        try:
            for i in range(128):
                setattr(stu, 'face_encoding%s' % i, known_face_encoding[0][i])
            # 提交会话
            db.session.commit()
            print('成功添加到数据库')
        except Exception as e:
            print(e)
            flash('添加人脸特征编码失败')
            # 回滚
            db.session.rollback()
            
    return known_images,known_face_encodings,known_face_names,known_face_snos
        
    
def load_face_encodings_MYSQL():
        '''
        从MYSQL中获取人脸数据库
        注意格式：
        known_face_encodings => [ndarray(128,) * data_len]
        known_face_names => [str(name) * data_len]
        known_face_snos => [int(sno) * data_len]
        return known_images,known_face_encodings,known_face_names,known_face_snos
        '''
        
        known_face_encodings = []
        known_face_names = []
        known_face_snos = []

        # 获取学生数据对象
        stus = Student.query.all()

        # 获取known_face_encodings,known_face_names
        for stu in stus:
            print('姓名：',stu.name)
            
            # 获取known_face_encoding,known_face_name
            known_face_encoding = []

            # 128组face_encoding
            names = locals()
            for i in range(128):
                fencoding = getattr(stu, 'face_encoding%s' % i)
                if fencoding != None:
                    known_face_encoding.append(fencoding)
                else:
                    print('人脸特征编码存在NULL')
                    # 存在None值抛出异常
                    raise Exception('人脸特征编码存在NULL')
            china_name = stu.name
            # 获取拼音
            pinyin_name = pinyin.get(china_name, format="strip", delimiter=" ")
            print(pinyin_name)
            
            known_face_snos.append(stu.sno)
            known_face_names.append(pinyin_name)
            known_face_encodings.append(np.asarray(known_face_encoding, dtype='float64'))
            
        print('成功从MYSQL中读取人脸数据')
        return known_face_encodings,known_face_names,known_face_snos
        

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000,threaded=True) 
    
    # 边框(透明png)
    bod = cv2.imread(r'C:\Users\14640\Pictures\bodimg\ds.png',-1)
    
    # 定义上课时间
    class_start_time = datetime.datetime.now().replace(hour=15, minute=0, second=0,microsecond=0)
    # 确定开始签到时间
    start_sign_time = datetime.datetime.now().replace(hour=14, minute=30, second=0,microsecond=0)
    
    # 加载人脸数据库(sno)到MYSQL
#     known_images,known_face_encodings,known_face_names,known_face_snos = load_face_encodings2MYSQL()
    
    # 从MYSQL中获取数据
    known_face_encodings,known_face_names,known_face_snos = load_face_encodings_MYSQL()
    
    # Fix Bug on MacOS
    if platform.system() == 'Darwin':
        set_start_method('forkserver')
        
    # 上传锁
    lock = Lock()

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
    p.append(threading.Thread(target=capture, args=(read_frame_list, Global, worker_num)))
    p[0].start()

    # Create arrays of known face encodings and their names
    Global.known_face_encodings = known_face_encodings
    Global.known_face_names = known_face_names
    Global.known_face_snos = known_face_snos
    Global.class_start_time = class_start_time
    Global.start_sign_time = start_sign_time
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
        p.append(Process(target=process, args=(worker_id, read_frame_list, write_frame_list, Global, worker_num,lock)))
        p[worker_id].start()
        
#     for item in p:
#         item.join()
        
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
