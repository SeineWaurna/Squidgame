import sys
from PyQt5 import QtWidgets
from MainFrame import Ui_MainFrame
from ImageQT import ImageQT
from PyQt5.QtCore import QTimer
import cv2
from datetime import datetime
from random import shuffle
from ultralytics import YOLO
import pygame
import time as ltime


def playsound(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

class Counter:
    def __init__(self):
        self.n_count = 0
    
    def count(self):
        self.n_count += 1


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_window = Ui_MainFrame()
        self.main_window.setupUi(self)
        self.model = YOLO("best (1).pt")

        self.timer = QTimer()
        self.timer.timeout.connect(self.stream)
        self.timer.start(20)

        self.main_window.start_button.clicked.connect(self.start_button_clicked)

        self.cam = cv2.VideoCapture(0)
        self.main_window.time_label.setText(f"00:00")
        # time.sleep(5)

        self.predicted_images = [
            cv2.imread('c.jpg'),
            cv2.imread('s.jpg'),
            cv2.imread('t.jpg'),
            cv2.imread('u.jpg'),
        ]

        self.image_labels = [
            self.main_window.predicted1_label,
        ]
        self.current_predicted = 0
        self.empty_img = cv2.imread("empty.jpg")

        self.images = [
            (cv2.imread('c1.png'),0),
            (cv2.imread('s1.png'),1),
            (cv2.imread('t1.png'),2),
            (cv2.imread('u1.jpg'),3),
            (cv2.imread('c1.png'),0),
            (cv2.imread('s1.png'),1),
            (cv2.imread('t1.png'),2),
            (cv2.imread('u1.jpg'),3),
            (cv2.imread('c1.png'),0),
            (cv2.imread('s1.png'),1),
            (cv2.imread('t1.png'),2),
            (cv2.imread('u1.jpg'),3),
            (cv2.imread('c1.png'),0),
            (cv2.imread('s1.png'),1),
            (cv2.imread('t1.png'),2),
            (cv2.imread('u1.jpg'),3),
        ]

        self.show_image_label()

        self.last_time = datetime.now()
        self.threshold = 2.6
        self.wait_time = self.threshold
        self.step = 0.01
        self.counter = Counter()
    

        self.run_time = 65 # second
        self.start_time = None
        self.flag_start = False
        self.last_cls = None

    def show_image_label(self):
        for ind in range(1):
            image = self.images[ind][0]
            if ind == 0:
                ImageQT.addToQT(image, self.main_window.image1_label, (400,400))
            elif ind == 1:
                ImageQT.addToQT(image, self.main_window.image1_label, (400,400))
            elif ind == 2:
                ImageQT.addToQT(image, self.main_window.image1_label, (400,400))
            elif ind == 3:
                ImageQT.addToQT(image, self.main_window.image1_label, (400,400))
    
    def get_class(self):
        txt = open("tmp.txt").read()
        if txt != "":
            cls = int(txt.strip())
            return cls
        return -1

    def reset_state(self):
        self.flag_start = False
        self.current_predicted = 0
        self.start_time = None

    def stream(self):

        ret, frame = self.cam.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()



        sizes = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [int(k) for k in boxes[i]]
            sizes.append( (y2-y1)*(x2-x1) )

        conf = None
        name = None
        if len(sizes) > 0:
            ind_max = sizes.index(max(sizes))
            box = boxes[ind_max]
            x1, y1, x2, y2 = [int(k) for k in box]
            name = self.model.names[classes[i]]
            conf = confs[i]
            # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
            # cv2.putText(frame, f"{name}:{conf*100:.2f}%", (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 1)

        if True:
            

            if self.wait_time < self.threshold:
                if (datetime.now() - self.last_time).total_seconds() >= 0.08:
                    shuffle(self.images)
                    self.show_image_label()
                    self.last_time = datetime.now()
                    self.wait_time += self.step
                    self.step *= 1
                    self.main_window.status_label.setText("Waiting ...")
            else:
                
                if self.flag_start:
                    if self.start_time is None:
                        self.start_time = datetime.now()
                        playsound("Squid Game 2 OST - Round the Circle I.mp3")

                    self.main_window.status_label.setText("Start!")
                    
                    time = self.run_time-(datetime.now()-self.start_time).total_seconds()
                    if time > 0:
                        self.main_window.time_label.setText(f"{time:.2f}")

                        if name is not None:
                            if name == 'circle':
                                cls = 0
                            elif name == 'triangle':
                                cls = 2
                            elif name == 'star':
                                cls = 1
                            elif name == 'umbellar':
                                cls = 3
                        else:
                            cls = -1
                        if cls == self.images[self.current_predicted][1]:
                            if self.counter.n_count >= 30:
                                ImageQT.addToQT(self.predicted_images[cls], self.image_labels[0], (400,400))
                                self.main_window.class_label.setText(name)
                                self.main_window.conf_label.setText(f"{conf*100:.2f}%")
                                self.current_predicted += 1
                                self.counter.n_count = 0
                                self.last_cls = cls
                                
                                if self.current_predicted < 10:
                                    while self.images[self.current_predicted][1] == cls:
                                        shuffle(self.images)
                                    ImageQT.addToQT(self.images[self.current_predicted][0], self.main_window.image1_label, (400,400))
                            else:
                                self.counter.n_count += 1
                            color = (0,255,0)
                        else:
                            self.counter.n_count = 0
                            color = (0,0,255)
                        if cls >= 0:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
                            cv2.putText(frame, f"{name}:{conf*100:.2f}%", (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 1)

                        if self.current_predicted == 10:
                            self.reset_state()
                            self.main_window.status_label.setText("Win")
                            playsound("Victory Sound Effect.mp3")
                            

                    else:
                        self.reset_state()
                        self.main_window.time_label.setText(f"00:00")
                        self.main_window.status_label.setText("Loss")
                        playsound("Desert Eagle Single Shot Gunshot Sound Effect.mp3")
                        
            ImageQT.addToQT(frame, self.main_window.realtime_label, (860,650))
    def start_button_clicked(self):
        playsound("Mingle Game Song Round and Round Lyric Video  Squid Game_ Season 2  Netflix.mp3")
        self.reset_state()
        self.counter.n_count = 0
        for i in range(len(self.image_labels)):
            ImageQT.addToQT(self.empty_img, self.image_labels[i], (400,400))
        self.main_window.time_label.setText("00:00")
        self.wait_time = 0
        self.step = 0.01
        self.flag_start = True
        self.color = (0,0,255)

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.showFullScreen()
    sys.exit(app.exec_())
