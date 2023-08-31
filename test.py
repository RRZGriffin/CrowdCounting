from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget,\
    QHBoxLayout, QVBoxLayout,  QTabWidget, QStackedWidget, QAction, QWidgetAction, QLabel, QFileDialog,\
    QGroupBox, QSpinBox, QComboBox, QLineEdit, QFormLayout, QDialogButtonBox, QTableView, QPushButton, QHeaderView, \
    QDateEdit, QCheckBox, QMessageBox, QGridLayout, QSpacerItem, QSizePolicy
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPixmap, QPalette, QColor, QFont, QRegExpValidator
from PyQt5 import QtGui, QtCore
import cv2
import numpy as np
import sys
import sqlite3
import datetime
import json

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
matplotlib.use('Qt5Agg')

from mylib import config, thread
from mylib.mailer import Mailer
from mylib.detection import detect_people
from imutils.video import VideoStream, FPS
from scipy.spatial import distance as dist
import numpy as np
import argparse, imutils, cv2, os, time, schedule

conn = sqlite3.connect('violations.db', check_same_thread=False)

cur = conn.cursor()

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot()
        super(MplCanvas, self).__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self ) -> None:
        super().__init__()
        self.setWindowTitle('Social Distancing Detection')
        self.setMinimumHeight(500)
        self.setMinimumWidth(700)
        self.showMaximized()

        self.tabsWidget = TabsWidget(self)
        self.setCentralWidget(self.tabsWidget)

class PlotWindow(QMainWindow):
    def __init__(self, parent=None):
        super(PlotWindow, self).__init__(parent)
        self.setMinimumHeight(200)
        self.setMinimumWidth(400)
        self.setWindowTitle('Graph')

        self.main_widget = QWidget(self)
        self.dateedit =QDateEdit(calendarPopup=True)
        self.dateedit.setDateTime(QtCore.QDateTime.currentDateTime())
        self.dateedit.setMaximumWidth(150)
        self.dateedit.dateChanged.connect(self.onDateChange)
        self.sc = MplCanvas(self, width=1, height=5, dpi=100)


        self.mainLayout = QVBoxLayout(self.main_widget)

        x, y  = self.getAllDataByDate()

        self.sc.axes.plot(x, y)
        self.sc.axes.set_xlabel('Time')
        self.sc.axes.set_ylabel('Violations')
        self.sc.axes.set_xticklabels(x, rotation=45)
        self.sc.axes.grid()
        # self.setLayout(self.layout)
        self.mainLayout.addWidget(self.dateedit)
        self.mainLayout.addWidget(self.sc)
        self.setCentralWidget(self.main_widget)
    
    def getAllDataByDate(self):
        format = "%Y-%m-%d %H:%M:%S"
        res = cur.execute('SELECT violations, created_at FROM violations')
        violations = res.fetchall()
        x = []
        y = []

        # print(violations)
        selectedDate = self.dateedit.dateTime().date().toPyDate()
        for violation, created_at in violations:
            date:str = created_at.split('.')[0]
            formatted_time = datetime.datetime.strptime(date, format)
            if selectedDate == formatted_time.date():
                x.append(formatted_time.strftime('%H:%M'))
                y.append(violation)
        
        print(violations)
        return (x, y)
    
    def onDateChange(self):
        x, y = self.getAllDataByDate()
        self.sc.axes.clear()
        self.sc.axes.plot(x,y)
        self.sc.axes.set_xticklabels(x, rotation=45)
        self.sc.axes.grid()
        self.sc.draw()

class TabsWidget(QWidget):
    activeButtonSignal = pyqtSignal(bool)

    def __init__(self,  parent) -> None:
        super().__init__()
        self.parent = parent
        self.currentStudentID = None

        self.layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()


        self.tabs.addTab(self.tab1, 'Social Distancing')
        self.tabs.addTab(self.tab2, 'Data')


        self.initStudentTab()
        self.initDatabaseTab()


        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
    
    def initStudentTab(self):
        self.isActive = False
        self.disply_width = 640
        self.display_height = 480
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # self.studentsEnteredData.setFont(QFont('Arial', 42))
  

        self.tab1.layout = QHBoxLayout()
        self.tab1.layout.addWidget(self.image_label)



        self.tab1.setLayout(self.tab1.layout)

        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.violationsSignal.connect(self.updateViolations)
        self.activeButtonSignal.connect(self.thread.buttonActive)
        self.thread.start()

        self.tab1Vertical = QVBoxLayout()
        self.tab1Grid = QGridLayout()

        self.startBtn = QPushButton('Start')
        self.startBtn.clicked.connect(self.gatherVideoOutput)
        # self.stopBtn = QPushButton('Stop')
        self.openGraphBtn = QPushButton('View Graph')
        self.violationsLabel = QLabel('Violations')
        self.violationsLabel.setStyleSheet("border: 3px solid red;")
        self.violationsLabel.setMaximumHeight(50)
        self.violationsLabel.setAlignment(Qt.AlignCenter)

        self.tab1Vertical.addWidget(self.startBtn)
        # self.tab1Vertical.addWidget(self.stopBtn)
        self.tab1Vertical.addWidget(self.openGraphBtn)
        self.tab1Vertical.addWidget(QLabel(''))
        self.tab1Vertical.addWidget(self.violationsLabel)

        self.openGraphBtn.clicked.connect(self.on_pushButton_clicked)

        self.dialog = PlotWindow(self)

        self.tab1.layout.addLayout(self.tab1Vertical)
    
    def gatherVideoOutput(self):
        self.isActive = (not self.isActive)
        if self.isActive:
            self.startBtn.setText('STOP')
            self.activeButtonSignal.emit(self.isActive)
            # self.startBtn.styleSheet("QPushButton:hover { background-color: lightgreen; };")
        else:
            self.startBtn.setText('Start')
            self.activeButtonSignal.emit(self.isActive)
    
    def initDatabaseTab(self):
        data = cur.execute("SELECT * FROM violations").fetchall()
        if len(data) == 0:
            data = None
        
        self.table = QTableView()
        self.model = TableModel(data)
        self.table.setModel(self.model)
        self.table.setSortingEnabled(True)
        self.table.setMaximumWidth(480)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        self.tab2.layout = QVBoxLayout()
        self.tab2.layout.addWidget(self.table)
        self.tab2.layout.setAlignment(QtCore.Qt.AlignCenter)
        self.tab2.setLayout(self.tab2.layout)

    
    @pyqtSlot(int)
    def updateViolations(self, violations):
        # print('violations: ',violations)
        self.violationsLabel.setText(f'Violations: {violations}')

    def on_pushButton_clicked(self):
        self.dialog.show()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)



class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    violationsSignal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.violationsEvery10Mins = 0
        self.warningsEvery10Mins = 0
        self.startTime = time.time()
        self.isActive = False
    
    @pyqtSlot(bool)
    def buttonActive(self, isActive):
        print(isActive)
        self.isActive = isActive;

    def run(self):

        #----------------------------Parse req. arguments------------------------------#
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--input", type=str, default="",
            help="path to (optional) input video file")
        ap.add_argument("-o", "--output", type=str, default="",
            help="path to (optional) output video file")
        ap.add_argument("-d", "--display", type=int, default=1,
            help="whether or not output frame should be displayed")
        args = vars(ap.parse_args())
        #------------------------------------------------------------------------------#

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
        configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # check if we are going to use GPU
        if config.USE_GPU:
            # set CUDA as the preferable backend and target
            print("")
            print("[INFO] Looking for GPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # if a video path was not supplied, grab a reference to the camera
        if not args.get("input", False):
            print("[INFO] Starting the live stream..")
            vs = cv2.VideoCapture(config.url)
            if config.Thread:
                    cap = thread.ThreadingClass(config.url)
            time.sleep(2.0)

        # otherwise, grab a reference to the video file
        else:
            print("[INFO] Starting the video..")
            vs = cv2.VideoCapture(args["input"])
            if config.Thread:
                    cap = thread.ThreadingClass(args["input"])

        writer = None
        # start the FPS counter
        fps = FPS().start()

        # loop over the frames from the video stream
        while True:
            # read the next frame from the file
            time_elapsed = time.time() - self.startTime
            if (time_elapsed/60) >= 10 and self.isActive:
                createdAt = datetime.datetime.now()
                params = (self.violationsEvery10Mins, self.warningsEvery10Mins, createdAt)
                cur.execute("INSERT INTO violations VALUES (NULL, ?, ?, ?)", params);
                conn.commit()
                self.violationsEvery10Mins = 0
                self.warningsEvery10Mins = 0
                self.startTime = time.time()
                print('time: ', time_elapsed)

            if config.Thread:
                frame = cap.read()

            else:
                (grabbed, frame) = vs.read()
                # if the frame was not grabbed, then we have reached the end of the stream
                if not grabbed:
                    break

            # resize the frame and then detect people (and only people) in it
            frame = imutils.resize(frame, width=700)
            results = detect_people(frame, net, ln,
                personIdx=LABELS.index("person"))

            # initialize the set of indexes that violate the max/min social distance limits
            serious = set()
            abnormal = set()

            # ensure there are *at least* two people detections (required in
            # order to compute our pairwise distance maps)
            if len(results) >= 2:
                # extract all centroids from the results and compute the
                # Euclidean distances between all pairs of the centroids
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                # loop over the upper triangular of the distance matrix
                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                        # check to see if the distance between any two
                        # centroid pairs is less than the configured number of pixels
                        if D[i, j] < config.MIN_DISTANCE:
                            # update our violation set with the indexes of the centroid pairs
                            serious.add(i)
                            serious.add(j)
                        # update our abnormal set if the centroid distance is below max distance limit
                        if (D[i, j] < config.MAX_DISTANCE) and not serious:
                            abnormal.add(i)
                            abnormal.add(j)

            # loop over the results
            for (i, (prob, bbox, centroid)) in enumerate(results):
                # extract the bounding box and centroid coordinates, then
                # initialize the color of the annotation
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

                # if the index pair exists within the violation/abnormal sets, then update the color
                if i in serious:
                    color = (0, 0, 255)
                elif i in abnormal:
                    color = (0, 255, 255) #orange = (0, 165, 255)

                # draw (1) a bounding box around the person and (2) the
                # centroid coordinates of the person,
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 2)

            # draw some of the parameters
            Safe_Distance = "Safe Distance: >{} px".format(config.MAX_DISTANCE)
            cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
            Threshold = "Threshold: {}".format(config.Threshold)
            cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

            # draw the total number of social distancing violations on the output frame
            text = "Violations: {}".format(len(serious))
            if self.isActive:
                self.violationsEvery10Mins = max(self.violationsEvery10Mins, len(serious))
                self.violationsSignal.emit(len(serious))
            cv2.putText(frame, text, (10, frame.shape[0] - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

            text1 = "Warning: {}".format(len(abnormal))
            cv2.putText(frame, text1, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
            if self.isActive:
                self.warningsEvery10Mins = max(self.warningsEvery10Mins, len(abnormal))

        #------------------------------Alert function----------------------------------#
            if len(serious) >= config.Threshold:
                cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
                    cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
                if config.ALERT:
                    print("")
                    print('[INFO] Sending mail...')
                    Mailer().send(config.MAIL)
                    print('[INFO] Mail sent')
                #config.ALERT = False
        #------------------------------------------------------------------------------#
            # check to see if the output frame should be displayed to our screen
            if args["display"] > 0:
                # show the output frame
                # cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
                self.change_pixmap_signal.emit(frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break
            # update the FPS counter
            fps.update()

            # if an output video file path has been supplied and the video
            # writer has not been initialized, do so now
            if args["output"] != "" and writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 25,
                    (frame.shape[1], frame.shape[0]), True)

            # if the video writer is not None, write the frame to the output video file
            if writer is not None:
                writer.write(frame)

        # stop the timer and display FPS information
        fps.stop()
        # print("===========================")
        # print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
        # print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

        # close any open windows
        cv2.destroyAllWindows()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data, columns=['Index', 'Violations', 'Warnings', 'Date']):
        super(TableModel, self).__init__()
        self._data = data
        self._columns = columns

    def data(self, index, role):
        if not self._data:
            return 0
        if role == QtCore.Qt.DisplayRole:
            value = self._data[index.row()][index.column()]
            # if self.is_date(value):
            #     value = value.strftime("%m/%d/%Y, %H:%M:%S")
            return value


    def rowCount(self, index):
        if not self._data:
            return 0
        return len(self._data)

    def columnCount(self, index):
        if not self._data:
            return len(self._columns)
        return len(self._data[0])

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._columns[section]

            # if orientation == QtCore.Qt.Vertical:
            #     return str(self._data.index[section])
    def sort(self, col, order):
        if not self._data:
            return None
        if order == 0:
            self._data = sorted(self._data, key=lambda x: x[col],reverse=True)
        if order == 1:
            self._data = sorted(self._data, key=lambda x: x[col])
        self.layoutChanged.emit()
    
    def is_date(self, value, fuzzy=False):
        try:
            datetime.datetime.strptime(str(value), '%Y-%m-%d')
            print(value)
            return True
        except Exception:
            return False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet("QLabel{font-size: 12pt;} QGroupBox{font-size: 14pt;}\
                       QComboBox{height: 20} QLineEdit{height: 20}\
                       QSpinBox{height:20}")
    window = MainWindow()

    sys.exit(app.exec())