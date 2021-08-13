from pathlib import Path

import os
import datetime
import argparse
import sys
import time
import pdb
import json

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from numpy.matlib import repmat

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import seaborn as sns
import pandas as pd

from GUI.dataload import *
from GUI.keys import *
from GUI.widgets import *
from GUI.userlog import *

LEN_EPOCH = 3000
POINTFIVE = 0.5
POINTSEVEN = 0.7
POINTONE = 0.1


class Data_Loader():
    def __init__(self,
                 paths_files,
                 subject,
                 start_epoch,
                 exp_start_epoch,
                 data_channel,
                 target_model,
                 num_epochs):

        self.subject = subject
        self.start_epoch = start_epoch
        self.target_model = target_model
        self.DATA_CHANNEL = data_channel
        self.num_epochs = num_epochs

        self.DATA_LIST = load_data_from_path(
            Path(paths_files['data']), 
            subject, 
            start_epoch, 
            data_channel
        )
        self.GRAD, self.PRED = load_grad_pred(
            [Path(paths_files['grad']), Path(paths_files['pred'])],
            target_model, 
            subject, 
            start_epoch
        )
        self.LABEL, self.LABEL2 = load_labels(
            [Path(paths_files['label1']), Path(paths_files['label2'])],
            subject, 
            start_epoch
        )
        activation = load_activation(
            Path(paths_files['features']),
            subject,
            start_epoch
        )
        if self.GRAD is not None:
            self.GRAD = self.GRAD[exp_start_epoch:exp_start_epoch + num_epochs]
        else:
            self.GRAD = np.random.rand(num_epochs, LEN_EPOCH)

        if self.PRED is not None:
            self.PRED = self.PRED[exp_start_epoch:exp_start_epoch + num_epochs]
        else:
            self.PRED = np.random.randint(5, size=num_epochs)
        
        if self.LABEL is not None:
            self.LABEL = self.LABEL[exp_start_epoch:exp_start_epoch + num_epochs]
        else:
            self.LABEL = np.random.randint(5, size=num_epochs)

        if self.LABEL2 is not None:
            self.LABEL2 = self.LABEL2[exp_start_epoch:exp_start_epoch + num_epochs]
        else:
            self.LABEL2 = np.random.randint(5, size=num_epochs)

        for i in range(len(self.DATA_LIST)):
            self.DATA_LIST[i] = self.DATA_LIST[i][exp_start_epoch*LEN_EPOCH:(
                exp_start_epoch + num_epochs)*LEN_EPOCH]

        self.activation = {}
        if activation is not None:
            for key in activation.keys():
                self.activation[key] = activation[key][exp_start_epoch:exp_start_epoch + num_epochs]
        else:
            self.activation = {'spindles', 'complexes', 'alphas', 'deltas', 'sawtooth'}
            for key in activation.keys():
                self.activation[key] = np.random.rand(num_epochs)


class UI_MainWindow():
    def setupUI(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1800, 1500)

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.main_layout = QVBoxLayout()
        self.main_layout.setObjectName("main_layout")

        self.scoring = Scoring(self)
        self.filter = Filters(self, MainWindow.session['exp_filter_path'], [])
        self.hypnogram = Hypnogram(self)
        self.spectrogram = Spectrogram_Docker_Widget(self)
        self.annotate = Annotate(self)
        self.model_pred_label = Label_Docker_Widget(self)

        new_docks = [{'name': 'Scoring',
                      'widget': self.scoring,
                      'main_area': Qt.RightDockWidgetArea,
                      'extra_area': Qt.LeftDockWidgetArea,
                     },
                     {'name': 'Filters',
                      'widget': self.filter,
                      'main_area': Qt.LeftDockWidgetArea,
                      'extra_area': Qt.RightDockWidgetArea,
                     },
                     {'name': 'Hypnogram',
                      'widget': self.hypnogram,
                      'main_area': Qt.TopDockWidgetArea,
                      'extra_area': Qt.BottomDockWidgetArea,
                     },
                     {'name': 'Annotate',
                      'widget': self.annotate,
                      'main_area': Qt.RightDockWidgetArea,
                      'extra_area': Qt.LeftDockWidgetArea,
                     }
                    ]

        self.idx_docks = {}
        for dock in new_docks:
            dockwidget = QDockWidget(dock['name'], self)
            dockwidget.setWidget(dock['widget'])
            dockwidget.setAllowedAreas(dock['main_area'] | dock['extra_area'])
            dockwidget.setObjectName(dock['name'])

            self.addDockWidget(dock['main_area'], dockwidget)
            self.idx_docks[dock['name']] = dockwidget

        self.idx_docks['Filters'].setFloating(True)
        self.idx_docks['Filters'].resize(QSize(1800, 1000))
        self.idx_docks['Filters'].hide()

        self.centralwidget.setLayout(self.main_layout)
        self.setCentralWidget(self.centralwidget)

        self.create_menubar()

    def beta_dock_widget(self):
        self.acc_dist_disp = Acc_Dist_Disp(self)
        acc_dist_dockwidget = QDockWidget('Accuracy & Class Distribution',
            self)
        acc_dist_dockwidget.setWidget(self.acc_dist_disp)
        acc_dist_dockwidget.setAllowedAreas(Qt.RightDockWidgetArea)
        acc_dist_dockwidget.setObjectName('Accuracy & Class Distribution')

        self.addDockWidget(Qt.RightDockWidgetArea, acc_dist_dockwidget)

    def create_toolbar(self):
        toolbar = self.addToolBar('scope')
        zoom_in = QAction(QIcon(
            'CHI_gui/buttons/zoom_in.png'),'zoom_in',self)
        zoom_in.triggered.connect(self.zoom_in_callback)
        toolbar.addAction(zoom_in)

        zoom_out = QAction(QIcon(
            'CHI_gui/buttons/zoom_out.png'),'zoom_out',self)
        zoom_out.triggered.connect(self.zoom_out_callback)
        toolbar.addAction(zoom_out)

        stretch_x = QAction(QIcon(
            'CHI_gui/buttons/leftright.png'),'stretch_x',self)
        stretch_x.triggered.connect(self.stretch_x_callback)
        toolbar.addAction(stretch_x)

        stretch_y = QAction(QIcon(
            'CHI_gui/buttons/updown.png'),'stretch_y',self)
        stretch_y.triggered.connect(self.stretch_y_callback)
        toolbar.addAction(stretch_y)

    def create_menubar(self):
        menubar = self.menuBar()
        widgetmenu = menubar.addMenu('Widgets')

        action = QAction('Filters', self)
        action.triggered.connect(self._open_filters)
        widgetmenu.addAction(action)

        action = QAction('Scoring', self)
        action.triggered.connect(self._open_scoring)
        widgetmenu.addAction(action)

        action = QAction('Hypnogram', self)
        action.triggered.connect(self._open_hypnogram)
        widgetmenu.addAction(action)

        action = QAction('Annotate', self)
        action.triggered.connect(self._open_annotate)
        widgetmenu.addAction(action)

    def _open_filters(self):
        print(self.idx_docks['Filters'])
        self.idx_docks['Filters'].setFloating(True)
        self.idx_docks['Filters'].show()

    def _open_scoring(self):
        print(self.idx_docks['Scoring'])
        self.idx_docks['Scoring'].show()

    def _open_hypnogram(self):
        print(self.idx_docks['Hypnogram'])
        self.idx_docks['Hypnogram'].show()

    def _open_annotate(self):
        print(self.idx_docks['Annotate'])
        self.idx_docks['Annotate'].show()


class MainWindow(QMainWindow, UI_MainWindow):
    def __init__(self, session, dataloader, user_num, session_num):
        super(MainWindow, self).__init__()
        self.session = session
        self.user_num = user_num
        self.dataloader = dataloader
        self.session_num = session_num
        self.DATA_CHANNEL = dataloader.DATA_CHANNEL
        self.LEN_EPOCH = LEN_EPOCH
        self.setupUI(self)
        self.NROW = len(self.dataloader.DATA_LIST)
        self.rescale_data(rescale_const=12)
        
        self.channel_height = 40
        self.channel_ratio = 2.5

        self.key_logs = []
        self.insert_buffer = []
        self.insert_R_buffer = []
        self.insert_T_buffer = []
        self.insert_mode = False
        self.insert_R_mode = False
        self.insert_T_mode = False
        self.annotate_mode = False
        self.annotate_local_buffer = None
        self.user_annotation_record = (np.ones(self.dataloader.LABEL.shape)
            *(-1))
        self.pen = pg.mkPen(color='k', width=0.5)
        self.status = {'bool_stat': {'signal': False,
                                     'fill': True,
                                     'gradient': False},
                       'counter': 0,
                       'calib': 8,
                       'X': False,
                       'Y': False
                      }
        self.start_timepoint = None
        self.create_hypnogram()
        self.create_item_list()
        self.create_signal()
        self.make_signal()
        self.make_fill()
        self.setWindowTitle("Sleep Staging Interaction Tool")
        self.activation = self.dataloader.activation

    def rescale_data(self, rescale_const):
        for row in range(self.NROW):
            self.dataloader.DATA_LIST[row] = (
                self.dataloader.DATA_LIST[row]*rescale_const)

    def get_main_channel_idx(self):
        for idx, channel in enumerate(self.dataloader.DATA_CHANNEL):
            if channel=='C3_A2':
                main_channel_idx = idx

        return main_channel_idx

    def create_signal(self):
        self.ax_list = []
        for row in range(self.NROW):
            self.ax_list.append(pg.PlotWidget())
            self.ax_list[row].setBackground('w')
            self.ax_list[row].showGrid(True, True, alpha=1)
            self.main_layout.addWidget(self.ax_list[row])
            xaxis = self.ax_list[row].getAxis('bottom')
            
            if row==self.get_main_channel_idx()-1:
                self.pred_label = Model_Pred_Label(self)
                self.main_layout.addWidget(self.pred_label)
            elif row==self.get_main_channel_idx():
                self.activation_ax = pg.PlotWidget()
                self.activation_ax.setBackground('w')
                self.activation_item_list = []
                self.main_layout.addWidget(self.activation_ax)
                self.activation_ax.setLabel('left',
                    '{}'.format('Activation'))
            self.ax_list[row].setYRange(
                np.min(self.dataloader.DATA_LIST[row]),
                np.max(self.dataloader.DATA_LIST[row])
                )

    def create_item_list(self):
        self.item_list = []
        for row in range(self.NROW):
            self.item_list.append({'signal': None, 
                                   'fill': None,
                                   'gradient': None}
                                 )

    def create_hypnogram(self):
        self.hypnogram_ax = pg.PlotWidget()
        self.hypnogram_ax.setBackground('w')
        self.hypnogram_ax.showGrid(False, True)
        self.hypnogram.layout.addWidget(self.hypnogram_ax)

    def create_spectrogram(self):
        self.spectrogram_list = []
        for row in range(self.NROW):
            spec_pg_widget = Spectrogram(self)
            self.spectrogram.layout.addWidget(spec_pg_widget)
            self.spectrogram_list.append(spec_pg_widget)

    def create_model_pred_label(self):
        self.pred_label = Model_Pred_Label(self)
        self.model_pred_label.layout.addWidget(QWidget(), 0, 1, 1, 1)
        self.model_pred_label.layout.addWidget(self.pred_label, 0, 2, 1, 1)
        self.model_pred_label.layout.addWidget(QWidget(), 0, 3, 1, 1)

    def keyPressEvent(self, event):
        sleep_level = ['W', 'N1', 'N2', 'N3', 'REM']
        self.key_logs.append(event.key())

        if not (self.insert_mode 
                or self.insert_R_mode 
                or self.insert_T_mode 
                or self.annotate_mode):
            if event.key() == Qt.Key_QuoteLeft:
                if not self.start_timepoint:
                    if self.session['exp_target_model'] in ['DeepSleepNet', 'DeepStrideNet']:
                        status = {'bool_stat': {'signal': True,
                                                'fill': False,
                                                'gradient': True},
                                  'counter': 0,
                                  'calib': 8,
                                  'X': True,
                                  'Y': True
                                 }
                    else:
                        status = {'bool_stat': {'signal': True,
                                                'fill': False,
                                                'gradient': False},
                                  'counter': 0,
                                  'calib': 8,
                                  'X': True,
                                  'Y': True
                                 }

                    self.start_timepoint = time.time()
                    self.log = User_Log(self)
                    print(status)
                    self.result_data_callback(status)

            elif event.key() == Qt.Key_Up:
                self.channel_height = self.channel_height-10
                self.log.event_append('{}'.format(35))
                status = self.status
                self.result_data_callback(status)
            elif event.key() == Qt.Key_Down:
                self.channel_height = self.channel_height + 10
                self.log.event_append('{}'.format(36))
                status = self.status
                self.result_data_callback(status)
            elif event.key() == Qt.Key_Left:
                if self.status['counter']>0:
                    self.backskip_annotate()
            elif event.key() == Qt.Key_Right:
                if self.status['counter']<len(self.dataloader.LABEL)-1:
                    self.skip_annotate()
            elif event.key() == Qt.Key_Delete:
                self.annotate_mode = True
                self.pred_label.setStyleSheet('color: gray')
                print('insert_mode')
            elif event.key() == Qt.Key_S:
                status = self.status
                status['bool_stat']['signal'] = (
                                            not status['bool_stat']['signal'])
                self.result_data_callback(status)
            elif event.key() == Qt.Key_F:
                self.fill_callback()
            elif event.key() == Qt.Key_G:
                status = self.status
                status['bool_stat']['gradient'] = (
                                        not status['bool_stat']['gradient'])
                self.result_data_callback(status)
            elif event.key() == Qt.Key_M:
                self.pred_label.setStyleSheet('color: black')
            elif event.key() == Qt.Key_I:
                self.log.event_append('{}'.format(31))
                self.zoom_in_callback()
            elif event.key() == Qt.Key_O:
                self.log.event_append('{}'.format(32))
                self.zoom_out_callback()
            elif event.key() == Qt.Key_X:
                self.log.event_append('{}'.format(33))
                self.stretch_x_callback()
            elif event.key() == Qt.Key_Y:
                self.log.event_append('{}'.format(34))
                self.stretch_y_callback()
            elif event.key() == Qt.Key_N:
                self.insert_mode = True
            elif event.key() == Qt.Key_R:
                self.insert_R_mode = True
            elif event.key() == Qt.Key_T:
                self.insert_T_mode = True
            elif event.key() == Qt.Key_P:
                self.filter.peaks = not (self.filter.peaks)
            elif event.key() in [Qt.Key_Return]:
                if self.session['exp_target_model'] in ['DeepSleepNet', 'DeepStrideNet']:
                    self.confirm_annotate()
            elif event.key() == Qt.Key_Q:
                self.log.event_append('{}'.format(99))
                self.log.close_line()
                print(len(self.user_annotation_record))
                if len(self.user_annotation_record)==len(self.dataloader.LABEL):
                    string = time.strftime('%Y%m%d %I%M%S', time.localtime(self.start_timepoint))
                    path = os.path.join('user_logs',
                                '{}'.format(self.user_num))
                    if not os.path.exists(path):
                        os.mkdir(path)
                    path = os.path.join('user_logs',
                                '{}'.format(self.user_num),
                                '{}'.format(self.session_num))
                    if not os.path.exists(path):
                        os.mkdir(path)
                    exp_result = {'Loaded data': self.session['exp_path']['data'],
                                  'subject': self.dataloader.subject,
                                  'data load start epoch number': self.dataloader.start_epoch,
                                  'exp start epoch number': int(self.session['exp_data'][2]),
                                  'user-rater1 f1_score': f1_score(self.dataloader.LABEL, self.user_annotation_record, average='macro'),
                                  'user-rater2 f1_score': f1_score(self.dataloader.LABEL2, self.user_annotation_record, average='macro'),
                                  'user-rater1 acc': np.sum(self.dataloader.LABEL==self.user_annotation_record)/len(self.dataloader.LABEL),
                                  'user-rater2 acc': np.sum(self.dataloader.LABEL2==self.user_annotation_record)/len(self.dataloader.LABEL2),
                                  'AI-user': f1_score(self.dataloader.PRED, self.user_annotation_record, average='macro'),
                                  'AI-rater1': f1_score(self.dataloader.LABEL, self.dataloader.PRED, average='macro'),
                                  'AI-rater2': f1_score(self.dataloader.LABEL2, self.dataloader.PRED, average='macro'),
                                  'rater1-rater2': f1_score(self.dataloader.LABEL, self.dataloader.LABEL2, average='macro'),
                                  'AI-user acc': f1_score(self.dataloader.PRED, self.user_annotation_record, average='micro'),
                                  'AI-rater1 acc': f1_score(self.dataloader.LABEL, self.dataloader.PRED, average='micro'),
                                  'AI-rater2 acc': f1_score(self.dataloader.LABEL2, self.dataloader.PRED, average='micro'),
                                  'rater1-rater2 acc': f1_score(self.dataloader.LABEL, self.dataloader.LABEL2, average='micro'),
                                  'Model': self.session['exp_target_model']
                                  }
                    with open(os.path.join(path, 'result.json'), 'w') as json_file:
                        json.dump(exp_result, json_file, indent=4)
                    np.save(os.path.join(path, 'user_annotation'), self.user_annotation_record)
                    self.log.f_events.close()
                    self.log.f_time.close()
                self.filter.close()
                self.close()

        elif self.insert_R_mode:
            if event.key() in NUMBER_KEYS:
                self.insert_R_buffer.append('{}'.format(event.key()-48))
            if event.key() == Qt.Key_R:
                seperator = ''
                if self.insert_R_buffer:
                    self.channel_ratio = int(
                        seperator.join(self.insert_R_buffer))
                    self.result_data_callback(self.status)
                    self.insert_R_buffer = []
                    self.insert_R_mode = False
                else:
                    self.insert_R_mode = False

        elif self.insert_T_mode:
            if event.key() in NUMBER_KEYS:
                self.insert_T_buffer.append('{}'.format(event.key()-48))
            if event.key() == Qt.Key_T:
                seperator = ''
                if self.insert_T_buffer:
                    self.channel_height = int(
                        seperator.join(self.insert_T_buffer))
                    self.result_data_callback(self.status)
                    self.insert_T_buffer = []
                    self.insert_T_mode = False
                else:
                    self.insert_T_mode = False

        elif self.annotate_mode:
            if event.key() == Qt.Key_0:
                self.annotate_local_buffer = 0
                self.timer_user_buffer_record()
            elif event.key() == Qt.Key_1:
                self.annotate_local_buffer = 1
                self.timer_user_buffer_record()
            elif event.key() == Qt.Key_2:
                self.annotate_local_buffer = 2
                self.timer_user_buffer_record()
            elif event.key() == Qt.Key_3:
                self.annotate_local_buffer = 3
                self.timer_user_buffer_record()
            elif event.key() in [Qt.Key_4, Qt.Key_5]:
                self.annotate_local_buffer = 4
                self.timer_user_buffer_record()

            if event.key() == Qt.Key_Delete:
                self.annotate_local_buffer = None
                self.annotate_mode = False
                self.pred_label.setStyleSheet('color: black')
                self.set_model_pred_label()
            elif event.key() in [Qt.Key_Return, Qt.Key_Enter]:
                if self.annotate_local_buffer is not None:
                    status = self.status
                    self.log.scoring = ('D_'
                        + sleep_level[self.annotate_local_buffer])
                    self.scoring.table.setItem(status['counter'], 2,
                        QTableWidgetItem(
                            sleep_level[self.annotate_local_buffer]))
                    self.user_annotation_record[status['counter']] = (
                        self.annotate_local_buffer)
                    self.log.event_append('{}'.format(0))
                    self.timer_user_record()
                    # self.skip_annotate()
                    self.annotate_local_buffer = None
                    self.annotate_mode = False
                    self.pred_label.setStyleSheet('color: black')
                else:
                    self.annotate_mode = False
                    self.pred_label.setStyleSheet('color: black')

        else:
            if event.key() in NUMBER_KEYS:
                self.insert_buffer.append('{}'.format(event.key()-48))
            if event.key() == Qt.Key_N:
                seperator = ''
                if self.insert_buffer:
                    self.log.event_append('{}'.format(30))
                    callback_num = int(seperator.join(self.insert_buffer))
                    status = self.status
                    self.log.close_line()
                    status['counter'] = callback_num - 1
                    self.result_data_callback(status)
                    self.log.add_new_line()
                    self.insert_buffer = []
                    self.insert_mode = False
                else:
                    self.insert_mode = False

    def skip_annotate(self):
        status = self.status
        self.log.close_line()
        if status['counter']<len(self.dataloader.LABEL)-1:
            status['counter'] += 1
        self.update_filters_callback()
        self.result_data_callback(status)
        self.log.add_new_line()

    def backskip_annotate(self):
        status = self.status
        self.log.close_line()
        status['counter'] -= 1
        self.update_filters_callback()
        self.result_data_callback(status)
        self.log.add_new_line()

    def confirm_annotate(self):
        self.timer_color_green()
        status = self.status
        self.scoring.table.setItem(self.status['counter'], 2,
                    QTableWidgetItem(self.scoring.table.item(
                        status['counter'], 1).text()))
        self.log.scoring = self.scoring.table.item(
            status['counter'], 1).text()
        self.user_annotation_record[status['counter']] = (
            self.dataloader.PRED[status['counter']])
        self.log.event_append('{}'.format(0))
        self.skip_annotate()

    def reannotate(self):
        self.reannotate_popup()

    def timer_color_green(self):
        self.pred_label.setStyleSheet('color: green')
        timer = QTimer()
        timer.setSingleShot(True)
        QTimer.singleShot(500, self.set_text_color_black)

    def timer_user_record(self):
        sleep_level = ['W', 'N1', 'N2', 'N3', 'REM']
        self.pred_label.setStyleSheet('color: green')
        timer = QTimer()
        timer.setSingleShot(True)
        QTimer.singleShot(500, self.skip_annotate)

    def timer_user_buffer_record(self):
        sleep_level = ['W', 'N1', 'N2', 'N3', 'REM']
        self.pred_label.setText('                                         '
                        + 'User: {}'.format(
                        sleep_level[self.annotate_local_buffer]))

    def set_text_color_black(self):
        self.pred_label.setStyleSheet('color: black')

    def moving_mean_callback(self):
        status = self.status

    def zoom_in_callback(self):
        status = self.status
        if self.status['calib'] > 2:
            status['calib'] = int(status['calib']/2)
        self.result_data_callback(status)

    def zoom_out_callback(self):
        status = self.status
        status['calib'] = int(status['calib']*2)
        self.result_data_callback(status)

    def stretch_x_callback(self):
        status = self.status
        status['X'] = not status['X']
        status['bool_stat']['gradient'] = status['X']
        status['bool_stat']['fill'] = not status['X']
        self.result_data_callback(status)

    def stretch_y_callback(self):
        status = self.status
        status['Y'] = not status['Y']
        self.result_data_callback(status)

    def fill_callback(self):
        status = self.status
        status['bool_stat']['fill'] = not status['bool_stat']['fill']
        self.result_data_callback(status)

    def update_activation(self):
        calib = self.status['calib']
        start_point = (self.status['counter'] - calib)*LEN_EPOCH
        focus_point = self.status['counter'] + 0.5
        end_point = (self.status['counter'] + calib)*LEN_EPOCH
        ticks = [[(v, str(v/1000)+'k') for v in LEN_EPOCH*np.arange(
            self.status['counter'] - calib, self.status['counter'] + calib)]]
        lin_space = np.arange(0, LEN_EPOCH)
        sleep_level = ['W', 'N1', 'N2', 'N3', 'REM']

        if int(self.dataloader.PRED[self.status['counter']])!=3:
            if self.activation_item_list:
                for item in self.activation_item_list:
                    self.activation_ax.removeItem(item)
                    self.activation_ax.clear()
                self.activation_item_list = []
            plot_act_item = self.activation['alphas'][self.status['counter']]
            color = 'b'
            item = pg.PlotCurveItem(
                lin_space,
                plot_act_item,
                pen=pg.mkPen(
                    color=color, width=0.5)
            )
            self.activation_ax.setYRange(-10, 10)
            self.activation_ax.addItem(item)
            self.activation_item_list.append(item)
        else:
            if self.activation_item_list:
                for item in self.activation_item_list:
                    self.activation_ax.removeItem(item)
                    self.activation_ax.clear()
                self.activation_item_list = []
            plot_act_item = self.activation['deltas'][self.status['counter']].astype(float)
            color = QColor('#009000')
            item = pg.PlotCurveItem(
                lin_space,
                plot_act_item,
                pen=pg.mkPen(
                    color=color, width=2)
            )
            self.activation_ax.setYRange(-0.5, 1.5)
            self.activation_ax.addItem(item)
            self.activation_item_list.append(item)

        if int(self.dataloader.PRED[self.status['counter']])==1:
            rect_tuple_list = self.make_rectangle_from_boolean(
                self.activation['complexes'][self.status['counter']])
        elif int(self.dataloader.PRED[self.status['counter']])==2:
            rect_tuple_list = self.make_rectangle_from_boolean(
                self.activation['spindles'][self.status['counter']])
            for rect_tuple in rect_tuple_list:
                start_rect = int((focus_point - 0.5)*LEN_EPOCH) + rect_tuple[0] - 50
                end_rect = int((focus_point - 0.5)*LEN_EPOCH) + rect_tuple[1] + 50
                data_at_rect = self.dataloader.DATA_LIST[
                    self.get_main_channel_idx()][start_rect:end_rect]
                rect_item = QGraphicsRectItem(
                    start_rect,
                    np.min(data_at_rect)-5,
                    end_rect - start_rect,
                    (np.max(data_at_rect)
                        -np.min(data_at_rect) + 10)
                    )
                rect_item.setPen(QPen(QColor('#C01E90FF'), 4))
                self.ax_list[self.get_main_channel_idx()].addItem(rect_item)
            rect_tuple_list = self.make_rectangle_from_boolean(
                self.activation['complexes'][self.status['counter']])
            for rect_tuple in rect_tuple_list:
                start_rect = int((focus_point - 0.5)*LEN_EPOCH) + rect_tuple[0] - 50
                end_rect = int((focus_point - 0.5)*LEN_EPOCH) + rect_tuple[1] + 50
                data_at_rect = self.dataloader.DATA_LIST[
                    self.get_main_channel_idx()][start_rect:end_rect]
                rect_item = QGraphicsRectItem(
                    start_rect,
                    np.min(data_at_rect)-5,
                    end_rect - start_rect,
                    (np.max(data_at_rect)
                        -np.min(data_at_rect) + 10)
                    )
                rect_item.setPen(QPen(QColor('#C0DC143C'), 4))
                self.ax_list[self.get_main_channel_idx()].addItem(rect_item)
        elif int(self.dataloader.PRED[self.status['counter']])==4:
            rect_tuple_list = self.make_rectangle_from_boolean(
                self.activation['sawtooth'][self.status['counter']])
            for rect_tuple in rect_tuple_list:
                start_rect = int((focus_point - 0.5)*LEN_EPOCH) + rect_tuple[0] - 50
                end_rect = int((focus_point - 0.5)*LEN_EPOCH) + rect_tuple[1] + 50
                data_at_rect = self.dataloader.DATA_LIST[
                    self.get_main_channel_idx()][start_rect:end_rect]
                rect_item = QGraphicsRectItem(
                    start_rect,
                    np.min(data_at_rect)-5,
                    end_rect - start_rect,
                    (np.max(data_at_rect)
                        -np.min(data_at_rect) + 10)
                    )
                rect_item.setPen(QPen(QColor('#C0800000'), 4))
                self.ax_list[self.get_main_channel_idx()].addItem(rect_item)

    def update_filters_callback(self):
        for ax in self.filter.ax_list:
            ax.pocket = False
            ax.item.setData(pen=pg.mkPen(color='b', width=0.3))

    def result_data_callback(self, status):
        self.status = status
        self.scoring.table.setCurrentCell(status['counter'], 0)
        self.make_signal()
        self.make_fill()
        if self.session['exp_target_model'] in ['DeepSleepNet', 'DeepStrideNet']:
            print(self.session['exp_target_model'])
            self.make_gradient()
        if self.session['exp_target_model'] in ['DeepSleepNet', 'DeepStrideNet']:
            self.plot_hypnogram()
        self.plot_signal()
        self.annotate.edit.setText('{}'.format(status['counter'] + 1))
        if self.session['exp_target_model'] in ['DeepSleepNet', 'DeepStrideNet']:
            self.set_model_pred_label()
        if self.session['exp_target_model'] in ['DeepStrideNet']:
            if not (self.session_num in [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]):
                self.update_activation()

    def set_model_pred_label(self):
        sleep_level = ['W', 'N1', 'N2', 'N3', 'REM']
        self.pred_label.setText('                                 '
            + 'Model Prediction: {}'.format(
            sleep_level[int(self.dataloader.PRED[self.status['counter']])]))
        # self.pred_label.setText(
        #     sleep_level[int(self.dataloader.PRED[self.status['counter']])])

    def plot_hypnogram(self):
        use_data = self.dataloader.PRED
        sleep_level = ['W', 'N1', 'N2', 'N3', 'REM']
        calib = self.status['calib']
        brush = pg.mkBrush(pg.hsvColor(0.6, 1, 0.3, 0.3))
        lin_space = np.array(
            (np.arange(self.dataloader.PRED.shape[0]),
            np.arange(self.dataloader.PRED.shape[0])+1))
        lin_space = lin_space.transpose().reshape(-1)
        # label_temp = np.array((self.dataloader.LABEL, self.dataloader.LABEL))
        # label_temp = label_temp.transpose().reshape(-1)
        pred_temp = np.array((use_data, use_data))
        pred_temp = pred_temp.transpose().reshape(-1)
        hypno_temp = np.array((self.user_annotation_record,
            self.user_annotation_record))
        hypno_temp = hypno_temp.transpose().reshape(-1)
        ticks = [[(v, '{}'.format(
            self.time_format([23, 0, 0 + int(v*30)]))) for v in (np.arange(
            self.dataloader.PRED.shape[0])[0::50])]]
        y_ticks = [[(v, '{}'.format(
            sleep_level[v])) for v in np.arange(0, 5)]]
        y_ticks[0].append((-1, 'Not scored'))
        self.hypnogram_ax.clear()
        self.hypnogram_ax.setBackground('w')

        self.hypnogram_ax.setYRange(-1, 5)
        self.hypnogram_ax.setXRange(0, self.dataloader.PRED.shape[0])
        self.hypnogram_ax.plot(lin_space, pred_temp,
            pen=pg.mkPen(color='b', width=POINTFIVE))
        self.hypnogram_ax.plot(lin_space, hypno_temp,
            pen=pg.mkPen(color='r', width=POINTFIVE))
        # self.hypnogram_ax.plot(lin_space, label_temp,
        #     pen=pg.mkPen(color='g', width=0.5))
        stem1 = pg.PlotDataItem(x=np.ones(5)*(self.status['counter']), 
            y=np.linspace(-1, 6, 5))
        stem2 = pg.PlotDataItem(x=np.ones(5)*(self.status['counter'] + 1), 
            y=np.linspace(-1, 6, 5))
        fbtwn = pg.FillBetweenItem(stem1, stem2, brush=brush)
        self.hypnogram_ax.addItem(fbtwn)

        x_axis = self.hypnogram_ax.getAxis('bottom')
        y_axis = self.hypnogram_ax.getAxis('left')
        x_axis.setTicks(ticks)
        y_axis.setTicks(y_ticks)
        self.annotate.info_pred_label.setText('Annotate as {}'.format(
            sleep_level[int(self.dataloader.PRED[self.status['counter']])]))
        self.annotate.reannotate_label.setText('Reannotate')
        self.hypnogram_ax.setTitle(
                'Model Prediction: {}, Epoch: {}/{}'
                .format(sleep_level[int(
                            self.dataloader.PRED[self.status['counter']])],
                        self.status['counter'] + 1,
                        len(self.dataloader.PRED)
                        )
                )

    def update_widget(self):
        for key_item in self.status['bool_stat']:
            if self.status['bool_stat'][key_item]:
                for row in range(self.NROW):
                    self.ax_list[row].addItem(self.item_list[row][key_item])
            else:
                for row in range(self.NROW):
                    self.ax_list[row].removeItem(
                        self.item_list[row][key_item])

    def make_signal(self):
        calib = self.status['calib']
        start_point = (self.status['counter'] - calib)*LEN_EPOCH
        focus_point = self.status['counter'] + 0.5
        end_point = (self.status['counter'] + calib)*LEN_EPOCH
        ticks = [[(v, str(v/1000)+'k') for v in LEN_EPOCH*np.arange(
            self.status['counter'] - calib, self.status['counter'] + calib)]]

        lin_space = np.arange(max(0, start_point), min(end_point,
            len(self.dataloader.DATA_LIST[0])))

        for row in range(self.NROW):
            ax = pg.PlotCurveItem(
                lin_space,
                self.dataloader.DATA_LIST[row][lin_space],
                pen=pg.mkPen(
                    color='k', width=(POINTSEVEN if row==self.get_main_channel_idx() 
                                    else POINTONE))
            )
            self.item_list[row]['signal'] = ax

    def make_fill(self):
        calib = self.status['calib']
        start_point = (self.status['counter'] - calib)*LEN_EPOCH
        focus_point = self.status['counter'] + 0.5
        end_point = (self.status['counter'] + calib)*LEN_EPOCH
        brush = pg.mkBrush(pg.hsvColor(0.6, 1, 0.3, 0.3))

        for row in range(self.NROW):
            stem1 = pg.PlotDataItem(
                x=np.ones(50)*(focus_point - 0.5)*LEN_EPOCH, 
                y=np.linspace(np.min(self.dataloader.DATA_LIST[row]), 
                              np.max(self.dataloader.DATA_LIST[row]), 50))
            stem2 = pg.PlotDataItem(
                x=np.ones(50)*(focus_point + 0.5)*LEN_EPOCH, 
                y=np.linspace(np.min(self.dataloader.DATA_LIST[row]),
                              np.max(self.dataloader.DATA_LIST[row]), 50))
            fbtwn = pg.FillBetweenItem(stem1, stem2, brush=brush)
            self.item_list[row]['fill'] = fbtwn

    def make_gradient(self):
        calib = self.status['calib']
        start_point = (self.status['counter'] - calib)*LEN_EPOCH
        focus_point = self.status['counter'] + 0.5
        end_point = (self.status['counter'] + calib)*LEN_EPOCH
        ticks = [[(v, str(v/1000)+'k') for v in LEN_EPOCH*np.arange(
            self.status['counter'] - calib, self.status['counter'] + calib)]]
        brush = pg.mkBrush(QColor(200, 30, 30, 50))
        block_size = 50
        th_ratio = 0.999

        grad = self.dataloader.GRAD[int(focus_point - 0.5)].squeeze()
        # grad = np.abs(grad)
        mask = np.zeros(grad.shape)

        asdf = np.argsort(grad)
        th = grad[asdf[int(LEN_EPOCH*th_ratio)]]
        ps = scipy.signal.find_peaks(grad, 
                                     prominence=th,
                                     distance=block_size*4)[0]
        ps[ps<block_size] = block_size
        ps[ps>len(grad)-block_size-1] = len(grad)-block_size-1
        for i in range(-block_size, block_size+1):
            mask[ps + i] = 1
        disp_space = np.arange( int(focus_point - 0.5)*LEN_EPOCH,
                                int(focus_point + 0.5)*LEN_EPOCH)
        mask[mask==1] = np.max(self.dataloader.DATA_LIST[0])
        mask[mask==0] = np.min(self.dataloader.DATA_LIST[0]) - 5
        fbtwn = pg.PlotDataItem(disp_space, 
                                mask, 
                                fillLevel=np.min(
                                    self.dataloader.DATA_LIST[0])-5,
                                brush=brush)
        for row in range(self.NROW):
            self.item_list[row]['gradient'] = pg.PlotDataItem()

        self.item_list[2]['gradient'] = fbtwn

    def time_format(self, time_list):
        tmp_list = time_list
        tmp_list[1] += int(time_list[2]/60)
        tmp_list[2] = tmp_list[2]%60
        tmp_list[0] += int(tmp_list[1]/60)
        tmp_list[1] = tmp_list[1]%60
        tmp_list[0] = tmp_list[0]%24

        return datetime.time(tmp_list[0], tmp_list[1], tmp_list[2])

    def make_rectangle_from_boolean(self, activation):
        rectangle_start_list = []
        rectangle_end_list = []
        rectangle_tuple_list = []
        for i in range(len(activation)-1):
            if int(activation[i])==0 and int(activation[i+1])!=0:
                rectangle_start_list.append(i)

        for i in range(1, len(activation)):
            if int(activation[i-1])==0 and int(activation[i])!=0:
                rectangle_end_list.append(i)

        iterator = 0
        temp_len = len(rectangle_start_list)
        while(iterator<temp_len-1):
            while(rectangle_start_list[iterator+1]-rectangle_end_list[iterator]<100):
                rectangle_start_list.pop(iterator+1)
                rectangle_end_list.pop(iterator)
                temp_len = temp_len-1
                if iterator==temp_len-1:
                    break
            iterator = iterator + 1

        assert len(rectangle_start_list)==len(rectangle_end_list)
        for i in range(len(rectangle_start_list)):
            rectangle_tuple_list.append((rectangle_start_list[i], rectangle_end_list[i]))

        return rectangle_tuple_list

    def plot_signal(self):
        calib = self.status['calib']
        start_point = (self.status['counter'] - calib)*LEN_EPOCH
        focus_point = self.status['counter'] + 0.5
        end_point = (self.status['counter'] + calib)*LEN_EPOCH

        sleep_level = ['W', 'N1', 'N2', 'N3', 'REM']
        start_time = [23, 0, 0]
        ticks = [[(v, '{}'.format(self.time_format(
            [23, 0, 0 + int(v/100)]))) for v in LEN_EPOCH*np.arange(
            self.status['counter'] - calib, self.status['counter'] + calib)]]

        for row in range(self.NROW):
            self.ax_list[row].clear()
            self.ax_list[row].setBackground('w')
            if self.status['Y']:
                self.ax_list[row].setYRange(
                    -self.channel_height,
                    self.channel_height
                    )
                y_axis = self.ax_list[row].getAxis('left')
                y_axis.setTicks([[(self.channel_height, 
                    '{}'.format(self.channel_height)),
                    (-self.channel_height, 
                    '{}'.format(-self.channel_height))]])
                if self.dataloader.DATA_CHANNEL[row] in ['LOC_A2', 'ROC_A1']:
                    self.ax_list[row].setYRange(
                        -self.channel_height*self.channel_ratio,
                        self.channel_height*self.channel_ratio
                        )
                    y_axis = self.ax_list[row].getAxis('left')
                    y_axis.setTicks([[(self.channel_height*self.channel_ratio, 
                        '{}'.format(self.channel_height*self.channel_ratio)),
                        (-self.channel_height*self.channel_ratio, 
                        '{}'.format(-self.channel_height*self.channel_ratio))
                        ]])
            else:
                self.ax_list[row].setYRange(
                    np.min(self.dataloader.DATA_LIST[row]),
                    np.max(self.dataloader.DATA_LIST[row])
                    )

            if self.status['X']:
                self.ax_list[row].setXRange(
                    int((focus_point - 0.5)*LEN_EPOCH),
                    int((focus_point + 0.5)*LEN_EPOCH)
                    )
            else:
                self.ax_list[row].setXRange(
                    start_point + int(LEN_EPOCH/2),
                    end_point + int(LEN_EPOCH/2)
                    )
            x_axis = self.ax_list[row].getAxis('bottom')
            ticks = [[(v, '{}'.format(self.time_format(
                [23, 0, 0 + int(v/100)])) if v%3000==0 else '') for v in LEN_EPOCH*np.arange(
                (self.status['counter'] - calib)*30,
                (self.status['counter'] + calib)*30)/30]]
            x_axis.setTicks(ticks)
            # x_axis.setPen(color='k', width=1)

            if row==self.get_main_channel_idx():
                x_axis = self.ax_list[row].getAxis('bottom')
                ticks = [[(v, '{}'.format(self.time_format(
                    [23, 0, 0 + int(v/100)])) if v%3000==0 else '') for v in LEN_EPOCH*np.arange(
                    (self.status['counter'] - calib)*5,
                    (self.status['counter'] + calib)*5)/5]]
                x_axis.setTicks(ticks)

            self.ax_list[row].setLabel('left', '{}'.format(self.dataloader.DATA_CHANNEL[row]))

        self.update_widget()

    def btn_filter_handler(self):
        self.openwindow()

    def openwindow(self):
        self.filter_window = FilterWindow()
        self.filter_window.show()

    def reannotate_popup(self):
        self.reannotate_window = ReannotateWindow(self)
        self.reannotate_window.show()
        sleep_level = ['W', 'N1', 'N2', 'N3', 'REM']
        self.reannotate_window.pushbutton_list[0].clicked.connect(
            self._scoring_W)
        self.reannotate_window.pushbutton_list[1].clicked.connect(
            self._scoring_N1)
        self.reannotate_window.pushbutton_list[2].clicked.connect(
            self._scoring_N2)
        self.reannotate_window.pushbutton_list[3].clicked.connect(
            self._scoring_N3)
        self.reannotate_window.pushbutton_list[4].clicked.connect(
            self._scoring_REM)
        for i in range(5):
            self.reannotate_window.pushbutton_list[i].clicked.connect(
                self.skip_annotate)

    def _scoring_W(self):
        self.scoring.table.setItem(self.status['counter'], 2,
            QTableWidgetItem('W'))
        self.user_annotation_record[status['counter']] = 0
        self.log.scoring = 'D_W'
        self.log.event_append('{}'.format(0))
    def _scoring_N1(self):
        self.scoring.table.setItem(self.status['counter'], 2,
            QTableWidgetItem('N1'))
        self.user_annotation_record[status['counter']] = 1
        self.log.scoring = 'D_N1'
        self.log.event_append('{}'.format(0))
    def _scoring_N2(self):
        self.scoring.table.setItem(self.status['counter'], 2,
            QTableWidgetItem('N2'))
        self.user_annotation_record[status['counter']] = 2
        self.log.scoring = 'D_N2'
        self.log.event_append('{}'.format(0))
    def _scoring_N3(self):
        self.scoring.table.setItem(self.status['counter'], 2,
            QTableWidgetItem('N3'))
        self.user_annotation_record[status['counter']] = 3
        self.log.scoring = 'D_N3'
        self.log.event_append('{}'.format(0))
    def _scoring_REM(self):
        self.scoring.table.setItem(self.status['counter'], 2,
            QTableWidgetItem('REM'))
        self.user_annotation_record[status['counter']] = 4
        self.log.scoring = 'D_REM'
        self.log.event_append('{}'.format(0))


class UI_ReannotateWindow():
    sleep_level = ['W', 'N1', 'N2', 'N3', 'REM']

    def setupUI(self, ReannotateWindow):
        ReannotateWindow.setObjectName("ReannotateWindow")
        ReannotateWindow.resize(200, 200)
        ReannotateWindow.move(3500, 1000)

        self.centralwidget = QWidget(ReannotateWindow)

        self.option_layout = QVBoxLayout()
        self.option_layout.setObjectName("option_layout")
        self.pushbutton_list = []

        for i in range(5):
            self.pushbutton_list.append(
                QPushButton('{}'.format(self.sleep_level[i])))
            self.option_layout.addWidget(self.pushbutton_list[i])
            self.pushbutton_list[i].clicked.connect(self.close)

        self.centralwidget.setLayout(self.option_layout)
        self.setCentralWidget(self.centralwidget)


class ReannotateWindow(QMainWindow, UI_ReannotateWindow):
    def __init__(self, parent):
        super(ReannotateWindow, self).__init__()
        self.parent = parent
        self.setupUI(self)

class Image(QLabel):
    def __init__(self):
        super().__init__()

    def mouseMoveEvent(self, event):

        mimeData = QMimeData()

        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setHotSpot(event.pos())

        dropAction = drag.exec_(Qt.MoveAction)

def peak_test(corr):
    LEN_EPOCH = 3000
    block_size = 30
    th_ratio = 0.85

    mask = np.zeros(corr.shape)

    asdf = np.argsort(corr)
    th = corr[asdf[int(LEN_EPOCH*th_ratio)]]
    ps = signal.find_peaks(corr, 
                                 prominence=th, 
                                 distance=block_size*4)[0]
    mask[ps] = 1

    return np.multiply(mask, corr)


if __name__ == '__main__':
    app = QApplication([])
    user_num = 0
    path = 'exp_setting/config_user{}.json'.format(user_num)
    session_order = [0, 1]

    with open(path, 'r') as json_file:
        config = json.load(json_file)
    print(len(session_order))
    for session_num in session_order:
        session = config['session{}'.format(session_num)]
        paths_files = session['exp_path']
        dataloader = Data_Loader(
            paths_files,
            subject=int(session['exp_data'][0]),
            start_epoch=int(session['exp_data'][1]),
            exp_start_epoch=int(session['exp_data'][2]),
            data_channel=session['exp_channel_list'],
            target_model=session['exp_target_model'],
            num_epochs=session['exp_num_epochs']
        )
        window = MainWindow(session, dataloader, user_num, session_num)
        window.show()
        app.exec_()

        print(window.key_logs)
        del window
        key = input()
        if key=='q':
            sys.exit()
