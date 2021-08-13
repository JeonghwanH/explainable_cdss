from PyQt5.QtCore import (Qt, QSize, pyqtSignal)
from PyQt5.QtWidgets import (QAbstractItemView,
                             QFileDialog,
                             QHBoxLayout,
                             QGridLayout,
                             QPushButton,
                             QVBoxLayout,
                             QTableWidget,
                             QTableWidgetItem,
                             QWidget,
                             QLabel,
                             QAbstractScrollArea,
                             QSizePolicy,
                             QLineEdit
                             )
from PyQt5.QtGui import QIcon, QIntValidator, QFont
import pyqtgraph as pg
from scipy import signal
import numpy as np
import datetime

class Scoring(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.create()

    def create(self):
        self.table = QTableWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.table)

        self.table.setColumnCount(3)
        self.table.setRowCount(self.parent.dataloader.PRED.shape[0])
        self.table.setHorizontalHeaderLabels(
            ['Epoch', 'Model Pred', 'Annotation'])
        self.table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.table.resizeColumnsToContents()
        self.setLayout(layout)

        sleep_level = ['W', 'N1', 'N2', 'N3', 'REM']
        ticks = ['{}'.format(
            self.time_format([23, 0, 0 + int(v*30)])) for v in np.arange(
            self.parent.dataloader.PRED.shape[0])]

        for idx, tick in enumerate(ticks):
            self.table.setItem(idx, 0, QTableWidgetItem(tick))
            if self.parent.session['exp_target_model'] in ['DeepSleepNet', 'DeepStrideNet']:
                self.table.setItem(idx, 1, QTableWidgetItem(
                    sleep_level[int(self.parent.dataloader.PRED[idx])]))
        self.table.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustToContents)
        self.table.resizeColumnsToContents()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def time_format(self, time_list):
        tmp_list = time_list
        tmp_list[1] += int(time_list[2]/60)
        tmp_list[2] = tmp_list[2]%60
        tmp_list[0] += int(tmp_list[1]/60)
        tmp_list[1] = tmp_list[1]%60
        tmp_list[0] = tmp_list[0]%24

        return datetime.time(tmp_list[0], tmp_list[1], tmp_list[2])


class MyPlotWidget(pg.GraphicsLayoutWidget):
    pltClicked = pyqtSignal()

    def __init__(self, parent, idx):
        self.parent = parent
        super().__init__()
        self.pocket = False
        self.set_index(idx)

    def mousePressEvent(self, event):
        print('clicked plot: {}, event: {}'.format(id(self), event))
        # self.clicked_action()
        # self.pocket_full()

    def _cos_corr(self, filter_, data):
        dat = np.correlate(data, filter_, 'same')
        mag = (np.sqrt(
            np.correlate(data*data, np.ones(np.shape(filter_)), 'same'))
            *np.sqrt(np.inner(filter_, filter_))
               )

        return dat/mag

    def peak_test(self, corr):
        LEN_EPOCH = 3000
        if self.parent.peaks:
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
        else:
            return corr

    def clicked_action(self):
        ax_list = self.parent.parent.ax_list
        log = self.parent.parent.log
        if not self.pocket:
            filter_ = self.parent.filters[self.idx].squeeze()
            lin_space = (np.arange(3000)
                + self.parent.parent.status['counter']*3000)
            self.child = pg.PlotDataItem(
                lin_space, 
                self.parent.parent.channel_height*2*self.peak_test(self._cos_corr(
                    filter_,
                    self.parent.parent.dataloader.DATA_LIST[0][lin_space])),
                pen=pg.mkPen(color='r', width=2))
            ax_list[self.parent.parent.get_main_channel_idx()].addItem(
                self.child)
            self.item.setData(pen=pg.mkPen(color='r', width=0.3))
            log.event_append('1' + '{}'.format(self.idx))
        else:
            ax_list[self.parent.parent.get_main_channel_idx()].removeItem(
                self.child)
            self.item.setData(pen=pg.mkPen(color='b', width=0.3))
            log.event_append('2' + '{}'.format(self.idx))

    def set_index(self, idx):
        self.idx = idx

    def set_item(self, item):
        self.item = item

    def pocket_full(self):
        self.pocket = not self.pocket


class Filters(QWidget):
    def __init__(self, parent, path, relevant_stage_dict):
        super().__init__()
        self.filters = np.load(path)
        self.parent = parent
        self.grid_layout = QGridLayout()
        self.ax_list = []
        self.ncols = 8
        self.nrows = 8
        self.peaks = True
        self.relevant_stage_dict = relevant_stage_dict

        for i in range(self.ncols):
            for j in range(self.nrows):
                y_axis = pg.AxisItem(orientation='left')
                y_axis.setTicks([[(0, '0'), (1, '1'), (2, '2')]])
                ax = MyPlotWidget(self, idx=i*self.nrows+j)
                ax.setBackground('w')
                view = ax.addViewBox(row=0, col=0)
                self.grid_layout.addWidget(ax, i, j, 1, 1)
                self.ax_list.append(ax)

                # plot_on_ax = ax.addPlot(row=0, col=0)
                item = pg.PlotDataItem(
                    x=np.arange(self.filters.shape[2]),
                    y=self.filters[i*self.nrows+j].squeeze(),
                    pen=pg.mkPen(color='b', width=0.3))
                # plot_on_ax.addItem(item)
                # plot_on_ax.setYRange(-50, 50)
                view.setRange(yRange=(-40, 40))
                # item.addItem(y_axis)
                view.addItem(item)
                ax.set_item(item)

        self.setLayout(self.grid_layout)


class Label_Docker_Widget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.layout = QGridLayout()
        self.setLayout(self.layout)


class Model_Pred_Label(QLabel):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setFont(QFont('Arial', 50))
        self.setStyleSheet('color: black')


class Hypnogram(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setSizePolicy(
            QSizePolicy.MinimumExpanding,
            QSizePolicy.MinimumExpanding
        )

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def sizeHint(self):
        return QSize(1500, 150)


class Spectrogram_Docker_Widget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setSizePolicy(
            QSizePolicy.MinimumExpanding,
            QSizePolicy.MinimumExpanding
        )

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def sizeHint(self):
        return QSize(300, 10)


class Spectrogram(pg.PlotWidget):
    fs = 100 # Hz
    LEN_EPOCH = 3000 # samples
    time = np.arange(LEN_EPOCH)/float(fs)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.img_array = np.zeros((1000, int(self.fs/2)+1))

        # bipolar colormap
        self.setBackground('w')

        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], 
            (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        self.img.setLookupTable(lut)
        self.img.setLevels([-50,40])

        freq = np.arange((self.LEN_EPOCH/2)+1)/(float(self.LEN_EPOCH)/self.fs)
        yscale = 1.0/(self.img_array.shape[1]/freq[-1])
        self.img.scale((1./self.fs)*self.LEN_EPOCH, yscale)

        self.setLabel('left', '(Hz)')

    def update(self, col):
        x = self.parent.dataloader.DATA_LIST[col][(
            3000*self.parent.status['counter']):(
            3000*self.parent.status['counter']+3000)]
        nperseg = 200
        f, t, Zxx = signal.stft(x, self.fs, nperseg=nperseg)

        x_ticks = [[(v*len(t), '{}'.format(v)) for v in t[::int(len(t)/2)]]]
        y_ticks = [[(v*nperseg/self.fs,
            '{}'.format(int(v))) for v in f[::int(len(f)/5)]]]
        x_axis = self.getAxis('bottom')
        y_axis = self.getAxis('left')

        x_axis.setTicks(x_ticks)
        y_axis.setTicks(y_ticks)
        
        psd = abs(Zxx)
        psd = np.log10(psd)*20
        self.img_array = psd.transpose()
        self.img.setImage(self.img_array, autoLevels=False)


class Annotate(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.create()

    def create(self):
        self.prev_button = QPushButton()
        self.prev_button.setIcon(QIcon('GUI/buttons/button_prev.png'))
        self.prev_button.clicked.connect(self.parent.backskip_annotate)

        self.epoch_layout = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setDragEnabled(True)
        self.edit.setAlignment(Qt.AlignRight)
        self.edit.setValidator(
            QIntValidator(1,self.parent.dataloader.GRAD.shape[0])
            )
        self.edit.returnPressed.connect(self.callback_set_status)

        self.epoch_layout.addWidget(self.edit)
        self.epoch_layout.addWidget(QLabel('/{}'.format(
            self.parent.dataloader.PRED.shape[0])))

        self.next_button = QPushButton()
        self.next_button.setIcon(QIcon('GUI/buttons/button_next.png'))
        self.next_button.clicked.connect(self.parent.skip_annotate)

        move_layout = QHBoxLayout()
        move_layout.addWidget(self.prev_button)
        move_layout.addLayout(self.epoch_layout)
        move_layout.addWidget(self.next_button)

        self.check_button = QPushButton()
        self.check_button.setIcon(QIcon('GUI/buttons/green_check.jpg'))
        self.check_button.clicked.connect(self.parent.confirm_annotate)
        self.info_pred_label = QLabel(self)
        confirm_layout = QHBoxLayout()
        confirm_layout.addWidget(self.check_button)
        confirm_layout.addWidget(self.info_pred_label)

        self.cross_button = QPushButton()
        self.cross_button.setIcon(QIcon('GUI/buttons/red_cross.jpg'))
        self.cross_button.clicked.connect(self.parent.reannotate)
        self.reannotate_label = QLabel(self)
        reannotate_layout = QHBoxLayout()
        reannotate_layout.addWidget(self.cross_button)
        reannotate_layout.addWidget(self.reannotate_label)

        annotate_layout = QHBoxLayout()
        annotate_layout.addLayout(confirm_layout)
        annotate_layout.addLayout(reannotate_layout)

        layout = QVBoxLayout()
        layout.addLayout(move_layout)
        layout.addLayout(annotate_layout)

        self.setLayout(layout)

    def callback_set_status(self):
        self.parent.log.event_append('{}'.format(30))
        self.parent.log.close_line()
        status = self.parent.status
        status['counter'] = int(self.edit.text()) - 1
        self.parent.result_data_callback(status)
        self.parent.log.add_new_line()


class Acc_Dist_Disp(QWidget):
    def __init__(self, parent):
        super().__init__()
        layout = QGridLayout()

        self.parent = parent
        self.min_edit = QLineEdit()
        self.min_edit.setDragEnabled(True)
        self.min_edit.setAlignment(Qt.AlignRight)
        self.min_edit.setValidator(
            QIntValidator(1,self.parent.dataloader.PRED.shape[0])
            )
        # self.min_edit.returnPressed.connect()

        self.max_edit = QLineEdit()
        self.max_edit.setDragEnabled(True)
        self.max_edit.setAlignment(Qt.AlignRight)
        self.max_edit.setValidator(
            QIntValidator(1, self.parent.dataloader.PRED.shape[0])
            )
        self.max_edit.returnPressed.connect(self._callback_set_item)

        self.lab_pred = QLabel()
        self.dist = QLabel()

        layout.addWidget(self.min_edit, 0, 0, 1, 1)
        layout.addWidget(self.max_edit, 0, 1, 1, 1)
        layout.addWidget(self.lab_pred, 0, 2, 1, 1)
        layout.addWidget(self.dist, 1, 0, 1, 3)

        self.setLayout(layout)

        self.min_edit.setText('0')
        self.max_edit.setText('60')
        self._callback_set_item()

    def _callback_set_item(self):
        min_value = int(self.min_edit.text())
        max_value = int(self.max_edit.text())

        correct_epoch = np.sum(
            (self.parent.dataloader.PRED[min_value:max_value]
                ==self.parent.dataloader.LABEL[min_value:max_value]))
        num_epoch = max_value-min_value
        acc = np.round(correct_epoch/num_epoch, 2)

        self.lab_pred.setText('{}/{} = {}'.format(
            correct_epoch, num_epoch, acc))

        sleep_level = ['W', 'N1', 'N2', 'N3', 'REM']
        dist = {}
        for i in range(5):
            dist[sleep_level[i]] = np.sum(
                self.parent.dataloader.LABEL[min_value:max_value]==i)

        self.dist.setText('{}'.format(dist))
