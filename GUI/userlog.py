import time
import os
import csv

class User_Log():
    def __init__(self, parent):
        self.parent = parent

        if not os.path.exists('user_logs'):
            os.mkdir('user_logs')

        path = os.path.join('user_logs',
            '{}'.format(parent.user_num))
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join('user_logs',
            '{}'.format(parent.user_num),
            '{}'.format(parent.session_num))
        if not os.path.exists(path):
            os.mkdir(path)

        self.f_events = open(os.path.join(path, 'events.csv'), 'w+')
        self.f_time = open(os.path.join(path, 'elapsed_time.csv'), 'w+')
        self.fieldnames = ['counter',
                           'epoch',
                           'elapsed_time',
                           'scoring',
                           'used_info']
        self.counter = self.parent.status['counter']
        self.epoch = self.parent.status['counter']
        self.event_buffer = []
        self.time_buffer = []
        self.elapsed_time = time.time()-self.parent.start_timepoint
        self.scoring = '-'

        self.event_writer = csv.DictWriter(self.f_events,
            fieldnames=self.fieldnames)
        self.time_writer = csv.DictWriter(self.f_time,
            fieldnames=['list_elapsed_time'])
        self.event_writer.writeheader()

    def event_append(self, event):
        self.event_buffer.append(event)
        self.time_buffer.append(
            round(time.time()
                -self.parent.start_timepoint-self.elapsed_time, 3))

    def add_new_line(self):
        self.counter += 1
        self.epoch = self.parent.status['counter']
        self.event_buffer = []
        self.time_buffer = []
        self.elapsed_time = round(time.time()-self.parent.start_timepoint, 3)
        self.scoring = '-'

    def close_line(self):
        self.event_writer.writerow({'counter': self.counter,
                                    'epoch': self.epoch,
                                    'elapsed_time': self.elapsed_time,
                                    'scoring': self.scoring,
                                    'used_info': str(self.event_buffer)
                                   })
        self.time_writer.writerow({'list_elapsed_time': list(self.time_buffer)})
        print('line closed')