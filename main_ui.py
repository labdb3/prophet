'''
本软件会将用户的习惯和个性化设置存放到conf.ini文件中。
'''
import configparser
import logging
import os
import pickle as pk
import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)
from matplotlib import font_manager as fm
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pylab as plt
import pandas as pd

from dialog import paramDialog
from model import RegisteredModel

logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='debug.log',
                    filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    # a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    # 日志格式
                    )


matplotlib.use('QT5Agg')


class Prophet(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        menubar = self.menuBar()
        fitMenu = menubar.addAction('拟合/打开文件')
        saveMenu = menubar.addAction('保存模型')
        loadMenu = menubar.addAction('加载模型')
        predMenu = menubar.addAction('预测/打开文件')
        savePredMenu = menubar.addAction('保存预测')
        helpMenu = menubar.addMenu('&帮助')
        aboutAction = helpMenu.addAction('About')
        tutorAction = helpMenu.addAction('教程')
        fitMenu.triggered.connect(self.open_fit)
        predMenu.triggered.connect(self.open_pred)
        saveMenu.triggered.connect(self.save_model)
        loadMenu.triggered.connect(self.load_model)
        savePredMenu.triggered.connect(self.save_pred)
        aboutAction.triggered.connect(self.about)
        tutorAction.triggered.connect(self.tutorial)

        plot_area = QWidget(self)
        self.setCentralWidget(plot_area)
        self.plot_area = QVBoxLayout(plot_area)
        self.plot_area.setContentsMargins(0, 0, 0, 0)

        self.statusBar().showMessage('ready')

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Prophet')
        self.finishedcount = 0
        self.artifact = dict()

        self.plotToolbar = None
        self.plotCanvas = None

        self.show()

    def about(self, event):
        QMessageBox.information(self, 'About Prophet', '版本0.1')

    def plot(self, Y=None, Y_hat=None, Y_labels=None):
        # clear plot area
        plt.clf()
        try:
            self.plot_area.removeWidget(self.plotCanvas)
            self.removeToolBar(self.plotToolbar)
        except Exception:
            pass
        # plot
        fig, ax = plt.subplots()
        fpath = os.path.join('./simkai.ttf')
        prop = fm.FontProperties(fname=fpath)
        if Y is not None:
            for i in range(Y.shape[1]):
                y = Y[:, i]
                ax.plot(y, label=Y_labels[i])
        if Y_hat is not None:
            for i in range(Y_hat.shape[1]):
                y = Y_hat[:, i]
                ax.plot(y, label='预测_'+Y_labels[i])
        ax.legend(prop=prop)
        self.plotCanvas = FigureCanvas(fig)
        self.plotToolbar = NavigationToolbar2QT(self.plotCanvas, self)
        self.plot_area.addWidget(self.plotCanvas)
        # add toolbar
        self.addToolBar(Qt.BottomToolBarArea, self.plotToolbar)

    def open_file(self, conf_key='open_dir', filter_ext='xlsx (*.xlsx)'):
        global cf
        filepath, filter_ = QFileDialog.getOpenFileName(
            self, '选取文件', cf['Personal']['open_dir'], filter_ext)
        if not filepath:
            return filepath
        cf['Personal'][conf_key] = os.path.dirname(filepath)
        return filepath

    def open_fit(self, event):
        filepath = self.open_file()
        if not filepath:
            return
        df = pd.read_excel(filepath)
        print(df)
        params = []
        for c in df.columns:
            p = [True, c, 1, False]
            params.append(p)
        params[-1][-1] = True
        self.sub = paramDialog(params)

        if not self.sub.exec_():
            return

        params = self.sub.get_params()
        print(params)

        X_params = list(filter(lambda t: t[0] and not t[-1], params))
        Y_params = list(filter(lambda t: t[0] and t[-1], params))

        X_labels = [p[1] for p in X_params]
        X_powers = [p[2] for p in X_params]
        Y_labels = [p[1] for p in Y_params]
        self.artifact['X_labels'] = X_labels
        self.artifact['Y_labels'] = Y_labels
        self.artifact['X_powers'] = X_powers

        X = df[X_labels]
        Y = df[Y_labels]

        print(X)
        print(Y)
        kwargs = {
            'power': X_powers
        }

        model = RegisteredModel('多项式模型')(**kwargs)
        X = X.to_numpy()
        Y = Y.to_numpy()
        r = model.fit(X, Y)
        Y_hat = model.pred(X)
        self.artifact['model'] = model
        self.statusBar().showMessage('拟合完毕，图中显示在该数据集上的拟合效果')
        self.plot(Y=Y, Y_hat=Y_hat, Y_labels=Y_labels)

    def open_pred(self, event):
        filepath = self.open_file()
        if not filepath:
            return
        df = pd.read_excel(filepath)
        X = df[self.artifact['X_labels']]
        X = X.to_numpy()
        Y_hat = self.artifact['model'].pred(X)
        self.Y_hat = Y_hat
        self.plot(Y_hat=Y_hat, Y_labels=self.artifact['Y_labels'])

    def save_model(self, event):
        filepath = QFileDialog.getSaveFileName(self, '保存模型', 'model.bin')
        if filepath is None or len(filepath) < 2 or filepath[0] is None or filepath[0] == '':
            return
        filepath = filepath[0]
        with open(filepath, 'wb') as f:
            pk.dump(self.artifact, f)
        self.statusBar().showMessage('模型保存完毕"{}"'.format(filepath))

    def save_pred(self, event):
        df_y = pd.DataFrame(
            self.Y_hat, columns=self.artifact['Y_labels'])
        filepath = QFileDialog.getSaveFileName(self, '保存预测', 'pred.xlsx')
        if filepath is None or len(filepath) < 2 or filepath[0] is None or filepath[0] == '':
            return
        filepath = filepath[0]
        df_y.to_excel(filepath, index=False)
        self.statusBar().showMessage('预测保存完毕"{}"'.format(filepath))

    def load_model(self, event):
        filepath = self.open_file('load_dir', 'bin (*.bin)')
        if not filepath:
            return
        with open(filepath, 'rb') as f:
            self.artifact = pk.load(f)
        self.statusBar().showMessage('模型"{}"加载完毕'.format(filepath))

    def closeEvent(self, event):
        global conf, cf
        reply = QMessageBox.warning(
            self, "温馨提示", "即将退出, 确定？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if(reply == QMessageBox.Yes):
            event.accept()
        if(reply == QMessageBox.No):
            event.ignore()
        event.accept()
        with open(conf, 'wt') as f:
            cf.write(f)

    def tutorial(self, event):
        text = '''
        流程：
        1. 先点击“拟合/打开文件”用来打开数据文件，弹出窗口选中相关列并设置相关参数后点击确定则会自动训练模型拟合数据。
        2. 点击保存模型保存已经训练好的模型到文件。
        3. 预测时先加载模型然后打开要预测的文件。

        数据文件： 只支持xlsx类型的文件，第一行必须是列名，列名不能有重复的，只支持回归任务。
        '''
        QMessageBox.information(self, '教程', text)


if __name__ == '__main__':
    maindir = os.path.dirname(sys.argv[0])
    conf = os.path.join(maindir, 'conf.ini')
    cf = configparser.ConfigParser()
    cf.read(conf)
    if 'Personal' not in cf:
        cf.add_section('Personal')
    for remembered_dir in ['open_dir', 'load_dir']:
        if remembered_dir not in cf['Personal']:
            cf['DEFAULT'][remembered_dir] = '~/'

    app = QApplication(sys.argv)
    ex = Prophet()
    try:
        status = app.exec_()
    except:
        print('错误')
        logging.error("错误:{}".format(traceback.format_exc()))
        status = -1
    sys.exit(status)
