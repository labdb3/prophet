from PyQt5.Qt import QDialog
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QRadioButton
)


class PolynomialModelParamDialog(QDialog):
    def __init__(self, params):
        '''
        params: [(checked, colname, power, is_y)]
        '''
        super(PolynomialModelParamDialog, self).__init__()

        header = ['选取', '列名', '阶数', '预测']

        self.setWindowTitle('自定义参数')

        cols = []
        for p in params:
            checkbox = QCheckBox()
            checkbox.setChecked(p[0])
            col_name = QLabel(p[1])
            spinbox = QSpinBox()
            spinbox.setMinimum(1)
            cb_y = QCheckBox()
            cb_y.setChecked(p[3])
            cols.append([checkbox, col_name, spinbox, cb_y])

        self.buttonOK = QPushButton('Ok')
        self.buttonCancel = QPushButton('cancel')

        params_lay = QGridLayout()
        lay = QVBoxLayout()
        rowCount = len(cols)
        for c, tip in enumerate(header):
            tip = QLabel(tip)
            params_lay.addWidget(tip, 0, c)
        for r in range(rowCount):
            for c in range(len(cols[r])):
                params_lay.addWidget(cols[r][c], r+1, c)

        self.params_lay = params_lay
        lay.addLayout(params_lay)

        lay.addWidget(self.buttonOK)
        lay.addWidget(self.buttonCancel)
        self.setLayout(lay)
        self.buttonOK.clicked.connect(self.accept)
        self.buttonCancel.clicked.connect(self.reject)

    def get_params(self):
        param_values = layout_children(self.params_lay)
        params = []
        colCount = self.params_lay.columnCount()
        for r in range(1, self.params_lay.rowCount()):
            cb, col_name, sb, cb_y = param_values[r*colCount:(r+1)*colCount]
            p = (cb.isChecked(), col_name.text(), sb.value(), cb_y.isChecked())
            params.append(p)
        return params


class GreyModel11ParamDialog(QDialog):
    def __init__(self, params):
        '''
        params: [(checked, colname)]
        '''
        super(GreyModel11ParamDialog, self).__init__()

        self.setWindowTitle('选择要预测的列')

        params_lay = QVBoxLayout()
        for checked, colname in params:
            rb = QRadioButton(colname)
            params_lay.addWidget(rb)

        self.buttonOK = QPushButton('Ok')
        self.buttonCancel = QPushButton('cancel')

        lay = QVBoxLayout()
        self.params_lay = params_lay
        lay.addLayout(params_lay)

        lay.addWidget(self.buttonOK)
        lay.addWidget(self.buttonCancel)
        self.setLayout(lay)
        self.buttonOK.clicked.connect(self.accept)
        self.buttonCancel.clicked.connect(self.reject)

    def get_params(self):
        param_values = layout_children(self.params_lay)
        params = []
        for p in param_values:
            p = (p.isChecked(), p.text())
            params.append(p)
        return params


def layout_children(layout):
    ws = []
    for i in range(layout.count()):
        w = layout.itemAt(i).widget()
        ws.append(w)
    return ws
