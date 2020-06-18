# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\XiaohanYuan\autoseg-system\model\auto_heart_seg.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("dialog")
        dialog.resize(524, 165)
        font = QtGui.QFont()
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        dialog.setFont(font)
        self.input_label = QtWidgets.QLabel(dialog)
        self.input_label.setGeometry(QtCore.QRect(20, 60, 111, 16))
        self.input_label.setObjectName("input_label")
        self.output_label = QtWidgets.QLabel(dialog)
        self.output_label.setGeometry(QtCore.QRect(20, 90, 111, 16))
        self.output_label.setObjectName("output_label")
        self.input_lineEdit = QtWidgets.QLineEdit(dialog)
        self.input_lineEdit.setGeometry(QtCore.QRect(140, 60, 261, 20))
        self.input_lineEdit.setObjectName("input_lineEdit")
        self.output_lineEdit = QtWidgets.QLineEdit(dialog)
        self.output_lineEdit.setGeometry(QtCore.QRect(140, 90, 261, 20))
        self.output_lineEdit.setObjectName("output_lineEdit")
        self.title_label = QtWidgets.QLabel(dialog)
        self.title_label.setGeometry(QtCore.QRect(190, 20, 151, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(15)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.title_label.setFont(font)
        self.title_label.setTextFormat(QtCore.Qt.AutoText)
        self.title_label.setScaledContents(False)
        self.title_label.setObjectName("title_label")
        self.sure = QtWidgets.QPushButton(dialog)
        self.sure.setGeometry(QtCore.QRect(320, 130, 81, 23))
        self.sure.setObjectName("sure")
        self.answer = QtWidgets.QLabel(dialog)
        self.answer.setGeometry(QtCore.QRect(50, 130, 200, 23))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        self.answer.setFont(font)
        self.answer.setText("")
        self.answer.setObjectName("answer")
        self.pushButton_1 = QtWidgets.QPushButton(dialog)
        self.pushButton_1.setGeometry(QtCore.QRect(420, 60, 81, 23))
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_2 = QtWidgets.QPushButton(dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(420, 90, 81, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.cancel = QtWidgets.QPushButton(dialog)
        self.cancel.setGeometry(QtCore.QRect(420, 130, 81, 23))
        self.cancel.setObjectName("cancel")

        self.retranslateUi(dialog)
        self.cancel.clicked.connect(dialog.close)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("dialog", "Heart CT Segmentation"))
        self.input_label.setText(_translate("dialog", "输入NRRD文件路径"))
        self.output_label.setText(_translate("dialog", "输出文件所在路径"))
        self.title_label.setText(_translate("dialog", "心脏CT自动分割"))
        self.sure.setText(_translate("dialog", "确定"))
        self.pushButton_1.setText(_translate("dialog", "添加文件"))
        self.pushButton_2.setText(_translate("dialog", "添加文件夹"))
        self.cancel.setText(_translate("dialog", "取消"))

