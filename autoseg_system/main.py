from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from seg_ui import Ui_Dialog
import sys
from run import Process


class mywindow(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.pushButton_1.clicked.connect(self.read_file)
        self.pushButton_2.clicked.connect(self.write_folder)
        self.sure.clicked.connect(self.process)

    def read_file(self):
        # 选取文件
        filename, filetype = QFileDialog.getOpenFileName(self, "选取文件", "C:/", "All Files(*);;Text Files(*.csv)")
        print(filename, filetype)
        self.input_lineEdit.setText(filename)

    def write_folder(self):
        # 选取文件夹
        foldername = QFileDialog.getExistingDirectory(self, "选取文件夹", "C:/")
        print(foldername)
        self.output_lineEdit.setText(foldername)

    # 进行处理
    def process(self):
        try:
            # 获取文件路径
            filename = self.input_lineEdit.text()
            # 获取文件夹路径
            foldername = self.output_lineEdit.text()

            # 处理
            success_result = r'运行中！'
            self.answer.setText(success_result)
            time = Process(filename, foldername)
            success_result = r'成功！运行' + time + 's'
            self.answer.setText(success_result)

        except:
            fail_result = r'失败！请重试!'
            self.answer.setText(fail_result)


if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = mywindow()
    ui.show()
    sys.exit(app.exec_())