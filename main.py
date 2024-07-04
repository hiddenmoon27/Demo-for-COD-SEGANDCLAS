# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1191, 784)
        Form.setStyleSheet("#Form{\n"
"    border-image:url(:/主页面背景图/wallhaven-433glv_1920x1200.png);}")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(140, 140, 291, 381))
        self.label.setStyleSheet("border-image: url(:/实例/实例.jpg);\n"
"")
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(500, 250, 110, 110))
        self.pushButton.setStyleSheet("QPushButton{\n"
"border:1px solid red;   /*边框的粗细，颜色*/\n"
"border-radius:55px;    /*设置圆角半径 */\n"
"padding:2px 4px;  /*QFrame边框与内部其它部件的距离*/\n"
"background-color: rgba(255, 184, 103, 155);    /*背景颜色*/\n"
"color:white;        /*字体颜色*/\n"
"min-width:100px;    /*设置最小宽度*/\n"
"min-height:100px;    /*设置最小高度*/\n"
"font:bold 14px;        /*设置按钮文字和大小*/\n"
"}")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/run.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon)
        self.pushButton.setIconSize(QtCore.QSize(40, 40))
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(660, 140, 291, 381))
        self.label_2.setStyleSheet("border-image: url(:/实例/实例.jpg);\n"
"")
        self.label_2.setObjectName("label_2")
        self.textBrowser = QtWidgets.QTextBrowser(Form)
        self.textBrowser.setGeometry(QtCore.QRect(150, 570, 791, 101))
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(150, 20, 160, 87))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icon/file-export.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_2.setIcon(icon1)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.comboBox = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.verticalLayout.addWidget(self.comboBox)
        self.comboBox_2 = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.verticalLayout.addWidget(self.comboBox_2)

        self.retranslateUi(Form)
        self.pushButton_2.clicked.connect(Form.openimage) # type: ignore
        self.pushButton.clicked.connect(Form.Convert) # type: ignore
        self.comboBox.currentIndexChanged['int'].connect(Form.select) # type: ignore
        self.comboBox_2.currentIndexChanged['int'].connect(Form.showim) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "TextLabel"))
        self.pushButton.setText(_translate("Form", "Convert"))
        self.label_2.setText(_translate("Form", "TextLabel"))
        self.pushButton_2.setText(_translate("Form", "load"))
        self.comboBox.setItemText(0, _translate("Form", "SInet"))
        self.comboBox.setItemText(1, _translate("Form", "b"))
        self.comboBox.setItemText(2, _translate("Form", "c"))
        self.comboBox.setItemText(3, _translate("Form", "d"))
        self.comboBox_2.setItemText(0, _translate("Form", "示例1"))
        self.comboBox_2.setItemText(1, _translate("Form", "示例2"))
        self.comboBox_2.setItemText(2, _translate("Form", "示例3"))
import mainpage_rc
