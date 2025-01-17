# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(988, 710)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Dialog)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget_menu = QtWidgets.QWidget(Dialog)
        self.widget_menu.setMinimumSize(QtCore.QSize(235, 0))
        self.widget_menu.setMaximumSize(QtCore.QSize(225, 16777215))
        self.widget_menu.setObjectName("widget_menu")
        self.layout_menu = QtWidgets.QVBoxLayout(self.widget_menu)
        self.layout_menu.setContentsMargins(0, 0, 0, 0)
        self.layout_menu.setObjectName("layout_menu")
        self.scrollArea = QtWidgets.QScrollArea(self.widget_menu)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, -107, 216, 712))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_signal_generation = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_signal_generation.setCheckable(True)
        self.pushButton_signal_generation.setChecked(True)
        self.pushButton_signal_generation.setObjectName("pushButton_signal_generation")
        self.verticalLayout_2.addWidget(self.pushButton_signal_generation)
        self.widget_loading_data = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.widget_loading_data.setObjectName("widget_loading_data")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget_loading_data)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(5)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget_signal_generation = QtWidgets.QWidget(self.widget_loading_data)
        self.widget_signal_generation.setObjectName("widget_signal_generation")
        self.layout_signal_generation = QtWidgets.QVBoxLayout(self.widget_signal_generation)
        self.layout_signal_generation.setContentsMargins(0, -1, 0, 5)
        self.layout_signal_generation.setObjectName("layout_signal_generation")
        self.groupBox_building_signal = QtWidgets.QGroupBox(self.widget_signal_generation)
        self.groupBox_building_signal.setObjectName("groupBox_building_signal")
        self.layout_building_signal = QtWidgets.QVBoxLayout(self.groupBox_building_signal)
        self.layout_building_signal.setContentsMargins(6, 0, 0, 5)
        self.layout_building_signal.setSpacing(0)
        self.layout_building_signal.setObjectName("layout_building_signal")
        self.radioButton_AM = QtWidgets.QRadioButton(self.groupBox_building_signal)
        self.radioButton_AM.setChecked(True)
        self.radioButton_AM.setAutoRepeat(False)
        self.radioButton_AM.setAutoExclusive(True)
        self.radioButton_AM.setObjectName("radioButton_AM")
        self.layout_building_signal.addWidget(self.radioButton_AM)
        self.radioButton_FM2 = QtWidgets.QRadioButton(self.groupBox_building_signal)
        self.radioButton_FM2.setAutoRepeat(False)
        self.radioButton_FM2.setAutoExclusive(True)
        self.radioButton_FM2.setObjectName("radioButton_FM2")
        self.layout_building_signal.addWidget(self.radioButton_FM2)
        self.radioButton_MCHM = QtWidgets.QRadioButton(self.groupBox_building_signal)
        self.radioButton_MCHM.setAutoRepeat(False)
        self.radioButton_MCHM.setAutoExclusive(True)
        self.radioButton_MCHM.setObjectName("radioButton_MCHM")
        self.layout_building_signal.addWidget(self.radioButton_MCHM)
        self.layout_signal_generation.addWidget(self.groupBox_building_signal)
        self.pushButton_generate_reference_signal = QtWidgets.QPushButton(self.widget_signal_generation)
        self.pushButton_generate_reference_signal.setObjectName("pushButton_generate_reference_signal")
        self.layout_signal_generation.addWidget(self.pushButton_generate_reference_signal)
        self.groupBox_noise = QtWidgets.QGroupBox(self.widget_signal_generation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_noise.sizePolicy().hasHeightForWidth())
        self.groupBox_noise.setSizePolicy(sizePolicy)
        self.groupBox_noise.setObjectName("groupBox_noise")
        self.layout_noise = QtWidgets.QVBoxLayout(self.groupBox_noise)
        self.layout_noise.setContentsMargins(0, 5, 0, 9)
        self.layout_noise.setSpacing(5)
        self.layout_noise.setObjectName("layout_noise")
        self.widget_noise = QtWidgets.QWidget(self.groupBox_noise)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_noise.sizePolicy().hasHeightForWidth())
        self.widget_noise.setSizePolicy(sizePolicy)
        self.widget_noise.setObjectName("widget_noise")
        self.layout_noise_text = QtWidgets.QHBoxLayout(self.widget_noise)
        self.layout_noise_text.setContentsMargins(-1, 0, -1, -1)
        self.layout_noise_text.setObjectName("layout_noise_text")
        self.label_text_noise = QtWidgets.QLabel(self.widget_noise)
        self.label_text_noise.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_noise.setObjectName("label_text_noise")
        self.layout_noise_text.addWidget(self.label_text_noise)
        self.lineEdit_noise = QtWidgets.QLineEdit(self.widget_noise)
        self.lineEdit_noise.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_noise.setObjectName("lineEdit_noise")
        self.layout_noise_text.addWidget(self.lineEdit_noise)
        self.layout_noise_text.setStretch(0, 1)
        self.layout_noise_text.setStretch(1, 1)
        self.layout_noise.addWidget(self.widget_noise)
        self.layout_signal_generation.addWidget(self.groupBox_noise)
        self.pushButton_generation_investigated_signal = QtWidgets.QPushButton(self.widget_signal_generation)
        self.pushButton_generation_investigated_signal.setObjectName("pushButton_generation_investigated_signal")
        self.layout_signal_generation.addWidget(self.pushButton_generation_investigated_signal)
        self.verticalLayout_3.addWidget(self.widget_signal_generation)
        self.verticalLayout_2.addWidget(self.widget_loading_data)
        self.pushButton_parameters = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_parameters.setCheckable(True)
        self.pushButton_parameters.setChecked(True)
        self.pushButton_parameters.setObjectName("pushButton_parameters")
        self.verticalLayout_2.addWidget(self.pushButton_parameters)
        self.groupBox_parameters = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_parameters.sizePolicy().hasHeightForWidth())
        self.groupBox_parameters.setSizePolicy(sizePolicy)
        self.groupBox_parameters.setObjectName("groupBox_parameters")
        self.layout_parameters = QtWidgets.QVBoxLayout(self.groupBox_parameters)
        self.layout_parameters.setContentsMargins(0, 5, 0, 9)
        self.layout_parameters.setSpacing(5)
        self.layout_parameters.setObjectName("layout_parameters")
        self.groupBox_frequency = QtWidgets.QGroupBox(self.groupBox_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_frequency.sizePolicy().hasHeightForWidth())
        self.groupBox_frequency.setSizePolicy(sizePolicy)
        self.groupBox_frequency.setObjectName("groupBox_frequency")
        self.layout_frequency = QtWidgets.QVBoxLayout(self.groupBox_frequency)
        self.layout_frequency.setContentsMargins(0, 5, 0, 9)
        self.layout_frequency.setSpacing(5)
        self.layout_frequency.setObjectName("layout_frequency")
        self.widget_sampling_rate = QtWidgets.QWidget(self.groupBox_frequency)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_sampling_rate.sizePolicy().hasHeightForWidth())
        self.widget_sampling_rate.setSizePolicy(sizePolicy)
        self.widget_sampling_rate.setObjectName("widget_sampling_rate")
        self.layout_sampling_rate = QtWidgets.QHBoxLayout(self.widget_sampling_rate)
        self.layout_sampling_rate.setContentsMargins(-1, 0, -1, -1)
        self.layout_sampling_rate.setObjectName("layout_sampling_rate")
        self.label_text_sampling_rate = QtWidgets.QLabel(self.widget_sampling_rate)
        self.label_text_sampling_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_sampling_rate.setObjectName("label_text_sampling_rate")
        self.layout_sampling_rate.addWidget(self.label_text_sampling_rate)
        self.lineEdit_sampling_rate = QtWidgets.QLineEdit(self.widget_sampling_rate)
        self.lineEdit_sampling_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_sampling_rate.setObjectName("lineEdit_sampling_rate")
        self.layout_sampling_rate.addWidget(self.lineEdit_sampling_rate)
        self.layout_sampling_rate.setStretch(0, 1)
        self.layout_sampling_rate.setStretch(1, 1)
        self.layout_frequency.addWidget(self.widget_sampling_rate)
        self.widget_frequency_carrier = QtWidgets.QWidget(self.groupBox_frequency)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_frequency_carrier.sizePolicy().hasHeightForWidth())
        self.widget_frequency_carrier.setSizePolicy(sizePolicy)
        self.widget_frequency_carrier.setObjectName("widget_frequency_carrier")
        self.layout_frequency_carrier = QtWidgets.QHBoxLayout(self.widget_frequency_carrier)
        self.layout_frequency_carrier.setContentsMargins(-1, 0, -1, -1)
        self.layout_frequency_carrier.setObjectName("layout_frequency_carrier")
        self.label_text_frequency_carrier = QtWidgets.QLabel(self.widget_frequency_carrier)
        self.label_text_frequency_carrier.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_frequency_carrier.setObjectName("label_text_frequency_carrier")
        self.layout_frequency_carrier.addWidget(self.label_text_frequency_carrier)
        self.lineEdit_frequency_carrier = QtWidgets.QLineEdit(self.widget_frequency_carrier)
        self.lineEdit_frequency_carrier.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_frequency_carrier.setObjectName("lineEdit_frequency_carrier")
        self.layout_frequency_carrier.addWidget(self.lineEdit_frequency_carrier)
        self.layout_frequency_carrier.setStretch(0, 1)
        self.layout_frequency_carrier.setStretch(1, 1)
        self.layout_frequency.addWidget(self.widget_frequency_carrier)
        self.layout_parameters.addWidget(self.groupBox_frequency)
        self.widget_bit_counts = QtWidgets.QWidget(self.groupBox_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_bit_counts.sizePolicy().hasHeightForWidth())
        self.widget_bit_counts.setSizePolicy(sizePolicy)
        self.widget_bit_counts.setObjectName("widget_bit_counts")
        self.layout_bit_counts = QtWidgets.QHBoxLayout(self.widget_bit_counts)
        self.layout_bit_counts.setContentsMargins(-1, 0, -1, -1)
        self.layout_bit_counts.setObjectName("layout_bit_counts")
        self.label_text_bit_counts = QtWidgets.QLabel(self.widget_bit_counts)
        self.label_text_bit_counts.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_bit_counts.setObjectName("label_text_bit_counts")
        self.layout_bit_counts.addWidget(self.label_text_bit_counts)
        self.lineEdit_bit_counts = QtWidgets.QLineEdit(self.widget_bit_counts)
        self.lineEdit_bit_counts.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_bit_counts.setObjectName("lineEdit_bit_counts")
        self.layout_bit_counts.addWidget(self.lineEdit_bit_counts)
        self.layout_bit_counts.setStretch(0, 1)
        self.layout_bit_counts.setStretch(1, 1)
        self.layout_parameters.addWidget(self.widget_bit_counts)
        self.widget_transfer_rate = QtWidgets.QWidget(self.groupBox_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_transfer_rate.sizePolicy().hasHeightForWidth())
        self.widget_transfer_rate.setSizePolicy(sizePolicy)
        self.widget_transfer_rate.setObjectName("widget_transfer_rate")
        self.layout_transfer_rate = QtWidgets.QHBoxLayout(self.widget_transfer_rate)
        self.layout_transfer_rate.setContentsMargins(-1, 0, -1, -1)
        self.layout_transfer_rate.setObjectName("layout_transfer_rate")
        self.label_text_transfer_rate = QtWidgets.QLabel(self.widget_transfer_rate)
        self.label_text_transfer_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_transfer_rate.setObjectName("label_text_transfer_rate")
        self.layout_transfer_rate.addWidget(self.label_text_transfer_rate)
        self.lineEdit_transfer_rate = QtWidgets.QLineEdit(self.widget_transfer_rate)
        self.lineEdit_transfer_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_transfer_rate.setObjectName("lineEdit_transfer_rate")
        self.layout_transfer_rate.addWidget(self.lineEdit_transfer_rate)
        self.layout_parameters.addWidget(self.widget_transfer_rate)
        self.widget_time_delay = QtWidgets.QWidget(self.groupBox_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_time_delay.sizePolicy().hasHeightForWidth())
        self.widget_time_delay.setSizePolicy(sizePolicy)
        self.widget_time_delay.setObjectName("widget_time_delay")
        self.layout_time_delay = QtWidgets.QHBoxLayout(self.widget_time_delay)
        self.layout_time_delay.setContentsMargins(-1, 0, -1, -1)
        self.layout_time_delay.setObjectName("layout_time_delay")
        self.label_text_time_delay = QtWidgets.QLabel(self.widget_time_delay)
        self.label_text_time_delay.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_time_delay.setObjectName("label_text_time_delay")
        self.layout_time_delay.addWidget(self.label_text_time_delay)
        self.lineEdit_time_delay = QtWidgets.QLineEdit(self.widget_time_delay)
        self.lineEdit_time_delay.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_time_delay.setObjectName("lineEdit_time_delay")
        self.layout_time_delay.addWidget(self.lineEdit_time_delay)
        self.layout_parameters.addWidget(self.widget_time_delay)
        self.widget_time_clippings_ms = QtWidgets.QWidget(self.groupBox_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_time_clippings_ms.sizePolicy().hasHeightForWidth())
        self.widget_time_clippings_ms.setSizePolicy(sizePolicy)
        self.widget_time_clippings_ms.setObjectName("widget_time_clippings_ms")
        self.layout_time_clippings_ms = QtWidgets.QHBoxLayout(self.widget_time_clippings_ms)
        self.layout_time_clippings_ms.setContentsMargins(-1, 0, -1, -1)
        self.layout_time_clippings_ms.setObjectName("layout_time_clippings_ms")
        self.label_text_time_clippings_ms = QtWidgets.QLabel(self.widget_time_clippings_ms)
        self.label_text_time_clippings_ms.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_time_clippings_ms.setObjectName("label_text_time_clippings_ms")
        self.layout_time_clippings_ms.addWidget(self.label_text_time_clippings_ms)
        self.lineEdit_time_clippings_ms = QtWidgets.QLineEdit(self.widget_time_clippings_ms)
        self.lineEdit_time_clippings_ms.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_time_clippings_ms.setObjectName("lineEdit_time_clippings_ms")
        self.layout_time_clippings_ms.addWidget(self.lineEdit_time_clippings_ms)
        self.layout_parameters.addWidget(self.widget_time_clippings_ms)
        self.groupBox_AM_manipulation = QtWidgets.QGroupBox(self.groupBox_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_AM_manipulation.sizePolicy().hasHeightForWidth())
        self.groupBox_AM_manipulation.setSizePolicy(sizePolicy)
        self.groupBox_AM_manipulation.setObjectName("groupBox_AM_manipulation")
        self.layout_AM_manipulation = QtWidgets.QVBoxLayout(self.groupBox_AM_manipulation)
        self.layout_AM_manipulation.setContentsMargins(0, 5, 0, 9)
        self.layout_AM_manipulation.setSpacing(5)
        self.layout_AM_manipulation.setObjectName("layout_AM_manipulation")
        self.widget_amplitude_am = QtWidgets.QWidget(self.groupBox_AM_manipulation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_amplitude_am.sizePolicy().hasHeightForWidth())
        self.widget_amplitude_am.setSizePolicy(sizePolicy)
        self.widget_amplitude_am.setObjectName("widget_amplitude_am")
        self.layout_amplitude_am = QtWidgets.QHBoxLayout(self.widget_amplitude_am)
        self.layout_amplitude_am.setContentsMargins(-1, 0, -1, -1)
        self.layout_amplitude_am.setObjectName("layout_amplitude_am")
        self.label_text_amplitude_am = QtWidgets.QLabel(self.widget_amplitude_am)
        self.label_text_amplitude_am.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_amplitude_am.setObjectName("label_text_amplitude_am")
        self.layout_amplitude_am.addWidget(self.label_text_amplitude_am)
        self.lineEdit_amplitude_am = QtWidgets.QLineEdit(self.widget_amplitude_am)
        self.lineEdit_amplitude_am.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_amplitude_am.setObjectName("lineEdit_amplitude_am")
        self.layout_amplitude_am.addWidget(self.lineEdit_amplitude_am)
        self.layout_amplitude_am.setStretch(0, 1)
        self.layout_amplitude_am.setStretch(1, 1)
        self.layout_AM_manipulation.addWidget(self.widget_amplitude_am)
        self.widget_coefficient_am = QtWidgets.QWidget(self.groupBox_AM_manipulation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_coefficient_am.sizePolicy().hasHeightForWidth())
        self.widget_coefficient_am.setSizePolicy(sizePolicy)
        self.widget_coefficient_am.setObjectName("widget_coefficient_am")
        self.layout_coefficient_am = QtWidgets.QHBoxLayout(self.widget_coefficient_am)
        self.layout_coefficient_am.setContentsMargins(-1, 0, -1, -1)
        self.layout_coefficient_am.setObjectName("layout_coefficient_am")
        self.label_text_coefficient_am = QtWidgets.QLabel(self.widget_coefficient_am)
        self.label_text_coefficient_am.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_coefficient_am.setObjectName("label_text_coefficient_am")
        self.layout_coefficient_am.addWidget(self.label_text_coefficient_am)
        self.lineEdit_coefficient_am = QtWidgets.QLineEdit(self.widget_coefficient_am)
        self.lineEdit_coefficient_am.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_coefficient_am.setObjectName("lineEdit_coefficient_am")
        self.layout_coefficient_am.addWidget(self.lineEdit_coefficient_am)
        self.layout_coefficient_am.setStretch(0, 1)
        self.layout_coefficient_am.setStretch(1, 1)
        self.layout_AM_manipulation.addWidget(self.widget_coefficient_am)
        self.layout_parameters.addWidget(self.groupBox_AM_manipulation)
        self.groupBox_MCHM_manipulation = QtWidgets.QGroupBox(self.groupBox_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_MCHM_manipulation.sizePolicy().hasHeightForWidth())
        self.groupBox_MCHM_manipulation.setSizePolicy(sizePolicy)
        self.groupBox_MCHM_manipulation.setObjectName("groupBox_MCHM_manipulation")
        self.layout_MCHM_manipulation = QtWidgets.QVBoxLayout(self.groupBox_MCHM_manipulation)
        self.layout_MCHM_manipulation.setContentsMargins(0, 5, 0, 9)
        self.layout_MCHM_manipulation.setSpacing(5)
        self.layout_MCHM_manipulation.setObjectName("layout_MCHM_manipulation")
        self.widget_frequency_deviations = QtWidgets.QWidget(self.groupBox_MCHM_manipulation)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_frequency_deviations.sizePolicy().hasHeightForWidth())
        self.widget_frequency_deviations.setSizePolicy(sizePolicy)
        self.widget_frequency_deviations.setObjectName("widget_frequency_deviations")
        self.layout_frequency_deviations = QtWidgets.QHBoxLayout(self.widget_frequency_deviations)
        self.layout_frequency_deviations.setContentsMargins(-1, 0, -1, -1)
        self.layout_frequency_deviations.setObjectName("layout_frequency_deviations")
        self.label_text_frequency_deviations = QtWidgets.QLabel(self.widget_frequency_deviations)
        self.label_text_frequency_deviations.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_frequency_deviations.setObjectName("label_text_frequency_deviations")
        self.layout_frequency_deviations.addWidget(self.label_text_frequency_deviations)
        self.lineEdit_frequency_deviations = QtWidgets.QLineEdit(self.widget_frequency_deviations)
        self.lineEdit_frequency_deviations.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_frequency_deviations.setObjectName("lineEdit_frequency_deviations")
        self.layout_frequency_deviations.addWidget(self.lineEdit_frequency_deviations)
        self.layout_MCHM_manipulation.addWidget(self.widget_frequency_deviations)
        self.layout_parameters.addWidget(self.groupBox_MCHM_manipulation)
        self.verticalLayout_2.addWidget(self.groupBox_parameters)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layout_menu.addWidget(self.scrollArea)
        self.pushButton_estimate_delay = QtWidgets.QPushButton(self.widget_menu)
        self.pushButton_estimate_delay.setObjectName("pushButton_estimate_delay")
        self.layout_menu.addWidget(self.pushButton_estimate_delay)
        self.widget_estimated_delay = QtWidgets.QWidget(self.widget_menu)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_estimated_delay.sizePolicy().hasHeightForWidth())
        self.widget_estimated_delay.setSizePolicy(sizePolicy)
        self.widget_estimated_delay.setObjectName("widget_estimated_delay")
        self.layout_estimated_delay = QtWidgets.QHBoxLayout(self.widget_estimated_delay)
        self.layout_estimated_delay.setContentsMargins(-1, 0, -1, -1)
        self.layout_estimated_delay.setObjectName("layout_estimated_delay")
        self.label_text_estimated_delay = QtWidgets.QLabel(self.widget_estimated_delay)
        self.label_text_estimated_delay.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_estimated_delay.setObjectName("label_text_estimated_delay")
        self.layout_estimated_delay.addWidget(self.label_text_estimated_delay)
        self.label_deviation_estimated_delay_ms = QtWidgets.QLabel(self.widget_estimated_delay)
        self.label_deviation_estimated_delay_ms.setText("")
        self.label_deviation_estimated_delay_ms.setAlignment(QtCore.Qt.AlignCenter)
        self.label_deviation_estimated_delay_ms.setObjectName("label_deviation_estimated_delay_ms")
        self.layout_estimated_delay.addWidget(self.label_deviation_estimated_delay_ms)
        self.layout_menu.addWidget(self.widget_estimated_delay)
        self.widget_number_experiments = QtWidgets.QWidget(self.widget_menu)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_number_experiments.sizePolicy().hasHeightForWidth())
        self.widget_number_experiments.setSizePolicy(sizePolicy)
        self.widget_number_experiments.setObjectName("widget_number_experiments")
        self.layout_number_experiments = QtWidgets.QHBoxLayout(self.widget_number_experiments)
        self.layout_number_experiments.setContentsMargins(-1, 0, -1, -1)
        self.layout_number_experiments.setObjectName("layout_number_experiments")
        self.label_text_number_experiments = QtWidgets.QLabel(self.widget_number_experiments)
        self.label_text_number_experiments.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_number_experiments.setObjectName("label_text_number_experiments")
        self.layout_number_experiments.addWidget(self.label_text_number_experiments)
        self.lineEdit_number_experiments = QtWidgets.QLineEdit(self.widget_number_experiments)
        self.lineEdit_number_experiments.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_number_experiments.setObjectName("lineEdit_number_experiments")
        self.layout_number_experiments.addWidget(self.lineEdit_number_experiments)
        self.layout_number_experiments.setStretch(0, 1)
        self.layout_number_experiments.setStretch(1, 1)
        self.layout_menu.addWidget(self.widget_number_experiments)
        self.pushButton_probability_of_detection = QtWidgets.QPushButton(self.widget_menu)
        self.pushButton_probability_of_detection.setObjectName("pushButton_probability_of_detection")
        self.layout_menu.addWidget(self.pushButton_probability_of_detection)
        self.progressBar_probability = QtWidgets.QProgressBar(self.widget_menu)
        self.progressBar_probability.setProperty("value", 0)
        self.progressBar_probability.setObjectName("progressBar_probability")
        self.layout_menu.addWidget(self.progressBar_probability)
        self.widget_execution_time = QtWidgets.QWidget(self.widget_menu)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_execution_time.sizePolicy().hasHeightForWidth())
        self.widget_execution_time.setSizePolicy(sizePolicy)
        self.widget_execution_time.setObjectName("widget_execution_time")
        self.layout_execution_time = QtWidgets.QHBoxLayout(self.widget_execution_time)
        self.layout_execution_time.setContentsMargins(-1, 0, -1, -1)
        self.layout_execution_time.setObjectName("layout_execution_time")
        self.label_text_execution_time = QtWidgets.QLabel(self.widget_execution_time)
        self.label_text_execution_time.setAlignment(QtCore.Qt.AlignCenter)
        self.label_text_execution_time.setObjectName("label_text_execution_time")
        self.layout_execution_time.addWidget(self.label_text_execution_time)
        self.label_execution_time = QtWidgets.QLabel(self.widget_execution_time)
        self.label_execution_time.setText("")
        self.label_execution_time.setAlignment(QtCore.Qt.AlignCenter)
        self.label_execution_time.setObjectName("label_execution_time")
        self.layout_execution_time.addWidget(self.label_execution_time)
        self.layout_menu.addWidget(self.widget_execution_time)
        self.horizontalLayout.addWidget(self.widget_menu)
        self.widget_main_1 = QtWidgets.QWidget(Dialog)
        self.widget_main_1.setObjectName("widget_main_1")
        self.verticalLayout_1 = QtWidgets.QVBoxLayout(self.widget_main_1)
        self.verticalLayout_1.setObjectName("verticalLayout_1")
        self.widget_plot = QtWidgets.QWidget(self.widget_main_1)
        self.widget_plot.setObjectName("widget_plot")
        self.layout_plot = QtWidgets.QVBoxLayout(self.widget_plot)
        self.layout_plot.setObjectName("layout_plot")
        self.verticalLayout_1.addWidget(self.widget_plot)
        self.widget_plot_3 = QtWidgets.QWidget(self.widget_main_1)
        self.widget_plot_3.setObjectName("widget_plot_3")
        self.layout_plot_3 = QtWidgets.QVBoxLayout(self.widget_plot_3)
        self.layout_plot_3.setObjectName("layout_plot_3")
        self.verticalLayout_1.addWidget(self.widget_plot_3)
        self.horizontalLayout.addWidget(self.widget_main_1)
        self.widget_main = QtWidgets.QWidget(Dialog)
        self.widget_main.setObjectName("widget_main")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget_main)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget_plot_2 = QtWidgets.QWidget(self.widget_main)
        self.widget_plot_2.setObjectName("widget_plot_2")
        self.layout_plot_2 = QtWidgets.QVBoxLayout(self.widget_plot_2)
        self.layout_plot_2.setObjectName("layout_plot_2")
        self.verticalLayout.addWidget(self.widget_plot_2)
        self.widget_plot_4 = QtWidgets.QWidget(self.widget_main)
        self.widget_plot_4.setObjectName("widget_plot_4")
        self.layout_plot_4 = QtWidgets.QVBoxLayout(self.widget_plot_4)
        self.layout_plot_4.setObjectName("layout_plot_4")
        self.verticalLayout.addWidget(self.widget_plot_4)
        self.horizontalLayout.addWidget(self.widget_main)

        self.retranslateUi(Dialog)
        self.pushButton_signal_generation.clicked['bool'].connect(self.widget_loading_data.setVisible) # type: ignore
        self.pushButton_parameters.clicked['bool'].connect(self.groupBox_parameters.setVisible) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton_signal_generation.setText(_translate("Dialog", "Генерация сигнала"))
        self.groupBox_building_signal.setTitle(_translate("Dialog", "Выбор построения сигнала"))
        self.radioButton_AM.setText(_translate("Dialog", "AM манипуляция"))
        self.radioButton_FM2.setText(_translate("Dialog", "ФМ2 манипуляция"))
        self.radioButton_MCHM.setText(_translate("Dialog", "МЧМ манипуляция"))
        self.pushButton_generate_reference_signal.setText(_translate("Dialog", "Сгенерировать опорный сигнал"))
        self.groupBox_noise.setTitle(_translate("Dialog", "Шум для исследуемого сигнала"))
        self.label_text_noise.setText(_translate("Dialog", "дБ"))
        self.lineEdit_noise.setText(_translate("Dialog", "-10"))
        self.pushButton_generation_investigated_signal.setText(_translate("Dialog", "Сгенерировать исследуемый сигнал"))
        self.pushButton_parameters.setText(_translate("Dialog", "Параметры"))
        self.groupBox_parameters.setTitle(_translate("Dialog", "Параметры "))
        self.groupBox_frequency.setTitle(_translate("Dialog", "Частота"))
        self.label_text_sampling_rate.setText(_translate("Dialog", "Частота дискр."))
        self.lineEdit_sampling_rate.setText(_translate("Dialog", "500"))
        self.label_text_frequency_carrier.setText(_translate("Dialog", "Несущая частота"))
        self.lineEdit_frequency_carrier.setText(_translate("Dialog", "20"))
        self.label_text_bit_counts.setText(_translate("Dialog", "Число бит"))
        self.lineEdit_bit_counts.setText(_translate("Dialog", "100"))
        self.label_text_transfer_rate.setText(_translate("Dialog", "V перед. данных"))
        self.lineEdit_transfer_rate.setText(_translate("Dialog", "20"))
        self.label_text_time_delay.setText(_translate("Dialog", "Времен. задержка, мс"))
        self.lineEdit_time_delay.setText(_translate("Dialog", "1920"))
        self.label_text_time_clippings_ms.setText(_translate("Dialog", "Обрезка сигнала, мс"))
        self.lineEdit_time_clippings_ms.setText(_translate("Dialog", "1500"))
        self.groupBox_AM_manipulation.setTitle(_translate("Dialog", "Параметры для АМ манипуляции"))
        self.label_text_amplitude_am.setText(_translate("Dialog", "Амплитуда"))
        self.lineEdit_amplitude_am.setText(_translate("Dialog", "10"))
        self.label_text_coefficient_am.setText(_translate("Dialog", "Коэффициент"))
        self.lineEdit_coefficient_am.setText(_translate("Dialog", "5"))
        self.groupBox_MCHM_manipulation.setTitle(_translate("Dialog", "Параметры для МЧМ манипуляции"))
        self.label_text_frequency_deviations.setText(_translate("Dialog", "Частота отклонения"))
        self.lineEdit_frequency_deviations.setText(_translate("Dialog", "0.5"))
        self.pushButton_estimate_delay.setText(_translate("Dialog", "Оценка временной задержки"))
        self.label_text_estimated_delay.setText(_translate("Dialog", "Оценка врем. задер."))
        self.label_text_number_experiments.setText(_translate("Dialog", "Количество экспериментов"))
        self.lineEdit_number_experiments.setText(_translate("Dialog", "100"))
        self.pushButton_probability_of_detection.setText(_translate("Dialog", "Провести исследование"))
        self.label_text_execution_time.setText(_translate("Dialog", "Время выполнения:"))
