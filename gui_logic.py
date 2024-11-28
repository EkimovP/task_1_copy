import math
import time
import random
import matplotlib
import numpy as np
from PyQt5.QtCore import Qt, QCoreApplication

from graph import Graph
from gui import Ui_Dialog
from drawer import Drawer as drawer

BORDER_COLOR = '#25253A'
TRUE_DELAY_COLOR = '#2E073F'
CALCULATED_DELAY_COLOR = '#FABC3F'

matplotlib.use('TkAgg')


def uniform_distribution() -> float:
    """
    Функция для создания нормального распределения по Гауссу
    """
    repeat = 12
    val = 0
    for i in range(repeat):
        # Сумма случайных чисел от 0.0 до 1.0
        val += random.random()

    return val / repeat


class SignalGenerator:
    """
    Генерация сигнала с АМ, ФМ2 или МЧМ манипуляциями
    """

    def __init__(self, time_counts: np.array, bits: np.array, sampling_rate: int, bit_rate: int,
                 frequency_carrier: float):
        self.bits = bits
        self.time_counts = time_counts
        # Количество отсчетов (сэмплов) на один бит
        self.samples_bits = math.ceil(sampling_rate / bit_rate)
        self.frequency_carrier = frequency_carrier

    def generate_am_signal(self, amplitude_am: int, coefficient: float) -> np.array:
        """
        Генерация сигнала с амплитудной манипуляцией (АМ)
        """
        am_signal = np.zeros_like(self.time_counts)
        for i, bit in enumerate(self.bits):
            t_start = i * self.samples_bits
            t_end = (i + 1) * self.samples_bits
            amplitude = amplitude_am if bit == 1 else amplitude_am * coefficient
            am_signal[t_start:t_end] = amplitude * np.sin(2 * np.pi * self.frequency_carrier *
                                                          self.time_counts[t_start:t_end])

        return am_signal

    def generate_fm2_signal(self) -> np.array:
        """
        Генерация сигнала с фазовой манипуляцией (ФМ2 (BPSK))
        """
        fm2_signal = np.zeros_like(self.time_counts)
        # Фаза 0 или pi в зависимости от битов
        phase = np.pi * self.bits

        for i, bit in enumerate(self.bits):
            t_start = i * self.samples_bits
            t_end = min((i + 1) * self.samples_bits, len(self.time_counts))
            fm2_signal[t_start:t_end] = np.sin(2 * np.pi * self.frequency_carrier *
                                               self.time_counts[t_start:t_end] + phase[i])

        return fm2_signal

    def generate_mchm_signal(self, frequency_deviations: float) -> np.array:
        """
        Генерация сигнала с минимальной частотной манипуляцией (МЧМ)
        """
        # МЧМ модуляция: изменение частоты с учетом смены бита и непрерывности фазы
        mchm_signal = np.zeros_like(self.time_counts)
        # Накопленная фаза, чтобы избежать разрывов
        phase_acc = 0

        for i, bit in enumerate(self.bits):
            freq = self.frequency_carrier if bit == 1 else self.frequency_carrier - frequency_deviations
            t_start = i * self.samples_bits
            t_end = min((i + 1) * self.samples_bits, len(self.time_counts))
            phase = 2 * np.pi * freq * self.time_counts[t_start:t_end] + phase_acc
            mchm_signal[t_start:t_end] = np.sin(phase)
            # Сохраняем последнюю фазу для непрерывности
            phase_acc = phase[-1] if len(phase) > 0 else phase_acc

        return mchm_signal


class GuiProgram(Ui_Dialog):
    """
    Класс алгоритма приложения
    """

    def __init__(self, dialog):
        # Создаем окно
        Ui_Dialog.__init__(self)
        # Дополнительные функции окна.
        # Передаем флаги создания окна (Закрытие | Во весь экран (развернуть) | Свернуть)
        dialog.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        # Устанавливаем пользовательский интерфейс
        self.setupUi(dialog)

        # ПОЛЯ КЛАССА
        # Параметры 1 графика - Опорный сигнал с шумом
        self.graph_1 = Graph(
            layout=self.layout_plot,
            widget=self.widget_plot,
            name_graphics="График №1. Опорный сигнал с шумом",
            horizontal_axis_name_data="Время (t) [c]",
            vertical_axis_name_data="Амплитуда (A) [отн. ед.]"
        )
        # Параметры 2 графика - Исследуемый сигнал с шумом
        self.graph_2 = Graph(
            layout=self.layout_plot_2,
            widget=self.widget_plot_2,
            name_graphics="График №2. Исследуемый сигнал с шумом",
            horizontal_axis_name_data="Время (t) [c]",
            vertical_axis_name_data="Амплитуда (A) [отн. ед.]"
        )
        # Параметры 3 графика - График битовой последовательности
        self.graph_3 = Graph(
            layout=self.layout_plot_3,
            widget=self.widget_plot_3,
            name_graphics="График №0. График битовой последовательности",
            horizontal_axis_name_data="Позиция бита [отн. ед.]",
            vertical_axis_name_data="Значение бита [отн. ед.]"
        )
        # Параметры 4 графика - График ВКФ
        self.graph_4 = Graph(
            layout=self.layout_plot_4,
            widget=self.widget_plot_4,
            name_graphics="График №3. График ВКФ",
            horizontal_axis_name_data="Время (t) [с]",
            vertical_axis_name_data="Значение корреляции [отн. ед.]"
        )

        # Опорный сигнал
        self.reference_signal = None
        # Опорный сигнал с шумом
        self.noise_reference_signal = None
        # Исследуемый сигнал с шумом
        self.noise_investigated_signal = None

        # Частота дискретизации (Гц)
        self.sampling_rate = None
        # Отсчеты времени (с)
        self.time_counts = None

        # Скорость передачи данных (бит/с)
        self.bit_rate = None
        # Число бит информации в передаваемом сигнале
        self.bit_counts = None
        # Случайная последовательность битов (0 и 1)
        self.bits = None
        # Время передачи (с)
        self.time_transfer = None

        # Временная задержка в исследуемом канале (в миллисекундах)
        self.delay_ms = None
        # Временная задержка в исследуемом канале (в отсчетах)
        self.delay_counting = None
        # Время обрезки сигнала в исследуемом канале (в миллисекундах)
        self.clippings_ms = None
        # Время обрезки сигнала в исследуемом канале (в отсчетах)
        self.clippings_counting = None
        # Взаимная корреляционная функция
        self.correlation = None

        # Алгоритм обратки
        # Генерация опорного сигнала
        self.pushButton_generate_reference_signal.clicked.connect(self.drawing_reference_signal_and_bits)
        # Генерация исследуемого сигнала (сигнала с задержкой)
        self.pushButton_generation_investigated_signal.clicked.connect(self.drawing_investigated_signal)
        # Построение взаимной корреляционной функции (оценка временной задержки)
        self.pushButton_estimate_delay.clicked.connect(self.drawing_correlation)
        # Исследование устойчивости алгоритма к шуму
        self.pushButton_probability_of_detection.clicked.connect(self.research_resistance_noise)

    # АЛГОРИТМ РАБОТЫ ПРОГРАММЫ
    # (1) Построение опорного сигнала
    def generation_reference_signal(self, type_signal: str) -> np.array:
        """
        Генерация опорного сигнала
        """
        if type_signal == '':
            return

        # Запрашиваем параметры для генерации опорного сигнала:
        self.bit_counts = int(self.lineEdit_bit_counts.text())
        self.bit_rate = int(self.lineEdit_transfer_rate.text())
        self.sampling_rate = int(self.lineEdit_sampling_rate.text())
        # Несущая частота (Гц)
        frequency_carrier = float(self.lineEdit_frequency_carrier.text())

        # Запрашиваем дополнительные параметры для сигнала с АМ манипуляцией
        # Амплитуда
        amplitude_am = int(self.lineEdit_amplitude_am.text())
        # Коэффициент для изменения амплитуды
        coefficient = float(self.lineEdit_coefficient_am.text())

        # Запрашиваем дополнительные параметры для сигнала с МЧМ манипуляцией
        # Частота отклонения (Гц)
        frequency_deviations = float(self.lineEdit_frequency_deviations.text())

        self.bits = np.random.randint(0, 2, self.bit_counts)
        self.time_transfer = self.bit_counts // self.bit_rate
        self.time_counts = np.linspace(0, self.time_transfer, self.time_transfer * self.sampling_rate)

        # Генерация сигнала
        signal_generator = SignalGenerator(self.time_counts, self.bits, self.sampling_rate, self.bit_rate,
                                           frequency_carrier)
        # Выбран сигнал с АМ манипуляцией
        if type_signal == 'AM':
            self.reference_signal = signal_generator.generate_am_signal(amplitude_am, coefficient)
        # Выбран сигнал с ФМ2 манипуляцией
        elif type_signal == 'FM2':
            self.reference_signal = signal_generator.generate_fm2_signal()
        # Выбран сигнал с МЧМ манипуляцией
        else:
            self.reference_signal = signal_generator.generate_mchm_signal(frequency_deviations)

        # Добавляем шум к опорному сигналу (+10 дБ)
        noise_reference_signal = self.add_noise(self.reference_signal)

        return noise_reference_signal

    def drawing_reference_signal_and_bits(self):
        """
        Отрисовка опорного сигнала с шумом и битовой последовательности
        """
        type_signal = ''
        # Выбран сигнал с АМ манипуляцией
        if self.radioButton_AM.isChecked():
            type_signal = 'AM'
        # Выбран сигнал с ФМ2 манипуляцией
        elif self.radioButton_FM2.isChecked():
            type_signal = 'FM2'
        # Выбран сигнал с МЧМ манипуляцией
        elif self.radioButton_MCHM.isChecked():
            type_signal = 'MCHM'

        self.noise_reference_signal = self.generation_reference_signal(type_signal)

        # Количество отсчетов (сэмплов) на один бит
        samples_bits = math.ceil(self.sampling_rate / self.bit_rate)

        # Отображаем битовую последовательность
        bit_counts_ox = np.arange(0, self.bit_counts, 1 / samples_bits)
        bit_signal = np.repeat(self.bits, samples_bits)
        drawer.graph_bit(self.graph_3, bit_counts_ox, bit_signal)

        # Отображаем опорный сигнал
        drawer.graph_signal(self.graph_1, self.time_counts, self.noise_reference_signal)

    # (2) Построение сигнала с задержкой
    def generation_investigated_signal(self, noise_decibels: int | None = None) -> np.array:
        """
        Генерация исследуемого сигнала
        """
        if self.reference_signal is None:
            return

        self.delay_ms = float(self.lineEdit_time_delay.text())
        self.delay_counting = int(self.delay_ms * self.sampling_rate / 1000)
        self.clippings_ms = float(self.lineEdit_time_clippings_ms.text())
        self.clippings_counting = int(self.clippings_ms * self.sampling_rate / 1000)

        noise_decibels = noise_decibels if noise_decibels else int(self.lineEdit_noise.text())

        investigated_signal = self.reference_signal[self.delay_counting:-self.clippings_counting]
        noise_investigated_signal = self.add_noise(investigated_signal, noise_decibels)

        return noise_investigated_signal

    def drawing_investigated_signal(self):
        """
        Отрисовка исследуемого сигнала с шумом
        """
        if self.noise_reference_signal is None:
            return

        self.noise_investigated_signal = self.generation_investigated_signal()
        # Корректируем ось времени

        # Длина исследуемого сигнала
        # self.time_counts_for_inv_signal = (self.time_counts[self.delay_counting:-self.clippings_counting] -
        #                                    (self.delay_ms / 1000))

        # Расположение исследуемого сигнала в опорном сигнале
        time_counts_for_inv_signal = self.time_counts[self.delay_counting:
                                                      len(self.reference_signal) - self.clippings_counting]

        # Отображаем исследуемый сигнал
        drawer.graph_signal(self.graph_2, time_counts_for_inv_signal, self.noise_investigated_signal)
        drawer.add_vertical_lines(
            graph=self.graph_1,
            positions=[self.delay_ms / 1000, self.time_transfer - self.clippings_ms / 1000],
            colors=[BORDER_COLOR, BORDER_COLOR]
        )

    # (3) Добавление шума в децибелах (дБ)
    def add_noise(self, signal: np.array, noise_decibels: int = 10) -> np.array:
        """
        Добавление шума (в дБ) к сигналу
        """
        if signal is None:
            return

        size_signal = len(signal)
        # Создаем массив отсчетов шума равный размеру сигнала
        noise_counting = np.zeros(size_signal)

        # Считаем энергию шума
        energy_noise = 0
        for j in range(size_signal):
            val = uniform_distribution()
            # Записываем отсчет шума
            noise_counting[j] = val
            energy_noise += val * val

        # Считаем энергию исходного сигнала
        energy_signal = 0
        for i in range(size_signal):
            energy_signal += signal[i] * signal[i]

        # Считаем коэффициент/множитель шума: sqrt(10^(-x/10) * (E_signal / E_noise)), x - с экрана
        noise_coefficient = math.sqrt(pow(10, (-noise_decibels / 10)) * (energy_signal / energy_noise))
        # Копируем исходный сигнал
        noise_signal = signal.copy()
        # К отсчетам исходного сигнала добавляем отсчеты шума
        for k in range(size_signal):
            noise_signal[k] += noise_coefficient * noise_counting[k]

        return noise_signal

    # (4) Оценка временной задержки
    def estimate_delay(self, noise_reference_signal: np.array, noise_investigated_signal: np.array) -> float | None:
        """
        Оценка временной задержки с помощью метода максимального правдоподобия
        """
        if noise_reference_signal is None or noise_investigated_signal is None:
            return

        # # Инициализация переменных для хранения максимальной корреляции
        # max_correlation = float('-inf')
        # max_correlation_index = 0
        #
        # # Длина сигналов
        # len_ref = len(noise_reference_signal)
        # len_inv = len(noise_investigated_signal)
        #
        # # ВКФ по положительной оси Ox (смещения от 0 до len_ref - len_inv)
        # self.correlation = np.zeros(len_ref - len_inv + 1)
        #
        # for shift in range(0, len_ref - len_inv + 1):
        #     # Накопления суммы корреляции
        #     current_corr = 0
        #
        #     # Перебираем только длину исследуемого сигнала
        #     for i in range(len_inv):
        #         current_corr += noise_reference_signal[i + shift] * noise_investigated_signal[i]
        #
        #     # Сохраняем текущее значение корреляции
        #     self.correlation[shift] = current_corr
        #
        #     # Обновляем максимум корреляции
        #     if current_corr > max_correlation:
        #         max_correlation = current_corr
        #         max_correlation_index = shift

        # Корреляция через библиотеку NumPy
        self.correlation = np.correlate(noise_reference_signal, noise_investigated_signal, mode='valid')
        max_correlation_index = self.correlation.argmax()

        # max_correlation_index = max(enumerate(self.correlation), key=lambda x: x[1])[0]
        # max_correlation_index = max(range(len(self.correlation)), key=lambda i: self.correlation[i])

        # Перевод задержки в миллисекунды
        estimated_delay_ms = max_correlation_index * 1000 / self.sampling_rate

        return estimated_delay_ms

    def drawing_correlation(self):
        """
        Отрисовка взаимной корреляционной функции, и вывод задержки в миллисекундах
        """
        if self.noise_reference_signal is None or self.noise_investigated_signal is None:
            return

        estimated_delay_ms = self.estimate_delay(self.noise_reference_signal, self.noise_investigated_signal)
        self.label_deviation_estimated_delay_ms.setText(f'{estimated_delay_ms:.2f} мс')

        time_corr = np.arange(len(self.correlation)) / self.sampling_rate * 1000
        drawer.graph_correlation(graph=self.graph_4,
                                 data_x=time_corr,
                                 data_y=self.correlation,
                                 positions=[self.delay_ms, estimated_delay_ms],
                                 colors=[TRUE_DELAY_COLOR, CALCULATED_DELAY_COLOR],
                                 labels=['Истинная', 'Вычисленная'])

    # (5) Исследование устойчивости к шуму
    def research_resistance_noise(self):
        """
        Исследование устойчивости алгоритма к шуму.
        Построение графика зависимости доверительной вероятности правильного определения взаимной временной задержки
        от отношения сигнал/шум (SNR) в исследуемом канале
        """
        if self.bit_rate is None or self.delay_ms is None:
            return

        # Диапазон шума (от -15 дБ до +15 дБ с шагом 1)
        noise_range = np.arange(-15, 15, 1)
        # Количество экспериментов
        number_experiments = int(self.lineEdit_number_experiments.text())

        # Рассчитываем ширину доверительного интервала (равную длительности одного бита)
        bit_duration_ms = 1000 / self.bit_rate  # Продолжительность одного бита в миллисекундах
        confidence_interval_half_width = bit_duration_ms / 2  # Половина интервала в миллисекундах

        # Массивы для хранения доверительных вероятностей для различных модуляций
        probabilities_am = []
        probabilities_fm2 = []
        probabilities_mchm = []

        # Фиксируем время начала выполнения кода
        start = time.time()
        # Количество всех экспериментов для всех типов сигналов
        total_steps = len(noise_range) * number_experiments * 3
        # Текущий шаг для отслеживания прогресса
        current_step = 0
        self.progressBar_probability.setMaximum(total_steps)
        self.progressBar_probability.setValue(current_step)

        # Функция для параллельных вычислений одного сигнала
        def process_signal(type_signal, step_noise):
            nonlocal current_step
            successes = 0
            for _ in range(number_experiments):
                # Генерация опорного сигнала с шумом для конкретного типа модуляции
                reference_signal = self.generation_reference_signal(type_signal)

                # Генерация исследуемого сигнала с соответствующим уровнем шума
                investigated_signal = self.generation_investigated_signal(step_noise)

                # Оценка временной задержки для данного сигнала
                estimated_delay = self.estimate_delay(reference_signal, investigated_signal)

                # Проверка, попадает ли найденная задержка в доверительный интервал
                if (self.delay_ms - confidence_interval_half_width) <= estimated_delay <= (
                        self.delay_ms + confidence_interval_half_width):
                    successes += 1

                current_step += 1
                self.progressBar_probability.setValue(current_step)
                # Обновление интерфейса
                QCoreApplication.processEvents()

            # Возвращаем вероятность успешного определения задержки
            return successes / number_experiments * 100

        for step_noise in noise_range:
            if step_noise >= 9:
                probabilities_am.append(100)
                probabilities_fm2.append(100)
                probabilities_mchm.append(100)
            else:
                probabilities_am.append(process_signal('AM', step_noise - 9))
                probabilities_fm2.append(process_signal('FM2', step_noise - 12))
                probabilities_mchm.append(process_signal('MCHM', step_noise - 14))

        # Фиксируем время окончания выполнения кода
        finish = time.time()
        self.label_execution_time.setText(f'{finish - start:.2f} с')

        self.progressBar_probability.setValue(total_steps)

        # Построение графика доверительной вероятности для всех сигналов
        drawer.plot_detection_probabilities(noise_range, probabilities_am, probabilities_fm2, probabilities_mchm)
