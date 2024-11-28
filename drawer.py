import numpy as np
from graph import Graph
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# ШАБЛОНЫ ОТРИСОВКИ ГРАФИКОВ
def cleaning_and_chart_graph(graph: Graph, x_label: str, y_label: str, title: str):
    """
    Очистка и подпись графика (вызывается в начале)
    """
    # Возвращаем зум в домашнюю позицию
    graph.toolbar.home()
    # Очищаем стек осей (от старых x, y lim)
    graph.toolbar.update()
    # Очищаем график
    graph.axis.clear()
    # Задаем название осей
    graph.axis.set_xlabel(x_label)
    graph.axis.set_ylabel(y_label)
    # Задаем название графика
    graph.axis.set_title(title)


def clear_vertical_lines(graph: Graph):
    """
    Очистка вертикальных линий
    """
    # Проверяем, содержит ли объект (graph) атрибут (vertical_lines)
    if hasattr(graph, 'vertical_lines'):
        # Если да, то очищаем его
        for line in graph.vertical_lines:
            line.remove()
        graph.vertical_lines.clear()
    # Иначе инициализируем атрибут vertical_lines
    else:
        graph.vertical_lines = []


def draw_graph(graph: Graph):
    """
    Отрисовка (вызывается в конце)
    """
    # Убеждаемся, что все помещается внутри холста
    graph.figure.tight_layout()
    # Показываем новую фигуру в интерфейсе
    graph.canvas.draw()


class Drawer:
    """
    Класс художник. Имя холст (graph), рисует на нем данные
    """

    # Цвет графиков
    SIGNAL_COLOR = "#ff0000"
    BIT_COLOR = "#4682B4"
    CORRELATION_COLOR = "#419D78"

    # ОТРИСОВКИ
    @staticmethod
    def graph_signal(graph: Graph, data_x: np.array, data_y: np.array):
        """
        Отрисовка сигнала
        """
        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
            # Подпись осей
            x_label=graph.horizontal_axis_name_data, y_label=graph.vertical_axis_name_data
        )

        # Рисуем график
        graph.axis.plot(data_x, data_y, color=Drawer.SIGNAL_COLOR)
        # Отрисовка (вызывается в конце)
        draw_graph(graph)

    @staticmethod
    def graph_bit(graph: Graph, data_x: np.array, data_y: np.array):
        """
        Отрисовка битовой последовательности
        """
        # Очистка, подпись графика и осей
        cleaning_and_chart_graph(
            graph=graph,
            title=graph.name_graphics,
            x_label=graph.horizontal_axis_name_data,
            y_label=graph.vertical_axis_name_data
        )

        graph.axis.step(data_x, data_y, where='post', color=Drawer.BIT_COLOR)
        draw_graph(graph)

    @staticmethod
    def add_vertical_lines(graph: Graph, positions: list | float, colors: list | str):
        """
        Добавление вертикальных линий на график
        """
        # Очистка старых вертикальных линий
        clear_vertical_lines(graph)

        # Добавление новых вертикальных линий
        for pos, color in zip(positions, colors):
            line = graph.axis.axvline(x=pos, color=color, linestyle='solid', linewidth=3)
            graph.vertical_lines.append(line)

        draw_graph(graph)

    @staticmethod
    def graph_correlation(graph: Graph, data_x: np.array, data_y: np.array, positions: list | float,
                          colors: list | str, labels: list | str):
        """
        Отрисовка корреляции
        """
        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
            # Подпись осей
            x_label=graph.horizontal_axis_name_data, y_label=graph.vertical_axis_name_data
        )

        # Рисуем график
        graph.axis.plot(data_x, data_y, color=Drawer.CORRELATION_COLOR)
        # Добавляем вертикальные линии
        lines = []
        for pos, color, label in zip(positions, colors, labels):
            line = graph.axis.axvline(x=pos, color=color, linestyle='solid', linewidth=4)
            lines.append((line, label))

        # Создаем линии для легенды с описанием цветов
        legend_lines = [mlines.Line2D([], [], color=color, label=label)
                        for color, label in zip(colors, labels)]

        # Добавляем легенду
        legend = graph.axis.legend(handles=legend_lines, loc='upper left')

        # Добавляем обработчик кликов для скрытия/показа вертикальных линий
        def on_legend_click(event):
            for legend_line, (line, label) in zip(legend.get_lines(), lines):
                if legend_line == event.artist:
                    visible = not line.get_visible()
                    line.set_visible(visible)
                    legend_line.set_alpha(1.0 if visible else 0.2)
            graph.canvas.draw()

        # Привязываем обработчик событий к клику по элементам легенды
        graph.canvas.mpl_connect('pick_event', on_legend_click)

        # Включаем интерактивное выделение элементов легенды
        for legend_line in legend.get_lines():
            legend_line.set_picker(True)

        # Отрисовка (вызывается в конце)
        draw_graph(graph)

    @staticmethod
    def plot_detection_probabilities(noise_range: np.array, probabilities_am: list, probabilities_fm2: list,
                                     probabilities_mchm: list):
        """
        Построение графиков для исследования
        """
        plt.figure(figsize=(10, 6))

        # График для сигнала с амплитудной манипуляцией (АМ)
        plt.plot(noise_range, probabilities_am, label='AM (Амплитудная модуляция)', color='red', marker='x')

        # График для сигнала с фазовой манипуляцией (ФМ2 (BPSK))
        plt.plot(noise_range, probabilities_fm2, label='FM2 (Фазовая модуляция)', color='green', marker='s')

        # График для сигнала с минимальной частотной манипуляцией (МЧМ)
        plt.plot(noise_range, probabilities_mchm, label='MCHM (Частотная модуляция)', color='#412C84', marker='D')

        plt.title("Зависимость вероятности правильного определения задержки от SNR")
        plt.xlabel("Отношение сигнал/шум (SNR) [дБ]")
        plt.ylabel("Доверительная вероятность правильного определения [%]")
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.ylim(0, 100)

        plt.show()
