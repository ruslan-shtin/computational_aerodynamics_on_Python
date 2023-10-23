import numpy as np

""" Все функции имеют одинаковую сигнатуру
    Вход:
        U: np.array
            Массив решения, от которого вычисляется пространственная производная
        mesh: Mesh
            Равномерная сетка, на которой вычисляется решение
    Выход:
        dUdx: np.array
            Массив пространственных производных в каждой точке сетки.
"""

def forward(U, mesh):
    """ метод прямого переноса 1-го порядка """
    dUdx = np.zeros_like(U)
    dUdx[:-1:1] = (U[1::1] - U[:-1:1]) / mesh.dx
    return dUdx

def backward(U, mesh):
    """ метод прямого переноса 1-го порядка """
    dUdx = np.zeros_like(U)
    dUdx[1::1] = (U[1::1] - U[:-1:1]) / mesh.dx
    return dUdx

def upwind2(U, mesh):
    """ Upwind 2-го порядка """
    dUdx = np.zeros_like(U)
    dUdx[2:] = (U[:-2] - 4 * U[1: -1] + 3 * U[2:]) / (2 * mesh.dx)
    return dUdx

dUdx_function_base = {
    "Forward" : forward,
    "Backward" : backward,
    "Upwind2" : upwind2,
    "Upwind3" : None,
    "Upwind5" : None,
    "CD2" : None,
    "CD4" : None,
    "CD6" : None,
}


def generate_dUdx_function(space_deriv_approx_method):
    """ Возвращает функцию, которая по текущему решению и сетке
        возвращает аппроксимацию пространственной производной.
        
        Вход:
            space_deriv_approx_method: str
                Название метода аппроксимации пространственной производной
        
        Выход:
            dUdx_calc_func: function
                Функция, вычисляющая пространственную производную.
                Функция двух переменных:
                    - U (вектор значений в узлах)
                    - mesh - сетка, на которой вычисляется решение
    """
    return dUdx_function_base[space_deriv_approx_method]