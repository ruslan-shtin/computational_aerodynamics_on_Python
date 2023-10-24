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

def get_linear_approximation_for_dUdx(U, mesh, indexes, coefs_list, dx_coef):
    """ Аппроксимация производной с помощью линейной компбинации.

        Вход:
            U: np.array
                Массив решения, от которого вычисляется пространственная производная
            mesh: Mesh
                Равномерная сетка, на которой вычисляется решение
            indexes : tuple(int, int)
                Кортеж из минимального и максимального индекса в шаблоне.
            coefs_list : list[float]
                список коэффициентов, с которыми нужно складывать элементы шаблона
                коэффициенты упорядочены по возрастанию индекса
            dx_coef : float
                коэффициент, на который в знаменателе нужно домножить dx
        Выход:
            dUdx: np.array
                Массив пространственных производных в каждой точке сетки.     
    """
    min_ind, max_ind = indexes
    assert (max_ind - min_ind + 1) == len(coefs_list)
    N = U.shape[0]
    start_ind = 0 - min_ind
    finish_ind = N - max_ind
    dUdx = np.zeros_like(U)
    for i, coef in enumerate(coefs_list):
        left_ind = start_ind + min_ind + i
        right_ind = finish_ind + min_ind + i 
        dUdx[start_ind : finish_ind] += coef * U[left_ind : right_ind]

    dUdx /= dx_coef * mesh.dx
    return dUdx



def forward(U, mesh):
    """ метод прямого переноса 1-го порядка """
    indexes = (0, 1)
    coefs_list = [-1, 1]
    dx_coef = 1
    return get_linear_approximation_for_dUdx(U, mesh, indexes, coefs_list, dx_coef)

def backward(U, mesh):
    """ метод прямого переноса 1-го порядка """
    indexes = (-1, 0)
    coefs_list = [-1, 1]
    dx_coef = 1
    return get_linear_approximation_for_dUdx(U, mesh, indexes, coefs_list, dx_coef)

def upwind2(U, mesh):
    """ Upwind 2-го порядка """
    indexes = (-2, 0)
    coefs_list = [1, -4, 3]
    dx_coef = 2
    return get_linear_approximation_for_dUdx(U, mesh, indexes, coefs_list, dx_coef)

def upwind3(U, mesh):
    """ Upwind 3-го порядка """
    indexes = (-2, 1)
    coefs_list = [1, -6, 3, 2]
    dx_coef = 6
    return get_linear_approximation_for_dUdx(U, mesh, indexes, coefs_list, dx_coef)

def upwind5(U, mesh):
    """ Upwind 5-го порядка """
    indexes = (-3, 2)
    coefs_list = [-2, 15, -60, 20, 30, -3]
    dx_coef = 60
    return get_linear_approximation_for_dUdx(U, mesh, indexes, coefs_list, dx_coef)

def central_diff2(U, mesh):
    """ Центральная разность 2-го порядка """
    indexes = (-1, 1)
    coefs_list = [-1, 0, 1]
    dx_coef = 2
    return get_linear_approximation_for_dUdx(U, mesh, indexes, coefs_list, dx_coef)

def central_diff4(U, mesh):
    """ Центральная разность 4-го порядка """
    indexes = (-2, 2)
    coefs_list = [1, 8, 0, 8, -1]
    dx_coef = 12
    return get_linear_approximation_for_dUdx(U, mesh, indexes, coefs_list, dx_coef)

def central_diff6(U, mesh):
    """ Центральная разность 6-го порядка """
    indexes = (-3, 3)
    coefs_list = [-1, 9, -45, 0, 45, -9, 1]
    dx_coef = 60
    return get_linear_approximation_for_dUdx(U, mesh, indexes, coefs_list, dx_coef)


dUdx_function_base = {
    "Forward" : forward,
    "Backward" : backward,
    "Upwind2" : upwind2,
    "Upwind3" : upwind3,
    "Upwind5" : upwind5,
    "CD2" : central_diff2,
    "CD4" : central_diff4,
    "CD6" : central_diff6,
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