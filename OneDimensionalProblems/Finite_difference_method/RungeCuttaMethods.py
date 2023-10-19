""" Здесь переислены методы Рунге-Кутты
    для решения одномерной задачи Коши: 
                dU/dt = F(U, mesh, t)
    где:
        U - массив решения 
        mesh - одномерная сетка на которой ищется решение
        t - параметр времени
"""

import numpy as np

def runge_cutta_general_method(U_curr, t_curr, mesh, dt, right_function, butcher_table):
    """ Общее описание метода Рунге-Кутты

        Вход:
        U_curr: np.array
            Массив текущего решения во всех узал расчёной сетки
        t_curr: float
            Текущее время
        mesh: Mesh
            Сетка, на которой вычисляется решение
        dt: float
            Совершаемый шаг по времени
        right_function: function
            Функция правой части от 3 аргументов: mesh, t, U
        butcher_table: dict
            Таблица Бутчера. Ключи: 'a', 'b', 'c'.
            Ключам 'c' и 'b' соответсвуют одномерные массивы,
            Ключу 'a' соответсвует список массивов, размер которых
            последовательно увеличивается на 1
        
        Выход:
            U_new: np.array
                массив значений на новом временном слое
    """
    s = butcher_table['c'].shape[0]    # количество слоёв по времени
    k = np.zeros((s, U_curr.shape[0])) # список решений на промежуточных 
                                       # вспомогательных временных слоях

    k[0] = right_function(mesh, t_curr, U_curr)
    for i in range(1, s):
        a = np.array(butcher_table['a'][i-1])
        U_star = U_curr + dt * np.sum(a[:i].reshape((-1, 1)) * k[:i], axis=0)
        t_star = t_curr + butcher_table['c'] * dt
        k[i] = right_function(mesh, t_star, U_star)
  
    b = butcher_table['b']
    U_new = U_curr + dt*np.sum(b.reshape((-1, 1))*k, axis=0)
    return U_new


def direct_euler(U_curr, t_curr, mesh, dt, right_function):
    """ Прямой метод Эйлера
    """
    butcher_table = {
        'c' : np.array([0], dtype=np.float32),
        'b' : np.array([1.0], dtype=np.float32),
        'a' : [np.array([0.0]),
              ]
    }
    return runge_cutta_general_method(U_curr, t_curr, mesh, dt, right_function, butcher_table)


def two_step_euler(U_curr, t_curr, mesh, dt, right_function):
    """ Двухшаговый метод Эйлера
    """
    butcher_table = {
        'c' : np.array([0, 0.5], dtype=np.float32),
        'b' : np.array([0, 1.0], dtype=np.float32),
        'a' : [np.array([0.5]),
              ]
    }
    return runge_cutta_general_method(U_curr, t_curr, mesh, dt, right_function, butcher_table)


def hoin(U_curr, t_curr, mesh, dt, right_function):
    """ Трёхшаговый метод Хойна
    """
    pass


def runge_cutta_6(U_curr, t_curr, mesh, dt, right_function):
    """ Метод Рунге-Кутты 6-го порядка
    """
    pass


def runge_cutta_7(U_curr, t_curr, mesh, dt, right_function):
    """ Метод Ренге-Кутты 7-го порядка
    """
    pass




runge_cutta_funcions_base = {
    "Euler-1" : direct_euler,
    "Euler-2" : two_step_euler,
    "Hoin" : None,
    "RK-6" : None,
    "RK-7" : None,
}

def generate_get_next_function(time_step_method):
    """ Возвращает функцию, которая по решению на текущем временном слое
        возвращает решение на следующем временном слое.
        
        Вход:
            time_step_method: str
                Название метода аппроксимации временной производной
        
        Выход
            get_next: function
                Функция, которая принимает на вход 4 аргумента:
                    - текущее решение (вектор-значений в узлах)
                    - сетку, на которой вычисляется решение
                    - шаг по времени
                    - функцию правых частей, зависящую от mesh, t, U
    """
    return runge_cutta_funcions_base[time_step_method]