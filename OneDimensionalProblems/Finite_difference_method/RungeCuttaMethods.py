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
    butcher_table = {
        'c' : np.array([0, 1/3, 2/3], dtype=np.float32),
        'b' : np.array([1/4, 0, 3/4], dtype=np.float32),
        'a' : [np.array([1/3]),
               np.array([0, 2/3])
              ]
    }
    return runge_cutta_general_method(U_curr, t_curr, mesh, dt, right_function, butcher_table)


def runge_cutta_6(U_curr, t_curr, mesh, dt, right_function):
    """ Метод Рунге-Кутты 6-го порядка
    """
    butcher_table = {
        'c' : np.array([0, 1/3, 2/3, 1/3, 5/6, 1/6, 1], dtype=np.float32),
        'b' : np.array([13/200, 0, 11/40, 11/40, 4/25, 4/25, 13/200], dtype=np.float32),
        'a' : [np.array([1/3]),
               np.array([0, 2/3]),
               np.array([1/12, 1/3, -1/12]),
               np.array([25/48, -55/24, 35/48, 15/8]),
               np.array([3/20, -11/24, -1/8, 1/2, 1/10]),
               np.array([-261/260, 33/13, 43/156, -118/39, 32/195, 80/39]),
              ]
    }
    return runge_cutta_general_method(U_curr, t_curr, mesh, dt, right_function, butcher_table)


def runge_cutta_7(U_curr, t_curr, mesh, dt, right_function):
    """ Метод Ренге-Кутты 7-го порядка
    """
    butcher_table = {
        'c' : np.array([0, 1/6, 1/3, 1/2, 2/11, 2/3, 6/7, 0, 1], dtype=np.float32),
        'b' : np.array([0, 0, 0, 32/105, 1771561/6289920, 243/2560, 16807/74880, 77/1440, 11/270], dtype=np.float32),
        'a' : [np.array([1/6]),
               np.array([0, 1/3]),
               np.array([1/8, 0, 3/8]),
               np.array([148/1331, 0, 150/1331, -56/1331]),
               np.array([-404/243, 0, -170/27, 4024/1701, 10648/1701]),
               np.array([2466/2401, 0, 1242/343, -19176/16807, -51909/16807, 1053/2401]),
               np.array([5/154, 0, 0, 96/539, -1815/20384, -405/2464, 49/1144]),
               np.array([-113/32, 0, -195/22, 32/7, 29403/3584, -729/512, 1029/1408, 21/16]),
               ]
    }
    return runge_cutta_general_method(U_curr, t_curr, mesh, dt, right_function, butcher_table)



runge_cutta_funcions_base = {
    "Euler-1" : direct_euler,
    "Euler-2" : two_step_euler,
    "Hoin" : hoin,
    "RK-6" : runge_cutta_6,
    "RK-7" : runge_cutta_7,
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