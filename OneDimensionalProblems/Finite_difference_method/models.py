import numpy as np

class Mesh:
    """ Одномерная равномерная сетка для метода конечных разностей.
    """
    
    def __init__(self, xleft, xright, N):
        """
        Создание одномерной равномерной сетки.
        
        Вход:
            xleft : float
                Левая граница сетки
            yleft : float
                Правая граница сетки
            N : int
                Число узлов в сетке
        """
        self.xleft = xleft
        self.xright = xright
        self.N = N
        
        self.xnodes = np.linspace(xleft, xright, N)
        self.dx = self.xnodes[1] - self.xnodes[0]

    def get_twice_grid(self):
        """ Метод, который создаёт сетку в два раза подробнее
        """
        new_N = self.N + (self.N - 1)
        return Mesh(self.xleft, self.xright, new_N)



class TaskParams:
    """ Описание параметров уравнения переноса.
    """
    def __init__(self, init_cond, speed_function, right_function):
        """ В уравнении переноса могут варьироваться:
                - начальное условие (init_cond)
                - скорость (speed_function)
                - функция правой части (right_function)
            
            Вход:
                init_cond : function
                    Функция одного переменного - координаты x,
                    возвращая значение функции в точке x в начальный момент времени
                speed_function : function
                    Функция двух переменных - сетки mesh и решения U,
                    возвращающая скорость на всей оси OX
                right_function : function
                    Функция правой части, зависящая от (mesh, t, U),
                    возвращающая значение правой части на всей оси OX
        """
        self.init_cond = init_cond
        self.speed_function = speed_function
        self.right_function = right_function
