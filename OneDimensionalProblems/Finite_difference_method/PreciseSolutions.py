import numpy as np

def transport_eq_solution(init_cond, mesh, speed, total_time):
    """ Аналитическое решение уравнения переноса 
                    dU/dt + a * dU/dx = 0, a = const
        С начальным условием init_cond.

        Аналитическое решение: U(x,t) = U0(x - at)

        Вход:
            init_cond: function
                функция одного переменного x - начальное условие
            mesh: Mesh
                одномерная конечно-разностная сетка, на которой решается уравнение
            speed: float
                Параметр постоянной скорости
            total_time: float
                Физическое время, на котором нужно вычислить решение
        
        Выход:
            U_solution: np.array
                Массив значений решения в узлах сетки mesh.xnodes
    """
    U_solution = np.array([init_cond(x - speed * total_time) for x in mesh.xnodes])
    return U_solution