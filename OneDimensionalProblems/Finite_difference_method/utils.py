import SpaceDerivApproxMethods
import RungeCuttaMethods

import numpy as np
from tqdm import tqdm


def get_dt(cur_t, total_time, mesh, Cu, U_curr):
    """ Вычисление временного шага для численного решения
        уравнения переноса.
    """
    dt = mesh.dx * Cu
    if cur_t + dt > total_time:
        dt = total_time - cur_t
    return dt


def get_init_field(mesh, init_cond):
    """ Пораждает дискретную проекцию начального условия на сетку.
    
        Вход:
            mesh: Mesh
                Сетка на которой вычисляется решение
            init_cond: function
                Функция одного переменного - начальное условие
        
        Выход:
            U: np.array
                Значения init_cond в узлах сетки mesh
    """
    U = np.array([init_cond(x) for x in mesh.xnodes])
    return U


def generate_right_function_for_dUdt_problem(task_params, space_deriv_approx_method):
    """ Создание правой функции для решения задачи dU/dt = F,
        Для нашей задачи F = f - speed*dU/dx.
        
        Вход:
            task_params: TaskParams
                Параметры уравнения переноса: скорость и функция правой части
            space_deriv_approx_method: str
                Название метода вычисления пространственной производной
        
        Выход:
            right_function_for_dUdt_problem: function
                Функция правой части, которую мы пошлём в метод Рунге-Кутты.
                Эта функция должна быть функцией 3-ёх аргументов: xmesh, t, U
    """
    dUdx_function = SpaceDerivApproxMethods.generate_dUdx_function(space_deriv_approx_method)
    
    def new_right_func(mesh, t, U):
        scr_right_func_arr = task_params.right_function(mesh, t, U)
        speed_dUdx_arr = task_params.speed_function(mesh, U) * dUdx_function(U, mesh)
        return scr_right_func_arr - speed_dUdx_arr
    
    return new_right_func



def main_runner(task_params, mesh, Cu, total_time, time_step_method, space_deriv_approx_method, N_iter_max):
    """ Основная функция для численного решения одномерного уравнения переноса.
        
        Вход:
            task_params: TaskParams
                Описание параметров уравнения переноса
            mesh: Mesh
                Сетка, на которой произиводится решение
            Cu: float
                Число Куранта
            total_time: float
                Физическое время, до которого нужно считать
            time_step_mathod: str
                Название метода аппроксимации временной производной
            space_deriv_approx_method: str
                Название метода аппроксимации производной по пространству
            N_iter_max : int
                Максимальное число итераций
        
        Выход:
            U: np.array
                Численное решение на сетке mesh.
                Массив длины mesh.N
            time_steps: np.array
                массив шагов по времени, чтобы после можно
                было проанализировать как работал алгоритм
    """
    U0 = get_init_field(mesh, task_params.init_cond)
    
    get_next = RungeCuttaMethods.generate_get_next_function(time_step_method)
    right_function_for_dUdt_problem = generate_right_function_for_dUdt_problem(task_params, space_deriv_approx_method)
    
    U_curr = U0.copy()
    U_new = U0.copy()
    
    _t = 0
    time_steps = list()
    for inter_num in tqdm(range(N_iter_max)):
        if _t >= total_time:
            break

        U_curr = U_new.copy()
        dt = get_dt(_t, total_time, mesh, Cu, U_curr)
        U_new = get_next(U_curr, _t, mesh, dt, right_function_for_dUdt_problem)
        time_steps.append(dt)
        _t += dt
    
    return U_new, time_steps