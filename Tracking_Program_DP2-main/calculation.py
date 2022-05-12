"""
CALCULATION.PY

Side file for hosting the functions involved in
mathematical analysis of data.
"""
import numpy as np
from typing import List
from sklearn.metrics import r2_score
from datamodels import Vector


def percent_diff(expected, actual) -> float:
    """
    Calculates the percent difference between 2 values

    Args:
        expected: Expected value
        actual: Real/acquired value

    Returns:
        Float value
    """
    sign = 1 if expected > actual else -1
    value = (abs(actual - expected) / ((actual + expected) / 2)) * 100
    return sign * round(value, 2)


def differentiate(data_set) -> List[Vector]:
    """
    Takes the captured data set and calculates the velocity
    using numerical differentiation.

    Args:
        data_set (list): List of values from video capture

    Returns:
        diff_data: Velocities along X and Y axes
    """
    initial_velocity = Vector(x=0, y=0)
    diff_data = [initial_velocity]
    for i in range(1, len(data_set)):
        prev, curr = data_set[i], data_set[i-1]
        x_p, y_p = prev.x, prev.y
        x_c, y_c = curr.x, curr.y
        t_c, t_p = curr.time, prev.time
        dx = round((x_c - x_p) / (t_c - t_p), 2)
        dy = round((y_c - y_p) / (t_c - t_p), 2)
        velocity = Vector(x=dx, y=dy)
        diff_data.append(velocity)

    return diff_data


def polynomial_data(x, y, deg: int = 2) -> dict:
    fit = np.polyfit(x, y, deg)
    polynomial = np.poly1d(fit)
    line = np.linspace(x[0], x[-1], max(y))
    poly_rel = round(r2_score(y, polynomial(x)), 4)
    coefficients = list(map(lambda c: float(c), fit))
    eq_comp = [
        f'{"+" if coefficients[i] > 0 else "-"} {abs(coefficients[i]):.3f}x^{deg - i}'
        for i in range(deg)
    ]
    poly_eq_form = ' '.join(eq_comp)

    return {
        'relation': poly_rel,
        'line': line,
        'polynomial': polynomial(line),
        'equation': poly_eq_form
    }
