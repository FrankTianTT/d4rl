import math


def sigmoid(x, t=1e3, alpha=1.):
    x = x / t
    y = 1 / (1 + math.exp(-x / t))
    return alpha * y


def tanh(x, t=1., alpha=1.):
    x = x / t
    y = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    return alpha * y

def artanh(y, t=1., alpha=1.):
    y = y / alpha
    try:
        x = math.log((1 + y) / (1 - y), math.e) / 2
    except ValueError:
        print(y)
        x = 0
    x = x * t
    return x


def calculate_distance(pos, goal):
    return ((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2) ** 0.5


def line(x, x1, x2, y1, y2):
    return y2 - (x2 - x) * (y2 - y1) / (x2 - x1)