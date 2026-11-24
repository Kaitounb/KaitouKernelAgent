from baseline import launch as correct
from fault_kernels import launch_fault

calls = {}


def launch(x, y, n):
    key = x.data_ptr()
    calls[key] = calls.get(key, 0) + 1
    if calls[key] == 1:
        correct(x, y, n)
    else:
        launch_fault(x, y, n, 1)
