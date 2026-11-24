from baseline import add_one


def launch(x, y, n):
    # Incorrect floor division leaves the last 1 or 13 elements untouched.
    add_one[(n // 256,)](x, y, n, 256, num_warps=4)
