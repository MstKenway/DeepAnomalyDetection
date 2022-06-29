def is_multiple(a, b):
    if a < b:
        a, b = b, a
    if a - 2 <= b * round(a / b) <= a + 2:
        return True
    return False
