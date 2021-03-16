from singleton_decorator import singleton

@singleton
class Test:
    class_var = 1

    def __init__(self):
        self.inst_var = 10

    def test(self, name, val):
        print(f'{name}: {self.class_var}')
        self.class_var += val
        print(f'{name} after incrementing: {self.class_var}')


T = Test().test




