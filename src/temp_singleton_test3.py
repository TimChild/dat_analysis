from src.temp_singleton_test2 import test_stuff
from src.temp_singleton_test1 import Test, T


if __name__ == '__main__':
    t = Test()
    T('page 3', 1)
    print(f'Page3 on inst: {t.class_var}, {t.inst_var}')
    t.class_var += 1
    t.inst_var += 1
    print(f'Page3 inst after_incrementing: {t.class_var}, {t.inst_var}')
    test_stuff()
    print(f'Page3 inst after_test_stuff: {t.class_var}, {t.inst_var}')
    T('page 3', 1)
    test_stuff()
    print(f'Page3 inst after_everything: {t.class_var}, {t.inst_var}')
