from src.temp_singleton_test1 import Test, T


def test_stuff():
    t = Test()
    T('test_stuff', 5)

    print(f'test_stuff on inst: {t.class_var}, {t.inst_var}')
    t.class_var += 5
    t.inst_var += 5
    print(f'test_stuff on inst after incrementing: {t.class_var}, {t.inst_var}')


