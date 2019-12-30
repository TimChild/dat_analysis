import pickle
import inspect


class testinputprint(object):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        inp = input('Write string to be repeated back:')
        print(inp)
        return None


if __name__ == '__main__':
    a = testinputprint()