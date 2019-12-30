import pickle
import inspect

class Test(object):

    def __new__(cls, **kwargs):
        stack = inspect.stack()
        for i, frame in enumerate(stack):
            for j, val in enumerate(frame):
                print(f'[{i}][{j}] = {val},', end='\t')
            print('')
        if stack[1][3] == '__new__':
            # return super(Test, cls).__new__(cls)
            return object.__new__(cls)
        name = kwargs['name']
        if name == 'john':
            print("using cached object")
            with open("cp.p", "rb") as f:
                obj = pickle.load(f)
                print("object unpickled")
                return obj
        else:
            print("using new object")
            return super(Test, cls).__new__(cls)

    def __init__(self, name):
        print("calling __init__")
        self.name = name
        with open("cp.p", "wb") as f:
            pickle.dump(self, f)


a = Test(name = 'jim')
b = Test(name = 'john')