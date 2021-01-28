import threading
from functools import wraps
import time


def synchronized(lock):
    def wrapper(f):
        @wraps(f)
        def inner_wrapper(*args, **kwargs):
            with lock:
                return f(*args, **kwargs)
        return inner_wrapper
    return wrapper


class Synchronized:
    def __init_subclass__(cls, **kwargs):
        sychronizer = synchronized(threading.RLock())
        for name in cls.__dict__:
            attr = getattr(cls, name)
            if callable(attr):
                setattr(cls, name, sychronizer(attr))


class BaseClass(Synchronized):
    def __init__(self, a=1):
        self.a = a

    def print(self):
        print(f'a = {self.a}')
        time.sleep(1)


class SubClass(BaseClass):
    def reentrant_method(self, i=0):
        print(f'a = {self.a}: i = {i}')
        time.sleep(0.5)
        if i < 3:
            self.reentrant_method(i+1)



if __name__ == '__main__':
    test1 = SubClass(1)
    test2 = SubClass(2)

    # t1 = threading.Thread(target=test1.print)
    # t2 = threading.Thread(target=test2.print)
    # t3 = threading.Thread(target=test1.print)
    t1 = threading.Thread(target=test1.reentrant_method)
    t2 = threading.Thread(target=test2.reentrant_method)
    t3 = threading.Thread(target=test1.reentrant_method)

    print('About to start all threads')

    for t in [t1, t2, t3]:
        t.start()
    print(f'All threads started')

    for t in [t1, t2, t3]:
        t.join()

    print(f'All threads Finished')





# def _get_method_names(obj):
#     """ Get all methods of a class or instance, inherited or otherwise. """
#     if type(obj) == types.InstanceType:
#         return _get_method_names(obj.__class__)
#     elif type(obj) == types.ClassType:
#         result = []
#         for name, func in obj.__dict__.items():
#             if type(func) == types.FunctionType:
#                 result.append((name, func))
#         for base in obj.__bases__:
#             result.extend(_get_method_names(base))
#         return result
#
#
# class _SynchronizedMethod:
#     """ Wrap lock and release operations around a method call. """
#
#     def __init__(self, method, obj, lock):
#         self.__method = method
#         self.__obj = obj
#         self.__lock = lock
#
#     def __call__(self, *args, **kwargs):
#         self.__lock.acquire()
#         try:
#             return self.__method(self.__obj, *args, **kwargs)
#         finally:
#             self.__lock.release()
#
#
# class SynchronizedObject:
#     """ Wrap all methods of an object into _SynchronizedMethod instances. """
#
#     def __init__(self, obj, ignore=None, lock=None):
#         if ignore is None:
#             ignore = []
#
#         # You must access __dict__ directly to avoid tickling __setattr__
#         self.__dict__['_SynchronizedObject__methods'] = {}
#         self.__dict__['_SynchronizedObject__obj'] = obj
#
#         if not lock:
#             lock = threading.RLock()
#         for name, method in _get_method_names(obj):
#             if not name in ignore and not name in self.__methods:
#                 self.__methods[name] = _SynchronizedMethod(method, obj, lock)
#
#     def __getattr__(self, name):
#         try:
#             return self.__methods[name]
#         except KeyError:
#             return getattr(self.__obj, name)
#
#     def __setattr__(self, name, value):
#         setattr(self.__obj, name, value)

