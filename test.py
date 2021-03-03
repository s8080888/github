import numpy as np
class test1:

    k = 10

    def __init__(self):
        self.a = 0
        self.b = 0
        self.c = 0
        self.list = []

    def do(self):
        for i in range(2):
            self.add1()
            self.b = self.b+1

    def add1(self):
        self.add2()

    def add2(self):
        self.c = self.c + 1


def test3(k):
    d = k + 1
    if(d < 3):
        test3(d)
    else:
        print(d)

test3(0)


