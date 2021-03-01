import numpy as np
class test1:
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


    def tt(self):
        self.do()
        self.add1()
        self.do()


k = [[0,0]]*2
k[0][0] += 1
print(k)

print()






