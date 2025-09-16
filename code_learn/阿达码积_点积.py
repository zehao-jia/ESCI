import torch

def test():
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    c = a.mul(b)
    print(c)

def mmd():# mm函数进行矩阵乘法
    a = torch.tensor([[1, 2, 3]])
    b = torch.tensor([[4],[5],[6]])
    c = a.mm(b)# 要求张量的维度都是二维
    print(c)

def bmmd():# bmm函数进行批量矩阵乘法,就是对批次内每个矩阵进行相乘
    a = torch.randn(10, 3, 4)
    b = torch.randn(10, 4, 5)
    c = a.bmm(b)# 要求张量的维度都是三维
    print(c)

def matmuld():
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b = torch.tensor([[7, 8], [9, 10], [11, 12]])
    c = torch.matmul(a, b)  # matmul函数可以进行矩阵乘法
    print(c)

if __name__ == "__main__":
    test()
    mmd()
    bmmd()
    matmuld()