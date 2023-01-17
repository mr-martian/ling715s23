#!/usr/bin/env python3

def matrix_zeros(a, b):
    return [[0]*b]*a

def copy_matrix(mat):
    ret = []
    for row in mat:
        cur = []
        for col in row:
            cur.append(col)
        ret.append(cur)
    return ret

def print_matrix(mat):
    print('[', end='')
    for i, row in enumerate(mat):
        if i != 0:
            print(' ', end='')
        print('[', end='')
        for j, col in enumerate(row):
            if j != 0:
                print(', ', end='')
            print(col, end='')
        print(']', end='')
        if i+1 == len(mat):
            print(']\n', end='')
        else:
            print(',\n', end='')

def matrix_add(a, b):
    ret = []
    if len(a) != len(b):
        raise ValueError('matrices with different dimensions cannot be added')
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            raise ValueError('matrices with different dimensions cannot be added')
        ret.append([i+j for i,j in zip(ra, rb)])
    return ret

def column(mat, n):
    return [row[n] for row in mat]

def dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError('cannot dot product vectors of different lengths')
    return sum(a*b for a,b in zip(v1, v2))

def matrix_multiply(a, b):
    ret = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            row.append(dot_product(column(b, j), a[i]))
        ret.append(row)
    return ret

def transpose(mat):
    return [column(mat, i) for i in range(len(mat[0]))]

def matrix_flatten(mat):
    ret = []
    for row in mat:
        ret += row
    return ret

if __name__ == '__main__':
    print('Zeros')
    mat = matrix_zeros(4, 2)
    print_matrix(mat)
    print('Addition')
    A = [[1, 4], [2, 1]]
    B = [[6, 4], [4, 8]]
    C = matrix_add(A, B)
    print_matrix(C)
    print('Multiplication')
    A = [[1, 4, 3], 
         [2, 1, 5]]
    B = [[6, 4], 
         [4, 8], 
         [3, 5]]
    C = matrix_multiply(A, B)
    print_matrix(C)
    print('Transposition')
    A = [[6, 4],
         [4, 8],
         [3, 5]]
    B = transpose(A)
    print_matrix(B)
    print('Flatten')
    M = [[6, 4], 
         [4, 8], 
         [3, 5]]
    print(matrix_flatten(M))
