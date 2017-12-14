#! /usr/bin/python3

import numpy


def gmres(A, b):
  bnorm = numpy.linalg.norm(b)

  K = numpy.array([b / bnorm])
  H = []
  Q = [numpy.array([-1.])]

  while True:
    n = len(Q)

    # update K
    if len(K) == n:
      K.resize((2*n, len(b)), refcheck=False)
    K[n] = A(K[n-1])
    h1 = K[:n].dot(K[n])
    K[n] -= h1.dot(K[:n])
    h2 = numpy.linalg.norm(K[n])  
    K[n] /= h2  

    # update H
    H.append(numpy.hstack([h1, h2]))

    # update Q (givens rotations)
    q = Q.pop()
    qh1 = q.dot(h1)
    norm = numpy.sqrt(qh1**2 + h2**2)
    c = qh1 / norm
    s = h2 / norm
    Q.append(numpy.hstack([q*c, s]))
    Q.append(numpy.hstack([q*-s, c]))

    # solve Q H y = Q[:,0] using back substitution
    y = numpy.empty(n)
    for i in range(n-1, -1, -1):
      m = [Q[i].dot(H[j][:i+2]) for j in range(i, n)]
      y[i] = (Q[i][0] * bnorm - y[i+1:].dot(m[1:])) / m[0]

    # construct solution x = y K
    yield y.dot(K[:n])


def arnoldi(A, b):
  D = numpy.empty((1, 2, len(b))) # initialize empty krylov space
  k = b # use right hand side as first krylov vector
  n = 0
  while True:
    if len(D) == n:
      D.resize((n*2, 2, len(b)), refcheck=False)
    D[n] = k, A(k) # add new krylov vector
    D[n] -= D[:n].T.dot(D[:n,1].dot(D[n,1])).T # orthogonalize new krylov vector
    D[n] /= numpy.linalg.norm(D[n,1]) # normalize new krylov vector
    x, Ax = D[:n+1].T.dot(D[:n+1,1].dot(b)).T # project x onto krylov space
    yield x
    k = Ax - b # use residual as new krylov vector
    n += 1
