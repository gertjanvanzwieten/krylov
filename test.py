import krylov, numpy, unittest

class Test(unittest.TestCase):

  def setUp(self):
    numpy.random.seed(0)
    self.n = 20
    self.A = numpy.eye(self.n) + numpy.random.normal(size=[self.n]*2, scale=.5)
    self.x = numpy.random.normal(size=[self.n])
    self.b = self.A.dot(self.x)

  def _test_solver(self, solver, tol=1e-10):
    res0 = numpy.inf
    for i, x in enumerate(solver(self.A.dot, self.b)):
      res = numpy.linalg.norm(self.A.dot(x) - self.b)
      self.assertLess(res, res0)
      res0 = res
      if res < tol:
        break
      self.assertLess(i, self.n)
    err = numpy.linalg.norm(self.x - x)
    self.assertLess(err, tol * abs(numpy.linalg.det(self.A)))

  def test_arnoldi(self):
    self._test_solver(krylov.arnoldi)

  def test_gmres(self):
    self._test_solver(krylov.gmres)

  def test_gmres_arnoldi(self):
    for i, x1, x2 in zip(range(self.n), krylov.gmres(self.A.dot, self.b), krylov.arnoldi(self.A.dot, self.b)):
      err = numpy.linalg.norm(x1 - x2) / numpy.linalg.norm(self.x)
      self.assertLess(err, 1e-10)
