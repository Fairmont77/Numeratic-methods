import numpy as np

class GEPP():
    def __init__(self, A, b, doPricing=True):
        self.A = A
        self.b = b
        self.doPricing = doPricing
        self.n = len(self.A)
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n)

        self._validate_input()
        self._elimination()
        self._backsub()

    def jacobi_method(self, tolerance=1e-10, max_iterations=1000):
        D = np.diag(self.A)
        R = self.A - np.diagflat(D)
        x = np.zeros_like(self.b)

        for iteration in range(max_iterations):
            x_new = (self.b - np.dot(R, x)) / D
            diff = np.linalg.norm(x_new - x, np.inf)
            if diff < tolerance:
                return x_new
            x = x_new
        return x

    def _validate_input(self):
        if self.b.size != self.n:
            raise ValueError("Invalid argument: incompatible sizes between A & b.", self.b.size, self.n)

    def _elimination(self):
        for k in range(self.n - 1):
            if self.doPricing:
                maxindex = abs(self.A[k:, k]).argmax() + k
                if self.A[maxindex, k] == 0:
                    raise ValueError("Matrix is singular.")
                if maxindex != k:
                    self.A[[k, maxindex]] = self.A[[maxindex, k]]
                    self.b[[k, maxindex]] = self.b[[maxindex, k]]
                    self.P[[k, maxindex]] = self.P[[maxindex, k]]  # Оновлення матриці P
            pivot = self.A[k, k]
            self.A[k] /= pivot
            self.b[k] /= pivot

            for row in range(k + 1, self.n):
                multiplier = self.A[row, k] / self.A[k, k]
                self.A[row, k:] -= multiplier * self.A[k, k:]
                self.b[row] -= multiplier * self.b[k]



    def _backsub(self):
        for k in range(self.n - 1, -1, -1):
            self.x[k] = (self.b[k] - np.dot(self.A[k, k + 1:], self.x[k + 1:])) / self.A[k, k]

def main():
    A = np.array([[3, 1, -2, -2],
                  [2, -1, 2, 2],
                  [2, 1, -1, -1],
                  [1, 1, -3, 2]], dtype='float64')

    b = np.array([-2, 2, -1, -3], dtype='float64')

    GaussElimPiv = GEPP(np.copy(A), np.copy(b), doPricing=True)
    print("Solution vector x:\n", GaussElimPiv.x)
    print("Transformed matrix M:\n", GaussElimPiv.A)
    print("Permutation matrix P:\n", GaussElimPiv.P)

    x_jacobi = GaussElimPiv.jacobi_method()
    print("Solution vector x using Jacobi method:\n", x_jacobi)

if __name__ == "__main__":
    main()
