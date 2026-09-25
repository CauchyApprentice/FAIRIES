import numpy as np
from Settings import parameter, Setting

class Func:
    def __init__(self):
        self.CTM_temp = 0.67938
        self.CTM_E0 = 0.77987

    def isint(self,val):
        try:
            int(val)
            if val == int(val):
                return True
            else:
                return False
        except:
            return False

    def isfloat(self, val):
        try:
            float(val)
            return True
        except:
            return False

    def spin_co_sqr(self):
        return 0.0145* parameter[Setting.g_nAMass] ** (5/3) * self.CTM_temp

    def rho_E(self, E):
        return 1/self.CTM_temp * np.exp((E-self.CTM_E0) / self.CTM_temp)

    def rho_Pi(self):
        return 1/2

    def rho_J(self, J = 1, E = 0):
        return (2*J+1)/(2*self.spin_co_sqr()) * np.exp(-(J + 1/2) ** 2 / (2*self.spin_co_sqr()))

    def rho(self, E, J = 1):
        return self.rho_E(E) * self.rho_J(J) * self.rho_Pi()

func = Func()