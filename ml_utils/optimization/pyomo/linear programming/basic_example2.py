import pyomo.environ as pyo 
from pyomo.opt import SolverFactory

"""
max 16a + 30b + 50c
s.t.
4a + 5b + 8c <= 112
2a + 4b + 5c <= 160
a + 2b + 3c <= 48

a, b, c >= 0
"""
opt = SolverFactory('glpk')

model = pyo.ConcreteModel()
model.a = pyo.Var(within=pyo.NonNegativeReals)
model.b = pyo.Var(within=pyo.NonNegativeReals)
model.c = pyo.Var(within=pyo.NonNegativeReals)
model.obj = pyo.Objective(expr=-(16 * model.a + 30 * model.b + 50 * model.c))
model.con1 = pyo.Constraint(expr=4 * model.a + 5 * model.b + 8 * model.c <= 112) 
model.con2 = pyo.Constraint(expr=2 * model.a + 4 * model.b + 5 * model.c <= 160)
model.con3 = pyo.Constraint(expr=model.a + 2 * model.b + 3 * model.c <= 48)

results = opt.solve(model)
model.display()
print('***** a = {} *****'.format(pyo.value(model.a)))
print('***** b = {} *****'.format(pyo.value(model.b)))
print('***** c = {} *****'.format(pyo.value(model.c)))
