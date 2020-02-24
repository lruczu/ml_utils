from pyomo.environ import ConcreteModel, Constraint, Objective, Var, NonNegativeReals

"""
min x1 + 2 * x2
s.t.
3 * x1 + 4 * x2 >= 1
2 * x1 + 5 * x2 >= 2
x1, x2 >= 0
"""

model = ConcreteModel()
model.x_1 = Var(within=NonNegativeReals)
model.x_2 = Var(within=NonNegativeReals)
model.obj = Objective(expr=model.x_1 + 2 * model.x_2)
model.con1 = Constraint(expr=3 * model.x_1 + 4 * model.x_2 >= 1) 
model.con2 = Constraint(expr=2 * model.x_1 + 5 * model.x_2 >= 2)

# pyomo solve --solver=glpk basic_example1.py
