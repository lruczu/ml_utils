from typing import Tuple

import numpy as np
import pyomo.environ as pyo 
from pyomo.opt import SolverFactory



def from_flat(index, n_rows: int, n_cols: int) -> Tuple[int, int]:
	"""
		Args:
			index: flat index starting from 0 
			n_rows: number of rows of considered matrix
			n_cols: number of columns of considered matrix
		Returns:
			(row index, col index), both starting from 0
	"""
	row_index = index // n_cols
	return row_index, index - row_index * n_cols


def to_flat(row_index: int, col_index: int, n_rows: int, n_cols: int) -> int:
	"""
		Args:
			row_index: row index starting from 0
			col_index: col index starting from 0
			n_rows: number of rows of considered matrix
			n_cols: number of columns of considered matrix
		Returns:
			flat index starting from 0
	"""
	return row_index * n_cols + col_index


def add_transportation_problem(
	model,
	costs: np.ndarray,
	supply: np.ndarray,
	demand: np.ndarray,
):
	"""
		Args:
			model: pymo ConcreteModel to be mutated by adding variables, constraints and objective
			costs: matrix of costs of transporting an item from a supplier to a customer, 
				of shape (# suppliers, # customers)
			supply: number of items suppliers can provide
			demand:  number of iterms requested by clents
	"""
	n_suppliers = len(supply)
	n_customers = len(demand)

	model.nVars = pyo.Param(initialize=n_suppliers * n_customers)
	model.N = pyo.RangeSet(model.nVars)
	model.x = pyo.Var(model.N, within=pyo.NonNegativeReals)
	model.supply_constraints = pyo.ConstraintList()
	model.demand_constraints = pyo.ConstraintList()

	for i, s in enumerate(supply):
		expr = 0
		for j in range(n_customers):
			variable_index = to_flat(i, j, n_rows=n_suppliers, n_cols=n_customers) + 1
			expr += model.x[variable_index]

		model.supply_constraints.add(expr <= s)
	
	for i, d in enumerate(demand):
		expr = 0
		for j in range(n_suppliers):
			variable_index = to_flat(j, i, n_rows=n_suppliers, n_cols=n_customers) + 1
			expr += model.x[variable_index]
		
		model.demand_constraints.add(expr == d)

	expr = 0
	for i in range(n_suppliers):
		for j in range(n_customers):
			var_index = to_flat(i, j, n_rows=n_suppliers, n_cols=n_customers) + 1
			expr += model.x[var_index] * costs[i, j]

	model.obj = pyo.Objective(expr=expr)

def get_solution(model, n_rows, n_cols):
	sol = np.zeros((n_rows, n_cols))
	for i in range(n_rows):
		for j in range(n_cols):
			var_index = to_flat(i, j, n_rows=n_rows, n_cols=n_cols) + 1
			sol[i, j] = pyo.value(model.x[var_index])
	return sol

"""
n = 1000
C = np.ones(shape=(n, n))
S = np.ones(n)
D = np.ones(n)
~ 144s
"""

C = np.array([
	[10, 8, 5, 7],
	[11, 12, 12, 15],
	[9, 8, 16, 12]
])
S = np.array([500, 900, 600])
D = np.array([350, 500, 450 ,700])

opt = SolverFactory('glpk')

model = pyo.ConcreteModel()
add_transportation_problem(model, C, S, D)
results = opt.solve(model)
model.display()
solution = get_solution(model, len(S), len(D))
print(solution)
