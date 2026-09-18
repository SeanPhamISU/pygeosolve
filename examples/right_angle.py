"""Make a right angle from two lines."""

from pygeosolve import Problem

# Create problem with initial line positions.
problem = Problem()
problem.add_line("a", (0, 0), (30, 0))
problem.add_line("b", problem["a"].start, (15, 15))

# Constrain the lines.
problem.constrain_position("a")
problem.constrain_line_length("b", 30)
problem.constrain_angle_between_lines("a", "b", 90)

# Solve
problem.solve()

# Print the current solution state.
print(problem)

# Plot the solved sketch.
problem.plot()
