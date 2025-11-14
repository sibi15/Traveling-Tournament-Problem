# Travelling Tournament Problem: Exact vs Heuristic Optimization

An optimization project comparing Constraint Programming and Genetic Algorithms for sports tournament scheduling. Developed as part of the "Optimization for Data Science" course at TU/e, this project minimizes team travel distances while satisfying complex tournament constraints across multiple benchmark instances.

### Tools & Technologies:
- Python
- Google OR-Tools (CP-SAT)
- Genetic Algorithms
- NumPy
- Pandas
- Matplotlib
- Git

### Key Features:
- Implemented CP-SAT solver for exact optimization and custom Genetic Algorithm for heuristic search
- Enforced tournament constraints: home/away patterns, no-repeat rules, and streak limitations
- Benchmarked on standard datasets (NL4, NL6, NL8, NL10) with up to 10 teams
- Generated visual schedules and team movement patterns for validation
- Achieved optimal solutions for small instances; GA solved 10-team problems in <3 seconds

### Outcome:
Demonstrated trade-offs between solution methods, with CP-SAT having provided proven optimality for small instances while Genetic Algorithm delivered practical scalability for larger problems, solving them 1000x faster with competitive solution quality.
