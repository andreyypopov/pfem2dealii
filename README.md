
# **pfem2dealii** library

Copyright (C) 2022-2023 Andrey Popov

Based on the [deal.ii C++ library](https://dealii.org/), implementation of the Particle Finite Element Method, 2nd generation (PFEM-2). Main features:

- Lagrangian particle handling (transport of different fields, including velocity);
- solution of the reduced hydrodynamic problem (without the convective term) on the Eulerian mesh using FEM;
- field projection procedures from particles onto mesh and vice versa;
- data output using VTK;
- template implementation for 2D and 3D problems;
- parallelization using MPI.