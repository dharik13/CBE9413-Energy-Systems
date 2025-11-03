# HW6 - Julia Project Environment

This directory contains a Julia project with all necessary dependencies for running the optimization and machine learning problems in Q2(a), Q2(b), and Q2(c).

## Dependencies

The project uses the following Julia packages:
- **Flux** - Deep learning framework for neural network training (Q2a)
- **JuMP** - Mathematical optimization modeling (Q2b, Q2c)
- **Gurobi** - Commercial optimization solver
- **CSV & DataFrames** - Data handling (Q2b, Q2c)
- **Plots** - Visualization
- **PrettyTables** - Nice table formatting
- **Statistics & Random** - Standard library utilities

## Setup Instructions

### 1. Install Julia
Make sure you have Julia 1.9 or later installed. Download from [julialang.org](https://julialang.org/downloads/).

### 2. Install Gurobi
You need a Gurobi license (free academic license available at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/)).

### 3. Activate the Project Environment

Open Julia in this directory and run:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

This will automatically install all required packages as specified in `Project.toml` and `Manifest.toml`.

### 4. Run the Scripts

Once the environment is set up, you can run any of the scripts:

```julia
# Make sure you're in the project environment
using Pkg
Pkg.activate(".")


## Files

- **Project.toml** - Lists the project dependencies and their compatible versions
- **Manifest.toml** - Complete dependency tree with exact versions (auto-generated)

## Reproducibility

The `Manifest.toml` file ensures that anyone using this project will get the exact same package versions, guaranteeing reproducible results.

## Troubleshooting

If you encounter issues:

1. **Gurobi not found**: Make sure you have a valid Gurobi license and it's properly configured
   ```julia
   using Gurobi
   ENV["GUROBI_HOME"] = "/path/to/gurobi"
   Pkg.build("Gurobi")
   ```

2. **Package conflicts**: Remove the Manifest.toml and regenerate it
   ```julia
   rm("Manifest.toml")
   Pkg.activate(".")
   Pkg.instantiate()
   ```

3. **Precompilation errors**: Clear the compiled cache
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.resolve()
   Pkg.precompile()
   ```

## Package Versions

This project was last tested with:
- Julia 1.11.0
- Flux 0.14.25
- JuMP 1.29.2
- Gurobi 1.7.6
- All other packages as specified in Manifest.toml
