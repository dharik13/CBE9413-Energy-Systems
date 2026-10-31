# HW6 - Julia Project Environment

This directory contains the data files, example code and Julia project environment for HW6:
- **Question 1** - Neural network surrogates and a triangle-based piecewise linear approximation of a heat pump with a hot water tank (parts A-D)
- **Question 2** - Piece-wise McCormick envelopes for a district heating system (parts A-C)

## Files

- **hp_tank_data.csv** - Hourly electricity price ($/MWh) and heat demand (MW) for one day, 24 hours (Q1)
- **ML_example.jl** - Example of training a neural network with Flux. Follow its structure for Q1 A
- **heat_demand_MW.csv** - Hourly heat demand (MW) for one week, 168 hours (Q2 C)
- **Project.toml** - Lists the project dependencies and their compatible versions
- **Manifest.toml** - Complete dependency tree with exact versions (auto-generated)

## Dependencies

The project uses the following Julia packages:
- **Flux** - Deep learning framework for training the neural network surrogates (Q1 A)
- **JuMP** - Mathematical optimization modeling (Q1 B-C, Q2 C)
- **Gurobi** - Commercial optimization solver (Q1 B-C, Q2 C)
- **CSV & DataFrames** - Reading the data files (Q1, Q2 C)
- **Plots** - Visualization
- **PrettyTables** - Nice table formatting
- **Statistics & Random** - Standard library utilities

## Setup Instructions

### 1. Install Julia
Install Julia 1.11 from [julialang.org](https://julialang.org/downloads/). The `Manifest.toml` was generated with Julia 1.11.6.

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

Once the environment is set up, activate it at the start of each session and run the example or your own script:

```julia
# Make sure you're in the project environment
using Pkg
Pkg.activate(".")

include("ML_example.jl")
```

## Reproducibility

The `Manifest.toml` file ensures that anyone using this project will get the exact same package versions, guaranteeing reproducible results.

## Troubleshooting

If you encounter issues:

1. **Gurobi not found**: Make sure you have a valid Gurobi license and it's properly configured
   ```julia
   ENV["GUROBI_HOME"] = "/path/to/gurobi"
   using Pkg
   Pkg.build("Gurobi")
   using Gurobi
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

The `Manifest.toml` pins:
- Julia 1.11.6
- Flux 0.14.25
- JuMP 1.29.2
- Gurobi 1.7.6
- All other packages as specified in Manifest.toml
