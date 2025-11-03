using Flux
using Flux: gradient
using Statistics
using Random
using Plots
Base.CoreLogging.disable_logging(Base.CoreLogging.Warn)

# 1️⃣ Define target function
f(x) = x.^3

# 2️⃣ Generate training data with NORMALIZATION
Random.seed!(42)  # For reproducibility
xs = Float32.(sort(-3 .+ 6 .* rand(100)))
ys = f(xs)

# Normalize inputs and outputs
x_mean, x_std = mean(xs), std(xs)
y_mean, y_std = mean(ys), std(ys)

xs_norm = (xs .- x_mean) ./ x_std
ys_norm = (ys .- y_mean) ./ y_std

x_train = reshape(xs_norm, 1, :)
y_train = reshape(ys_norm, 1, :)

# 3️⃣ Define model
model = Chain(
    Dense(1, 32, relu),
    Dense(32, 1)
)

# 4️⃣ Define optimizer and setup state
opt = Flux.Optimisers.Adam(0.001)
state = Flux.setup(opt, model)

# 5️⃣ Define loss function
loss(model, x, y) = Flux.Losses.mse(model(x), y)

# 6️⃣ Training loop with loss tracking
println("Training...")
train_losses = Float64[]
for epoch in 1:10000
    grads = gradient(model -> loss(model, x_train, y_train), model)
    Flux.update!(state, model, grads[1])

    if epoch % 100 == 0
        current_loss = loss(model, x_train, y_train)
        push!(train_losses, current_loss)
        if epoch % 1000 == 0
            println("Epoch $epoch → Loss: ", round(current_loss, digits=6))
        end
    end
end

# 7️⃣ Predictions (normalize input, then denormalize output)
x_test = Float32.(-3:0.01:3)
x_test_norm = (x_test .- x_mean) ./ x_std
y_pred_norm = model(reshape(x_test_norm, 1, :)) |> vec
y_pred = y_pred_norm .* y_std .+ y_mean  # Denormalize predictions

y_true = f(x_test)

# Calculate errors on test set
mae = mean(abs.(y_pred .- y_true))
max_err = maximum(abs.(y_pred .- y_true))
mse = mean((y_pred .- y_true).^2)

# 8️⃣ Display normalization parameters
println("\n" * "="^70)
println("Normalization Parameters:")
println("="^70)
println("  x_mean = $(x_mean)f0")
println("  x_std  = $(x_std)f0")
println("  y_mean = $(y_mean)f0")
println("  y_std  = $(y_std)f0")
println("="^70)

println("\nApproximation Quality:")
println("  MSE:       $(round(mse, digits=6))")
println("  MAE:       $(round(mae, digits=4))")
println("  Max Error: $(round(max_err, digits=4))")
println("="^70 * "\n")

# 9️⃣ Plot results
# Training curve
p1 = plot(100:100:length(train_losses)*100, train_losses,
          label="Training Loss", lw=2, xlabel="Epoch", ylabel="MSE Loss (normalized)",
          title="Training Curve", legend=:topright)

# Function approximation
p2 = plot(xs, ys, seriestype=:scatter, label="Training Data", 
          legend=:topleft, markersize=4, alpha=0.6)
plot!(p2, x_test, y_true, label="True f(x) = x³", lw=3, color=:blue)
plot!(p2, x_test, y_pred, label="NN Approximation", lw=2, ls=:dash, color=:red)
xlabel!(p2, "x")
ylabel!(p2, "f(x)")
title!(p2, "Neural Network Approximation of x³\nMSE = $(round(mse, digits=6))")

final_plot = plot(p1, p2, layout=(2,1), size=(800, 800))
display(final_plot)
