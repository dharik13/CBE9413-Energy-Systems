using Flux
using Plots
Base.CoreLogging.disable_logging(Base.CoreLogging.Warn)
# 1️⃣ Define target function
f(x) = x.^3

# 2️⃣ Generate training data
xs = Float32.(sort(-3 .+ 6 .* rand(100)))
ys = f(xs)


x_train = reshape(xs, 1, :)
y_train = reshape(ys, 1, :)

# 3️⃣ Define model
model = Chain(
    Dense(1, 32, relu),
    Dense(32, 1)
)

# 4️⃣ Define optimizer and setup state
opt = Descent(0.01)
state = Flux.setup(opt, model)

# 5️⃣ Define loss function
loss(model, x, y) = Flux.Losses.mse(model(x), y)

# 6️⃣ Training loop
println("Training...")
for epoch in 1:100000
    grads = gradient(model -> loss(model, x_train, y_train), model)
    Flux.update!(state, model, grads)

    if epoch % 200 == 0
        println("Epoch $epoch → Loss: ", loss(model, x_train, y_train))
    end
end

# 7️⃣ Predictions
x_test = Float32.(-2:0.01:2)
y_pred = model(reshape(x_test, 1, :)) |> vec

# 8️⃣ Plot results
#plot(xs, ys, label="True f(x) = x³", lw=3)
plot(xs, ys, seriestype = :scatter, label = "Random samples", legend = :top)
plot!(x_test, y_pred, label="NN Approximation", lw=3, ls=:dash)
xlabel!("x")
ylabel!("f(x)")
title!("Neural Network Approximation of x³")
