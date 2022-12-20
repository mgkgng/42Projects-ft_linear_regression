using DataFrames, LinearAlgebra, Statistics, Plots

struct LRModel
    thetas::Array{Float64,2}
    alpha::Float64
    max_iter::Int
end

function LRModel(;alpha=0.001, max_iter=100000)
    thetas = [0.0 0.0]
    return LRModel(thetas, alpha, max_iter)
end

function estimatePrice(model::LRModel, mileage)
    return ceil(round((model.thetas[2] * mileage + model.thetas[1]).item(1)) / 5) * 5
end

function trainModel(model::LRModel, x, y)
    for _ in 1:model.max_iter
        model.thetas -= loss(model, x, y)
    end
    plot(x, y, "o")
    plot!(x, [ones(length(x)) x] * model.thetas)
    title!("Please quit this plot in order to estimate the price 😉")
    xlabel!("Standard deviation for Mileage(km)")
    ylabel!("Price")
    display(plot)
end

function calculatePrecision(model::LRModel, x, y)
    vfunc(x) = estimatePrice(model, x)
    sumX, sumY = sum(vfunc.(x)), sum(y)
    return round((1 - abs((sumY - sumX) / sumY)) * 100)
end

function loss(model::LRModel, x, y)
    x_prime = [ones(length(x)) x]
    cost = x_prime * model.thetas - y
    return (x_prime' * cost) / (size(x_prime, 1) * 2)
end

function unison_shuffled_copies(a, b)
    p = shuffle(1:length(a))
    return a[p], b[p]
end

function data_spliter(x, y, proportion)
    shuffled = unison_shuffled_copies(x, y)
    testNb = Int(size(x, 1) * proportion)
    splitX = split(shuffled[1], [testNb, size(x, 1)])
    splitY = split(shuffled[2], [testNb, size(x, 1)])
    return splitX[1], splitX[2], splitY[1], splitY[2]
end

function zscore(x)
    vfunc(e) = (e - mean(x)) / std(x)
    return vfunc.(x)
end

function main()
    data = convert(Array{Float64}, DataFrame(CSV.File("../../assets/data.csv")))
    X, Y = data[:,1], data[:,2]
    trainX, testX, trainY, testY = data_spliter(X, Y, 0.8)
    mean_X, std_X = mean(trainX), std(trainX)

    lr = LRModel()
    trainModel(lr, zscore(trainX), trainY)
	zfunc(x) = (x - mean_X) / std_X
    precision = calculatePrecision(lr, zfunc.(testX), testY)

    while true
        s = readline("Enter the mileage to get the estimate the price of your car, or insert QUIT\n>>> ")
        if s == "QUIT"
            exit("Bye bye~")
        elseif !isdigit(s)
            println("Wrong input")
        else
            estimation = estimatePrice(lr, (parse(Int, s) - mean_X) / std_X)
            println()
            println(@sprintf("The estimated price is $%d", estimation))
        end
    end
end

main()


