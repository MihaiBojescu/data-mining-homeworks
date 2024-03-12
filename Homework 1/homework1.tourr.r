library("tourr")

main <- function() {
    data <- read()
    filtered_data <- filter(data)
    normalised_data <- normalise(filtered_data)
    render(normalised_data, data)
}

read <- function() {
    data <- read.csv("./data/ObesityDataSet.csv", header = TRUE)
    data
}

filter <- function(data) {
    numerical_variables = c(
        "Weight",
        "FAF",
        "NCP"
    )
    filtered_data <- data[, numerical_variables]
    filtered_data
}

normalise <- function(data) {
    normalize_z_score <- function(x) {
        return ((x - mean(x)) / sd(x))
    }

    df <- data.frame(data)
    df_z_score <- as.data.frame(lapply(df, normalize_z_score))
    df_z_score
}

render <- function(normalised_data, data) {
    out_index <- function(data, weights) {
        center <- colMeans(data)
        outlyingness <- sqrt(rowSums((data - center)^2))
        mean(outlyingness)
    }

    render_gif(
        normalised_data,
        tour_path = guided_tour(out_index),
        display = display_xy(
            cex = 2,
            col = data$NObeyesdad
        ),
        gif_file = "./resources/animation.gif",
        frames = 60 * 5
    )
}

main()