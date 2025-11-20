import math
import statistics
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import norm

import utils
from final.models import Model

# initialize a model from a name
# TODO: make this easier to set - it shouldn't need to be so manual!
# model_name = "1763331094"
# model_folder = f"{model_name}_large_100"
# model_name = "1763331560"
# model_folder = f"{model_name}_large_1000"
# model_name = "1763475397"
# model_folder = f"{model_name}_1000_6"
# model_name = "1763475738"
# model_folder = f"{model_name}_1000_6"
# model_name = "1763475888"
# model_folder = f"{model_name}_500_6"
model_name = "1763476236"
model_folder = f"{model_name}_1000_6"


if __name__ == "__main__":

    # load the test dataset
    test_data_path = Path("data", "test_full.csv")
    test_inputs, test_labels = utils.load_model_data(test_data_path)

    # load the normalization conversion key
    norm_key_path = Path("data", "full_normalization_key.csv")
    norm_key = utils.load_data(norm_key_path)
    norm_max = norm_key["snod_delta"][0]
    norm_min = norm_key["snod_delta"][1]

    model = Model()
    weights_path = Path("trainings", model_folder, f"{model_name}_weights.pth")
    state_dict = torch.load(weights_path, weights_only=True)
    model.load_state_dict(state_dict)

    test_outputs = []
    test_deltas = []
    real_expected = []
    real_predictions = []
    real_deltas = []
    all_results = []
    print(f"Starting testing of: {model_name}")
    # no need to calculate the gradients in testing mode
    # and no batches in test mode
    with torch.no_grad():
        for i, (inp, label) in enumerate(zip(test_inputs, test_labels)):
            # forward
            output = model(inp)

            delta = output - label
            test_outputs.append(float(output))
            test_deltas.append(delta)

            # denormalize
            real_exp = utils.denormalize_min_max(label, norm_min, norm_max)
            real_out = utils.denormalize_min_max(output, norm_min, norm_max)
            real_delta = float(real_out - real_exp)
            real_deltas.append(real_delta)

            real_expected.append(real_exp)
            real_predictions.append(real_out)
            print(f"Expected: {round(float(label), 2)}, "
                  f"actual: {round(float(output), 2)},  "
                  f"delta: {round(float(delta), 2)}, "
                  f"Real expected: {round(float(real_exp), 2)}, "
                  f"Real actual: {round(float(real_out), 2)}, "
                  f"Real delta: {round(float(real_delta), 2)}")
            all_results.append( (label, output, delta, real_exp, real_out, real_delta) )

    # calculate the rmse
    mse = mean_squared_error(test_labels, test_outputs)
    rmse = math.sqrt(mse)
    print(f"RMSE: {rmse}")

    # plot the test results
    # let's try making a plot of the deltas
    sorted_results = sorted(all_results, key=lambda x: abs(x[2]))
    sorted_real_expected = [i[3] for i in sorted_results]
    sorted_real_actual = [i[4] for i in sorted_results]
    indexes = list(range(len(sorted_results)))

    plt.clf()
    plt.scatter(indexes, sorted_real_expected, label="Expected Delta")
    plt.scatter(indexes, sorted_real_actual, label="Predicted Delta")
    plt.ylabel("Snow Depth Delta (cm)")
    plt.legend()
    plt.title("Predicted vs Expected")
    save_path = Path("trainings", model_folder, f"{model_name}_real_exp_comparison.png")
    plt.savefig(save_path)

    # and how about a histogram
    plt.clf()
    plt.hist(real_deltas, bins=int(len(real_deltas)/4), color='royalblue', density=True)

    # let's also calculate some distribution
    mean_depth_delta = statistics.mean(real_deltas)
    std_dev_depth_delta = statistics.stdev(real_deltas)
    print(f"Mean delta between predicted and actual: {mean_depth_delta}, stdev: {std_dev_depth_delta}")

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_depth_delta, std_dev_depth_delta)
    plt.plot(x, p, label=f'Estimated Dist (μ={mean_depth_delta:.2f}, σ={std_dev_depth_delta:.2f})', color="indigo")

    plt.xlabel("Difference Between Actual and Predicted SNOD change (cm)")
    plt.ylabel("Count")
    plt.title("Distribution of Prediction Error for Snow Depth Change on Test Dataset")
    plt.legend()

    save_path = Path("trainings", model_folder, f"{model_name}_histogram.png")
    plt.savefig(save_path)

    # let's also show a scatter plot of actual vs predicted
    plt.clf()
    plt.scatter(real_expected, real_predictions, label="Test Data")
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot([xmin, xmax], [ymin, ymax], label="Ideal", color="indigo")
    plt.ylabel("Predicted Snow Depth Change (cm)")
    plt.xlabel("Actual Now Depth Change (cm)")
    plt.title("Predicted Snow Depth Change vs Actual on Test Dataset")
    plt.axis('equal')
    plt.legend()
    save_path = Path("trainings", model_folder, f"{model_name}_scatter.png")
    plt.savefig(save_path)
