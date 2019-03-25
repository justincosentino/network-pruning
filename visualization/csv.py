"""
Helper methods for converting experiment output to csv.
"""
import os
import pandas as pd


def write_to_csv(
        weight_losses,
        weight_accuracies,
        unit_losses,
        unit_accuracies,
        k_vals,
        experiment_dir,
        output_name='results.csv'):
    """
    Writes pruned losses and accuracies to a csv at
    '<experiment_dir>/<output_name>'.
    """
    pd.options.display.float_format = '{:2.4f}'.format
    frame = pd.DataFrame(data={
        "Sparsity (%)": k_vals,
        "Test Accuracy (Unit Pruning)": weight_accuracies,
        "Test Accuracy (Weight Pruning)": unit_accuracies,
        "Test Loss: (Weight Pruning)": weight_losses,
        "Test Loss: (Unit Pruning)": unit_losses,
    })
    frame.to_csv(
        os.path.join(experiment_dir, output_name),
        index=False,
        float_format="%.4f")
