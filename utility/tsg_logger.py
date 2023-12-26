import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_absolute_percentage_error as mape


class ModelLogger:
    def __init__(self, modelMainPath):
        self.modelMainPath = modelMainPath
        self.modelSavedPath = None
        self.model = None
        self.savetime = None

    def create_directory(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Failed to create the directory.")

    def save_model(self, model):
        self.savetime = time.time()
        modelpath = f"{self.modelMainPath}{self.savetime}"
        self.create_directory(modelpath)
        torch.save(model.state_dict(), os.path.join(modelpath, 'model.pth'))
        self.modelSavedPath = modelpath
        return modelpath

    def model_log_save(self, batch_size, window_size, epochs, pred, Y_test_sequenced):
        log_file_path = os.path.join(self.modelSavedPath, 'log.txt')
        with open(log_file_path, 'w') as log_file:
            print(f"Batch size: {batch_size}  |  window size: {window_size}  |  epoch: {epochs}", file=log_file)
            print("Saving prediction...", file=log_file)
            preddf, Y_testdf = pd.DataFrame(pred), pd.DataFrame(Y_test_sequenced)
            preddf.to_csv(os.path.join(self.modelSavedPath, 'pred.csv'))
            Y_testdf.to_csv(os.path.join(self.modelSavedPath, 'true.csv'))
            print("Prediction save complete!", file=log_file)

            self.plot_predictions(pred, Y_test_sequenced)
            self.calculate_performance(pred, Y_test_sequenced)

            print("Finished", file=log_file)

    def calculate_performance(self, pred, Y_test):
        mae_sub = np.mean(np.abs(Y_test - pred))
        mse_sub = np.mean((Y_test - pred) ** 2)
        rmse_sub = np.sqrt(mse_sub)
        r2_sub = 1 - (mse_sub / np.mean((Y_test - np.mean(Y_test)) ** 2))
        print("MAE: ", mae_sub, "  MSE: ", mse_sub, "   RMSE: ", rmse_sub, "  R2: ", r2_sub)

    def plot_model_history(self, acc, val_acc, loss, val_loss):
        plt.figure("model history", figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label="Training Accuracy")
        plt.plot(val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.ylabel("Accuracy")
        plt.ylim([min(plt.ylim()), 1])
        plt.title("Training and Validation Accuracy")
        plt.subplot(2, 1, 2)
        plt.plot(loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.ylabel("Cross Entropy")
        plt.ylim([0, 1.0])
        plt.title("Training and Validation Loss")
        plt.xlabel("epoch")
        plt.show()

    def plot_predictions(self, pred, Y_test_sequenced):
        plt.figure(figsize=(12, 6))
        time_steps = range(len(Y_test_sequenced))
        plt.plot(time_steps, pred, label='Predicted', color='blue')
        plt.plot(time_steps, Y_test_sequenced, label='True', color='green')
        plt.title('Predicted vs True Values')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f"{self.modelSavedPath}/trajectory2D.png")
        plt.show()
        

if __name__ == '__main__':  
    logger = ModelLogger(modelMainPath="/your/path")