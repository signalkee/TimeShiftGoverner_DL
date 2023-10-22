from tsg_util import *

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_absolute_percentage_error as mape


def model_log_save(batch_size, window_size, epochs, pred, Y_test_sequenced, modelpath):
    sys.stdout = open(f"{modelpath}/log.txt", 'w')

    print(f"Batch size: {batch_size}  |  window size: {window_size}  |  epoch: {epochs}")
    print("Saving prediction...")
    preddf, Y_testdf = pd.DataFrame(pred), pd.DataFrame(Y_test_sequenced)
    preddf.to_csv(f"{modelpath}/pred.csv")
    Y_testdf.to_csv(f"{modelpath}/true.csv")
    print("Prediction save complete!")

    # # Check Model Prediction
    plot_predictions(pred, Y_test_sequenced, modelpath)
    # # Calculate MAE, MSE, RMSE Value of 3D trajectory
    calculate_performance(pred, Y_test_sequenced)

    print("finished")
    sys.stdout.close()


def calculate_performance(pred, Y_test):
    mae_sub = mae(Y_test, pred)
    mse_sub = mse(Y_test, pred)
    rmse_sub = np.sqrt(mse_sub)
    r2_sub = r2(Y_test, pred)
    print("MAE: ", mae_sub, "  MSE: ", mse_sub,
            "   RMSE: ", rmse_sub, "  R2: ", r2_sub)


def plot_model_history(acc, val_acc, loss, val_loss):
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
    
    
def plot_predictions(pred, Y_test_sequenced, modelpath):
    plt.figure(figsize=(12, 6))
    time_steps = range(len(Y_test_sequenced))
    plt.plot(time_steps, pred, label='Predicted', color='blue')
    plt.plot(time_steps, Y_test_sequenced, label='True', color='green')
    plt.title('Predicted vs True Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f"{modelpath}/trajectory2D.png")
    plt.show()

    
'''  

def plot_predictions1(model, X, y, start=0, end=100):
    predictions = model.predict(X).flatten()
    df = pd.DataFrame(data={'Predictions':predictions, 'Actuals':y})
    
    plt.figure()
    plt.plot(df['Predictions'][start:end], label='Predictions')
    plt.plot(df['Actuals'][start:end], label='Actuals')
    plt.legend(fontsize=15)
    plt.xlabel('Prediction Time + 58 min (sec)',fontsize=15)
    plt.ylabel('Time shift (min)',fontsize=15)

    return df, mse(y, predictions)


def plot_predictions2(model, X, y, X_train0, start=0, end=100):
    predictions = model.predict(X)
    p_preds, temp_preds =predictions[:, 0], predictions[:, 1]
    p_actuals, temp_actuals = y[:, 0], y[:, 1]
    df = pd.DataFrame(data={'Temperature Predictions':temp_preds, 
                            'Temperature Actuals':temp_actuals,
                            'Pressure Predictions':p_preds, 
                            'Pressure Actuals':p_actuals
                            })
    plt.figure()
    plt.plot(df['Temperature Predictions'][start:end], label='Temperature Predictions')
    plt.plot(df['Temperature Actuals'][start:end], label='Temperature Actuals')  
    plt.plot(df['Pressure Predictions'][start:end], label='Pressure Predictions')
    plt.plot(df['Pressure Actuals'][start:end], label='Pressure Actuals')  
    plt.legend(fontsize=15)
    return df[start:end], mse(y, predictions)


def plot_post_predictions2(model, X, y, X_train0, start=0, end=100):
    predictions = model.predict(X)
    p_preds, temp_preds = postprocess_p(predictions[:, 0], X_train0), postprocess_temp(predictions[:, 1], X_train0)
    p_actuals, temp_actuals = postprocess_p(y[:, 0], X_train0), postprocess_temp(y[:, 1], X_train0)
    df = pd.DataFrame(data={'Temperature Predictions':temp_preds, 
                            'Temperature Actuals':temp_actuals,
                            'Pressure Predictions':p_preds, 
                            'Pressure Actuals':p_actuals
                            })
    plt.figure()
    plt.plot(df['Temperature Predictions'][start:end], label='Temperature Predictions')
    plt.plot(df['Temperature Actuals'][start:end], label='Temperature Actuals')  
    plt.plot(df['Pressure Predictions'][start:end], label='Pressure Predictions')
    plt.plot(df['Pressure Actuals'][start:end], label='Pressure Actuals')  
    plt.legend(fontsize=15)
    return df[start:end], mse(y, predictions)

'''
