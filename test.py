from utility.tsg_util import *
from utility.tsg_plot import *
from utility.tsg_models import *


# Parameter
Param_tuning = False
num_variables=12
num_outputs=1
window_size = 60
batch_size = 512
epochs = 1

# Import Data
mat_file_name= dataMainPath + "sampe_data4learning.mat"
mat_file=scipy.io.loadmat(mat_file_name)
mat_file_name2= dataMainPath + "sampe_data4learning2.mat"
mat_file2=scipy.io.loadmat(mat_file_name2)
concatenated_data = np.concatenate((mat_file['Traj_samples'], mat_file2['Traj_samples']), axis=2)

# Input Parameters - (2nd ~ 13th elements)
time_Deputy_Chief=np.real(concatenated_data[:,1:13])

# Output Parameters - Time shift parameter (24th element)
tbackward=np.real(concatenated_data[:,23])

# Split data into training/validation and test parts
# Train & Val: 20 differenct initial conditions
# Test: 4 differenct initial conditions
X_train_val = time_Deputy_Chief[:, :, :40]  
Y_train_val = tbackward[:, :40]  
X_test = time_Deputy_Chief[:, :, 40:] 
Y_test = tbackward[:, 40:]        

# Calculate normalization scaler
output_scaler= GenerateNormalization(Y_train_val, Y_test)
Y_train_val_norm, Y_test_norm = NormalizeAllOutputDataWithScaler(output_scaler, Y_train_val, Y_test)

# Sliding window of training/validation & test parts
X_sequenced, Y_sequenced = make_sliding_window(X_train_val, Y_train_val_norm, window_size)
X_test_sequenced, Y_test_sequenced = make_sliding_window(X_test, Y_test_norm, window_size)


# HyperParameter Tuning with KerasTuner    
if Param_tuning:
    # Find optimized hyperparameter
    tuner = kt.Hyperband(
        MyHyperModel(window_size=window_size, X_train=X_sequenced),
        objective='val_loss',
        max_epochs=30,
        executions_per_trial=3,
        overwrite=True,
        factor=3,
    )
    tuner.search(X_sequenced, Y_sequenced, epochs=50, validation_split=0.2)
    # extract optimized parameters
    best_hps = tuner.get_best_hyperparameters()[0]
    # print optimized parameters
    print(
        f"""
    units_1 : {best_hps.get('units_1')}
    units_2 : {best_hps.get('units_2')}
    units_3 : {best_hps.get('units_3')}
    units_4 : {best_hps.get('units_4')}
    dropout_1 : {best_hps.get('dropout_1')}
    regularizer : {best_hps.get('regularizer')}
    learning_rate : {best_hps.get('learning_rate')}
    batch_size : {best_hps.get('batch_size')}
    """
    )
    # print(
    #     f"""
    # units_1 : {best_hps.get('units_1')}
    # units_4 : {best_hps.get('dropout_1')}
    # regularizer : {best_hps.get('regularizer')}
    # learning_rate : {best_hps.get('learning_rate')}
    # batch_size : {best_hps.get('batch_size')}
    # """
    # )
    # extract batch size for 'fit' method
    batch_size = best_hps.get("batch_size")
    # build optimized model
    model = tuner.hypermodel.build(best_hps)


# Train model
model = stacked_lstm_model(window_size, num_variables, X_sequenced, num_outputs)
model.summary()
early_stopping_callback = EarlyStopping(monitor="loss", min_delta=0, patience=15, verbose=0)
history = model.fit(
    X_sequenced,
    Y_sequenced,
    validation_split=0.25,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    callbacks=[early_stopping_callback],
)

# De-normalize Output
pred_norm = model.predict(X_test_sequenced)
pred = pred_norm    
pred, Y_test_sequenced = DeNormalizeOutputData(output_scaler, pred_norm, Y_test_sequenced)

# Save Data
savetime = time()
modelpath = f"{modelMainPath}{savetime}"
createDirectory(modelpath)
model.save(modelpath)
model_log_save(model, batch_size, window_size, epochs, pred, Y_test_sequenced, modelpath)







# Plot model history
# acc = [0.0] + history.history["accuracy"]
# loss = history.history["loss"]
# val_acc = [0.0] + history.history["val_accuracy"]
# val_loss = history.history["val_loss"]
# plot_model_history(acc, val_acc, loss, val_loss)

