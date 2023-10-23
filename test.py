from utility.tsg_dataloader import *
from utility.tsg_preprocessor import *
from utility.tsg_logger import *
from utility.tsg_models import *
from utility.values import *


# Parameter
Param_tuning = False
num_variables=12
num_outputs=1
window_size = 60
batch_size = 512
epochs = 1
file_names = ["sample_data4learning.mat", "sample_data4learning2.mat"]


# Load and split data from multiple .mat files
dataLoader = DataLoader(dataMainPath=dataMainPath)
dataLoader.load_mat_files(file_names)
X_train_val, Y_train_val, X_test, Y_test = dataLoader.split_data(train_val_samples=20, test_samples=4)    

# Preprocess data
preprocessor = TimeSeriesPreprocessor(window_size)
output_scaler = preprocessor.generate_normalization(Y_train_val, Y_test)
Y_train_val_norm, Y_test_norm = preprocessor.normalize_all_output_data_with_scaler(Y_train_val, Y_test)
X_sequenced, Y_sequenced = preprocessor.make_sliding_window(X_train_val, Y_train_val_norm)
X_test_sequenced, Y_test_sequenced = preprocessor.make_sliding_window(X_test, Y_test_norm)


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
pred, Y_test_sequenced = preprocessor.denormalize_output_data(output_scaler, pred_norm, Y_test_sequenced)

# Save Data
logger = ModelLogger(modelMainPath=modelMainPath)
model_path = logger.save_model(model)
logger.model_log_save(batch_size, window_size, epochs, pred, Y_test_sequenced)



