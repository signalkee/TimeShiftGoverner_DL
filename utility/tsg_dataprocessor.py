from utility.tsg_dataloader import *
from utility.tsg_preprocessor import *
from utility.tsg_logger import *
from utility.tsg_models import *
from utility.values import *


class DataProcessor:
    def __init__(self, num_variables, num_outputs, window_size):
        self.num_variables = num_variables
        self.num_outputs = num_outputs
        self.window_size = window_size
        self.preprocessor = TimeSeriesPreprocessor(self.window_size)
        self.logger = ModelLogger(modelMainPath=modelMainPath)

    def load_and_preprocess_data(self, dataMainPath, file_names, train_val_samples, test_samples):
        dataLoader = DataLoader(dataMainPath=dataMainPath)
        dataLoader.load_mat_files(file_names)
        X_train_val, Y_train_val, X_test, Y_test = dataLoader.split_data(train_val_samples, test_samples)

        output_scaler = self.preprocessor.generate_normalization(Y_train_val, Y_test)
        Y_train_val_norm, Y_test_norm = self.preprocessor.normalize_all_output_data_with_scaler(Y_train_val, Y_test)

        X_sequenced, Y_sequenced = self.preprocessor.make_sliding_window(X_train_val, Y_train_val_norm)
        X_test_sequenced, Y_test_sequenced = self.preprocessor.make_sliding_window(X_test, Y_test_norm)

        return X_sequenced, Y_sequenced, X_test_sequenced, Y_test_sequenced, output_scaler

    def perform_hyperparameter_tuning(self, Param_tuning, X_sequenced, Y_sequenced, model = None, batch_size = 512):
        if Param_tuning:
            tuner = kt.Hyperband(
                MyHyperModel(window_size=self.window_size, X_train=X_sequenced),
                objective='val_loss',
                max_epochs=30,
                executions_per_trial=3,
                overwrite=True,
                factor=3,
            )
            tuner.search(X_sequenced, Y_sequenced, epochs=50, validation_split=0.25)
            
            best_hps = tuner.get_best_hyperparameters()[0]
            print(f"Optimized Hyperparameters:")
            try:
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
            except:
                print(
                    f"""
                units_1 : {best_hps.get('units_1')}
                units_4 : {best_hps.get('dropout_1')}
                regularizer : {best_hps.get('regularizer')}
                learning_rate : {best_hps.get('learning_rate')}
                batch_size : {best_hps.get('batch_size')}
                """
                )
            batch_size = best_hps.get("batch_size")
            model = tuner.hypermodel.build(best_hps)
            
        else:
            model = model
        
        return model, batch_size

    def train_model(self, model, X_sequenced, Y_sequenced, epochs, batch_size):
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
        return model, history

    def denormalize_and_save_data(self, model, output_scaler, X_test_sequenced, Y_test_sequenced, batch_size, epochs):
        pred_norm = model.predict(X_test_sequenced)
        pred = pred_norm
        pred, Y_test_sequenced = self.preprocessor.denormalize_output_data(output_scaler, pred_norm, Y_test_sequenced)
        
        model_path = self.logger.save_model(model)
        self.logger.model_log_save(batch_size, self.window_size, epochs, pred, Y_test_sequenced)

    def check_model(self, model2load, output_scaler, X_test_sequenced, Y_test_sequenced, batch_size, epochs):
        # Load Model
        with CustomObjectScope({"MultiHeadAttention": MHeadAttention}):
            modelpath = f"{modelMainPath}{model2load}"
            loaded_model = load_model(modelpath) 
        loaded_model.summary()

        # De-normalize Output
        pred_norm = loaded_model.predict(X_test_sequenced)
        pred = pred_norm
        pred, Y_test_sequenced = self.preprocessor.denormalize_output_data(output_scaler, pred_norm, Y_test_sequenced)

        model_path = self.logger.save_model(loaded_model)
        self.logger.model_log_save(batch_size, self.window_size, epochs, pred, Y_test_sequenced)


if __name__ == '__main__':  
    num_variables = 12
    num_outputs = 1
    window_size = 60
    data_processor = DataProcessor(num_variables, num_outputs, window_size)

