from utility.tsg_dataprocessor import *

# Input
Model_load = True
file_names = ["sample_data4learning.mat", "sample_data4learning2.mat"]
# file_names = ["sample_data4learning.mat"]
model2load = "lstm_200"

# Parameter
num_variables = 12
num_outputs = 1
window_size = 60
batch_size = 512
epochs = 200


if __name__ == '__main__':  
    data_processor = DataProcessor(num_variables, num_outputs, window_size)
    X_sequenced, Y_sequenced, X_test_sequenced, Y_test_sequenced, output_scaler = data_processor.load_and_preprocess_data(dataMainPath, file_names, 20, 4)
    
    if Model_load:
        data_processor.check_model(model2load, output_scaler, X_test_sequenced, Y_test_sequenced, batch_size, epochs)
    else:
        # Define and train the model
        model = LSTMModel(window_size, num_variables, num_outputs)
        model = data_processor.train_model(model, X_sequenced, Y_sequenced, epochs, batch_size)
        
        # Denormalize and save the results
        data_processor.denormalize_and_save_data(model, output_scaler, X_test_sequenced, Y_test_sequenced, batch_size, epochs)
