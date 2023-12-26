from utility.tsg_dataloader import *
from utility.tsg_preprocessor import *
from utility.tsg_logger import *
from utility.tsg_models import *
from utility.values import *
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm


class DataProcessor:
    def __init__(self, num_variables, num_outputs, window_size):
        self.num_variables = num_variables
        self.num_outputs = num_outputs
        self.window_size = window_size
        self.preprocessor = TimeSeriesPreprocessor(self.window_size)
        self.logger = ModelLogger(modelMainPath=modelMainPath)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_and_preprocess_data(self, dataMainPath, file_names, train_val_samples, test_samples):
        dataLoader = DataLoaderWrapper(dataMainPath=dataMainPath)
        dataLoader.load_mat_files(file_names)
        X_train_val, Y_train_val, X_test, Y_test = dataLoader.split_data(train_val_samples, test_samples)

        output_scaler = self.preprocessor.generate_normalization(Y_train_val, Y_test)
        Y_train_val_norm, Y_test_norm = self.preprocessor.normalize_all_output_data_with_scaler(Y_train_val, Y_test)

        X_sequenced, Y_sequenced = self.preprocessor.make_sliding_window(X_train_val, Y_train_val_norm)
        X_test_sequenced, Y_test_sequenced = self.preprocessor.make_sliding_window(X_test, Y_test_norm)

        return X_sequenced, Y_sequenced, X_test_sequenced, Y_test_sequenced, output_scaler

    def train_model(self, model, X_sequenced, Y_sequenced, epochs, batch_size):
        # Split data into training and validation sets
        train_indices, val_indices = train_test_split(np.arange(len(Y_sequenced)), test_size=0.25, random_state=42)

        train_dataset = Subset(TimeSeriesDataset(X_sequenced, Y_sequenced), train_indices)
        val_dataset = Subset(TimeSeriesDataset(X_sequenced, Y_sequenced), val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model.to(self.device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            # Training loop
            model.train()
            train_loss = 0
            for inputs, targets in tqdm(train_loader, desc='Training loop'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc='Validation loop'):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')

        return model

    def denormalize_and_save_data(self, model, output_scaler, X_test_sequenced, Y_test_sequenced, batch_size, epochs):
        model.eval()
        test_loader = DataLoader(TimeSeriesDataset(X_test_sequenced, Y_test_sequenced), batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy())

        pred, Y_test_sequenced = self.preprocessor.denormalize_output_data(output_scaler, predictions, Y_test_sequenced)

        model_path = self.logger.save_model(model)
        self.logger.model_log_save(model, batch_size, self.window_size, epochs, pred, Y_test_sequenced)

    def check_model(self, model2load, output_scaler, X_test_sequenced, Y_test_sequenced, batch_size, epochs):
        loaded_model = LSTMModel(window_size=self.window_size, num_variables=self.num_variables, num_outputs=self.num_outputs)
        
        model_path = f"{modelMainPath}{model2load}"
        loaded_model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
        loaded_model.to(self.device)

        test_loader = DataLoader(TimeSeriesDataset(X_test_sequenced, Y_test_sequenced), batch_size=batch_size, shuffle=False)
        
        loaded_model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = loaded_model(inputs)
                predictions.extend(outputs.cpu().numpy())

        pred, Y_test_sequenced = self.preprocessor.denormalize_output_data(output_scaler, predictions, Y_test_sequenced)

        self.logger.save_model(loaded_model)
        self.logger.model_log_save(loaded_model, batch_size, self.window_size, epochs, pred, Y_test_sequenced)



if __name__ == '__main__':  
    num_variables = 12
    num_outputs = 1
    window_size = 60
    data_processor = DataProcessor(num_variables, num_outputs, window_size)
