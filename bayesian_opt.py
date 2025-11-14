# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy

from bayes_opt import BayesianOptimization


try:
    from bayes_opt.logger import JSONLogger
    from bayes_opt.event import Events

    USE_LOGGER = True
except ImportError:
    try:
        from bayes_opt import JSONLogger
        from bayes_opt import Events

        USE_LOGGER = True
    except ImportError:
        print("Warning: JSONLogger not available, logging disabled")
        USE_LOGGER = False

from transformer_dataloader_random import get_online_dataloader
from transformer_model_64x64 import FullModel


class BayesianOptimizer:
    def __init__(self):
        # Cesty k CSV souborům
        self.train_csv = "train_10G_balanced.csv"
        self.val_csv = "valid_10G_segmented.csv"

        # Parametry pro optimalizaci
        self.num_epochs = 100
        self.segment_length = 40000
        self.d_model = 64
        self.n_heads = 2
        self.n_layers = 1
        self.num_classes = 11
        self.val_frequency = 1

        # Parametry pro dataloader
        self.positive_per_class = 1
        self.num_positive_classes = self.num_classes - 1
        self.negative_per_batch = 12
        self.iterations_per_epoch = 200

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{'=' * 60}")
        print(f"DEVICE INFO:")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("CUDA not available - running on CPU")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"{'=' * 60}")

        self.results_history = []

    def objective_function(self, learning_rate, dropout, batch_size):
        """
        Objective function pro Bayesian Optimization.
        Vrací metriku k MAXIMALIZACI
        """
        batch_size = int(batch_size)

        print(f"\n{'=' * 60}")
        print(f"Testing hyperparameters:")
        print(f"Learning Rate: {learning_rate:.6f}")
        print(f"Dropout: {dropout:.4f}")
        print(f"Batch Size: {batch_size}")
        print(f"{'=' * 60}")

        try:

            expected_train_batch_size = self.positive_per_class * self.num_positive_classes + self.negative_per_batch

            train_loader = get_online_dataloader(
                csv_file=self.train_csv,
                batch_size=expected_train_batch_size,
                segment_length=self.segment_length,
                positive_per_class=self.positive_per_class,
                negative_per_batch=self.negative_per_batch,
                iterations_per_epoch=self.iterations_per_epoch,
                num_classes=self.num_classes,
                validation=False
            )

            val_loader = get_online_dataloader(
                csv_file=self.val_csv,
                batch_size=128,
                segment_length=self.segment_length,
                shuffle=True,
                validation=True
            )

            model = FullModel(
                segment_length=self.segment_length,
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=dropout,
                n_transformer_layers=self.n_layers,
                num_classes=self.num_classes
            )
            model = model.to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            best_val_loss = float('inf')
            best_val_acc = 0.0

            for epoch in range(self.num_epochs):
                start_time = time.time()
                model.train()
                total_loss = 0.0
                total_corrects = 0
                total_samples = 0

                batch_loss = 0.0
                batch_corrects = 0
                batch_samples = 0

                total_batches = len(train_loader)
                validation_points = []
                val_interval = total_batches // self.val_frequency

                for i in range(self.val_frequency - 1):
                    validation_points.append((i + 1) * val_interval)
                validation_points.append(total_batches)

                for batch_idx, batch_data in enumerate(train_loader):
                    batch_segments, batch_labels = batch_data
                    inputs = batch_segments.squeeze(0)  # [1, 32, 40000] -> [32, 40000]
                    labels = batch_labels.squeeze(0)  # [1, 32] -> [32]

                    inputs = inputs.to(self.device, dtype=torch.float)
                    labels = labels.to(self.device, dtype=torch.long)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                    loss.backward()
                    optimizer.step()

                    batch_loss += loss.item() * inputs.size(0)
                    batch_corrects += torch.sum(preds == labels).item()
                    batch_samples += inputs.size(0)

                    total_loss += loss.item() * inputs.size(0)
                    total_corrects += torch.sum(preds == labels).item()
                    total_samples += inputs.size(0)

                    if (batch_idx + 1) in validation_points:
                        # Validace
                        model.eval()
                        val_loss = 0.0
                        val_corrects = 0
                        val_samples = 0

                        with torch.no_grad():
                            for val_batch_data in val_loader:
                                val_batch_segments, val_batch_labels = val_batch_data
                                val_inputs = val_batch_segments.squeeze(0)
                                val_labels = val_batch_labels.squeeze(0)

                                val_inputs = val_inputs.to(self.device, dtype=torch.float)
                                val_labels = val_labels.to(self.device, dtype=torch.long)
                                val_outputs = model(val_inputs)
                                v_loss = criterion(val_outputs, val_labels)
                                v_preds = torch.argmax(val_outputs, dim=1)
                                val_loss += v_loss.item() * val_inputs.size(0)
                                val_corrects += torch.sum(v_preds == val_labels).item()
                                val_samples += val_inputs.size(0)

                        val_epoch_loss = val_loss / val_samples
                        val_epoch_acc = val_corrects / val_samples * 100

                        if val_epoch_loss < best_val_loss:
                            best_val_loss = val_epoch_loss
                            best_val_acc = val_epoch_acc

                        model.train()
                        batch_loss = 0.0
                        batch_corrects = 0
                        batch_samples = 0

                epoch_train_loss = total_loss / total_samples
                epoch_train_acc = total_corrects / total_samples * 100
                epoch_time = (time.time() - start_time) / 60

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.num_epochs}: Train Loss: {epoch_train_loss:.4f}, "
                          f"Train Acc: {epoch_train_acc:.2f}%, Best Val Loss: {best_val_loss:.4f}, "
                          f"Best Val Acc: {best_val_acc:.2f}%, Time: {epoch_time:.2f}min")

            result = {
                'learning_rate': learning_rate,
                'dropout': dropout,
                'batch_size': batch_size,
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'timestamp': datetime.now().isoformat()
            }
            self.results_history.append(result)

            print(f"Final Results - Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%")

            return -best_val_loss

        except Exception as e:
            print(f"Error in objective function: {str(e)}")
            import traceback
            traceback.print_exc()
            return -999

    def optimize(self, n_calls=3, init_points=5):
        """
        Spuštění Bayesian Optimization
            n_calls: Celkový počet evaluací
            init_points: Počet náhodných inicializačních bodů
        """
        bounds = {
            'learning_rate': (0.00001, 0.001),
            'dropout': (0.0, 0.5),
            'batch_size': (8, 64)
        }

        print("Hyperparameter bounds:")
        for param, (low, high) in bounds.items():
            print(f"  {param}: [{low}, {high}]")

        optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=bounds,
            random_state=42,
            verbose=2
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"bayesian_opt_log_{timestamp}.json"

        if USE_LOGGER:
            try:
                logger = JSONLogger(path=log_path)
                optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
                print(f"Logging enabled - results will be logged to: {log_path}")
            except Exception as e:
                print(f"Warning: Could not setup logging: {e}")
        else:
            print("Logging disabled - JSONLogger not available")

        print(f"\nStarting Bayesian Optimization with {n_calls} total calls...")

        from bayes_opt import UtilityFunction

        acquisition_function = UtilityFunction(kind="ucb", kappa=2.576)

        optimizer.maximize(
            init_points=init_points,
            n_iter=n_calls - init_points,
            acquisition_function=acquisition_function
        )

        print(f"\n{'=' * 60}")
        print("OPTIMIZATION COMPLETED!")
        print(f"{'=' * 60}")
        print(f"Best parameters found:")
        print(f"Learning Rate: {optimizer.max['params']['learning_rate']:.6f}")
        print(f"Dropout: {optimizer.max['params']['dropout']:.4f}")
        print(f"Batch Size: {int(optimizer.max['params']['batch_size'])}")
        print(f"Best Score (negative val loss): {optimizer.max['target']:.4f}")
        print(f"Best Validation Loss: {-optimizer.max['target']:.4f}")

        results_path = f"optimization_results_{timestamp}.json"
        results_summary = {
            'best_params': optimizer.max['params'],
            'best_score': optimizer.max['target'],
            'best_val_loss': -optimizer.max['target'],
            'all_results': self.results_history,
            'optimization_settings': {
                'n_calls': n_calls,
                'init_points': init_points,
                'bounds': bounds,
                'num_epochs': self.num_epochs,
                'iterations_per_epoch': self.iterations_per_epoch
            }
        }

        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"Detailed results saved to: {results_path}")
        return optimizer.max['params']

    def train_with_best_params(self, best_params, num_epochs=150):
        """
        Trénování finálního modelu s nejlepšími nalezenými hyperparametry
        """
        print(f"\n{'=' * 60}")
        print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
        print(f"{'=' * 60}")

        learning_rate = best_params['learning_rate']
        dropout = best_params['dropout']
        batch_size = int(best_params['batch_size'])

        print(f"Learning Rate: {learning_rate:.6f}")
        print(f"Dropout: {dropout:.4f}")
        print(f"Batch Size: {batch_size}")
        print(f"Epochs: {num_epochs}")

        expected_train_batch_size = self.positive_per_class * self.num_positive_classes + self.negative_per_batch

        train_loader = get_online_dataloader(
            csv_file=self.train_csv,
            batch_size=expected_train_batch_size,
            segment_length=self.segment_length,
            positive_per_class=self.positive_per_class,
            negative_per_batch=self.negative_per_batch,
            iterations_per_epoch=200,
            num_classes=self.num_classes,
            validation=False
        )

        val_loader = get_online_dataloader(
            csv_file=self.val_csv,
            batch_size=128,
            segment_length=self.segment_length,
            shuffle=True,
            validation=True
        )

        model = FullModel(
            segment_length=self.segment_length,
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=dropout,
            n_transformer_layers=self.n_layers,
            num_classes=self.num_classes
        )
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            start_time = time.time()

            model.train()
            total_loss = 0.0
            total_corrects = 0
            total_samples = 0

            for batch_idx, batch_data in enumerate(train_loader):
                batch_segments, batch_labels = batch_data
                inputs = batch_segments.squeeze(0)
                labels = batch_labels.squeeze(0)

                inputs = inputs.to(self.device, dtype=torch.float)
                labels = labels.to(self.device, dtype=torch.long)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                total_corrects += torch.sum(preds == labels).item()
                total_samples += inputs.size(0)

            # Validace
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            val_samples = 0

            with torch.no_grad():
                for val_batch_data in val_loader:
                    val_batch_segments, val_batch_labels = val_batch_data
                    val_inputs = val_batch_segments.squeeze(0)
                    val_labels = val_batch_labels.squeeze(0)

                    val_inputs = val_inputs.to(self.device, dtype=torch.float)
                    val_labels = val_labels.to(self.device, dtype=torch.long)
                    val_outputs = model(val_inputs)
                    v_loss = criterion(val_outputs, val_labels)
                    v_preds = torch.argmax(val_outputs, dim=1)
                    val_loss += v_loss.item() * val_inputs.size(0)
                    val_corrects += torch.sum(v_preds == val_labels).item()
                    val_samples += val_inputs.size(0)

            val_epoch_loss = val_loss / val_samples
            val_epoch_acc = val_corrects / val_samples * 100
            train_epoch_loss = total_loss / total_samples
            train_epoch_acc = total_corrects / total_samples * 100

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            epoch_time = (time.time() - start_time) / 60
            print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.2f}%, "
                  f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%, "
                  f"Time: {epoch_time:.2f}min")

        unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        model.load_state_dict(best_model_wts)
        model_path = f"best_model_optimized_{unique_id}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"\nBest model saved as: {model_path}")
        print(f"Best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    print("Hello, world!")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected or CUDA not available")

    bay_opt = BayesianOptimizer()

    best_params = bay_opt.optimize(
        n_calls=3,  # Celkový počet kombinací hyperparametrů k vyzkoušení
        init_points=5  # Počet náhodných inicializačních bodů
    )

    #bay_opt.train_with_best_params(best_params, num_epochs=150)