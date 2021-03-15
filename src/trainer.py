import time
import datetime
import numpy as np
import torch

from os.path import isdir, join
from collections import Counter
from src.bert_model import BertBinaryClassification
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from src.evaluation import accuracy_over_mentions, accuracy_over_candidates


def format_time(elapsed):
    """
  Takes a time in seconds and returns a string hh:mm:ss
  """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as "d days, hh:mm:ss"
    return str(datetime.timedelta(seconds=elapsed_rounded))


class ModelTrainer:
    def __init__(self,
                 model: BertBinaryClassification,
                 train_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 epochs: int = 3):
        self.model = model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model is moved to device in-place, but tensors are not:
        # Source: https://discuss.pytorch.org/t/model-move-to-device-gpu/105620
        self.model.to(self.device)
        self.model.set_class_weights(self.model.class_weights.to(self.device))

        # Use Cuda if Cuda enabled GPU is available
        if torch.cuda.is_available():
            print('Using device:', torch.cuda.get_device_name(0))
        else:
            print('Using CPU')


        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = AdamW(
                self.model.parameters(),
                lr=2e-5,  # base learning rate (TODO: do HPO on this parameter)
                weight_decay=0.001  # weight decay (TODO: HPO)
            )
        self.epochs = epochs

        total_steps = len(self.train_dataloader) * self.epochs

        # Create the learning rate scheduler. TODO: experiment! e.g. get_cosine_with_hard_restarts_schedule_with_warmup
        self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,  # warm start
                num_training_steps=total_steps
            )

    def run_epoch(self, type: str, feedback_frequency: int = 50):
        """
        :param type: is 'train', 'validation' or 'test'
        :param feedback_frequency: print progress after this many batches
        """
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Used if in evaluation mode (type 'validation' or 'test')
        epoch_logits = np.ndarray((0,2))
        epoch_labels = np.ndarray((0,1))

        # Setup for different epoch modes
        if type == 'train':
            # Put the model into training mode
            self.model.train()
            dataloader = self.train_dataloader
            torch.set_grad_enabled(True)
        elif type == 'validation':
            # Put the model into evaluation mode
            self.model.eval()
            dataloader = self.validation_dataloader
            # TODO: Test that this works (is persistent over the function)
            torch.set_grad_enabled(False)
        elif type == 'test':
            # Put the model into evaluation mode
            self.model.eval()
            dataloader = self.test_dataloader
            torch.set_grad_enabled(False)
        else:
            print(f"type must be 'train', 'validation' or 'test'. Was {type}.")
            return

        # For each batch of training data...
        for step, batch in enumerate(dataloader):
            # Progress update every few batches.
            if step % feedback_frequency == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print(f"  Batch {step:>5,}  of  {len(dataloader):>5,}.    "
                      f"Elapsed: {elapsed:}.    Avg loss: {total_loss / step:.4f}")

            # Unpack this training batch from the dataloader and move to correct device
            b_input_ids = batch[0].to(device=self.device, dtype=torch.long)
            b_attention_mask = batch[1].to(device=self.device, dtype=torch.long)
            b_token_type_ids = batch[2].to(device=self.device, dtype=torch.long)
            b_labels = batch[3].to(device=self.device, dtype=torch.long)

            # Reset gradients
            self.model.zero_grad()

            # Forward pass on the model
            outputs = self.model(
                    b_input_ids,
                    token_type_ids=b_token_type_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                 )
            # Loss of the batch from the forward pass' loss function
            loss = outputs[0]
            # The final activations (i.e. the prediction)
            logits = outputs[1]

            # Add to epoch loss
            total_loss += loss.item()

            # Take a training step, if training
            if type == 'train':
                # Perform a backward pass to calculate the gradients
                loss.backward()

                # Clip the norm of the gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update the weights with the optimizer
                self.optimizer.step()

                # Tell the scheduler to update the learning rate
                self.scheduler.step()

            # If evaluating, get predictions and labels
            else:
                _logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                epoch_logits = np.append(epoch_logits, _logits, axis=0)
                epoch_labels = np.append(epoch_labels, label_ids, axis=0)

        epoch_duration = time.time()-t0
        return total_loss, epoch_duration, epoch_logits, epoch_labels

    def train(self, train_update_freq: int = 50, validation_update_freq: int = 50):
        """
        :param test_update_freq: how many training batches to run before printing feedback
        :param validation_update_freq: how many validation batches to run before printing feedback
        """
        # Holds some metrics on the duration and result of the training and validation
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # If validation accuracy doesn't improve by more than this threshold,
        # early stopping is triggered
        early_stopping_threshold = 0.001

        print(f"\n   Training starts at {time.ctime(total_t0)}\n")

        # For each epoch...
        for epoch_i in range(0, self.epochs):

            # Perform one full pass over the training set

            print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('\nTraining...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            total_train_loss, training_duration, _, _ = self.run_epoch('train', train_update_freq)

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.train_dataloader)

            # Measure how long this epoch took.
            training_duration = format_time(training_duration)

            print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_duration))

            # Perform one full pass over the validation set

            print("\nRunning Validation...")

            t0 = time.time()

            total_eval_loss, eval_duration, val_logits, val_labels = self.run_epoch('validation', validation_update_freq)

            validation_duration = format_time(eval_duration)

            # Report the final accuracy for this validation run.
            # total_eval_accuracy = 0
            avg_val_accuracy = accuracy_over_candidates(logits, labels)
            print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(self.validation_dataloader)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_duration))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_duration,
                    'Validation Time': validation_duration
                }
            )

            # Early stopping if validation loss doesn't improve sufficiently
            if epoch_i > 1:
                delta_val_loss = training_stats[-2]['Valid. Accur'] - training_stats[-1]['Valid. Accur']

                if delta_val_loss < early_stopping_threshold:
                    print(f"Triggering early stopping after {epoch_i + 1} epoch: "
                          f"Validation loss change {delta_val_loss:.4} < threshold {early_stopping_threshold}")
                    break

        print("\nTraining complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

        stats_h1 = "      Training            |    Validation"
        stats_h2 = "Epoch |   Time   |  Loss  |   Time   |  Loss  | Accuracy "
        stats_fs = "{:^5} | {:>8} | {:6.4f} | {:>8} | {:6.4f} | {:6.4f}"

        print("\n")
        print(stats_h1)
        print('_' * len(stats_h2))
        print(stats_h2)
        print('_' * len(stats_h2))

        for s in training_stats:
            print(stats_fs.format(s['epoch'], s['Training Time'], s['Training Loss'],
                    s['Validation Time'], s['Valid. Loss'], s['Valid. Accur.']))

        return training_stats


    def test(self, dataset_to_doc, dataset_to_mention, \
             test_update_freq: int = 50, \
             dataset_to_candidate = None, result_file = '/data/evaluation_result.csv'):
        """
        Evaluate the model with the test dataset.
        Relies on mappings from data point to documents and mentions
        in order to group data points over mentions.
        If docs_mentions is provided, the functions prints the full result
        of the evaluation to either sysout or result_file if provided.

        :param dataset_to_doc: A list of doc "IDs"
            for all datasets (not just training data).
        :param dataset_to_mention: A list of mention "IDs"
            for all datasets (not just training data).
        :param test_update_freq: how many batches to run before printing feedback
        :param dataset_to_candidate: A list of candidates
            for all datasets (not just training data).
            If provided, enters a very verbose state, printing
            full results of the prediction in a csv format.
        :param result_file: Path to output file for full results.
            Provide a false value if you don't want to save output to file.
        """
        print("Running Testing...")

        total_loss, test_duration, preds, labels = self.run_epoch('test', test_update_freq)

        testdata_start = len(self.train_dataloader.dataset.indices) + \
                         len(self.validation_dataloader.dataset.indices)
        docs = dataset_to_doc[testdata_start:testdata_start+len(labels)]
        mentions = dataset_to_mention[testdata_start:testdata_start+len(labels)]
        candidates = None
        if dataset_to_candidate:
            candidates = dataset_to_candidate[testdata_start:testdata_start+len(labels)]

        # Average accuracy over mentions (uses full test dataset)
        avg_accuracy, result_str = accuracy_over_mentions(preds, labels, docs, mentions, candidates)

        if result_str:
            if result_file:
                result_dir = '/'.join(result_file.split('/')[:-1])
                if isdir(result_dir):
                    print(f"\nWriting results to file {result_file}\n")
                    print(result_str, file=open(result_file, 'w'))
                else:
                    print(f"\nCould not print to {result_file}. Could not find parent directory {result_dir}.")
            else:
                print(result_str)

        # Average loss over batches.
        avg_loss = total_loss / len(self.test_dataloader)

        test_duration = format_time(test_duration)

        print("\nTesting complete!")

        print(f"  Testing took {test_duration} (h:mm:ss)")
        print(f"  Test Loss: {avg_loss:.2f}")
        print(f"  Test accuracy: {avg_accuracy:.4f}")
