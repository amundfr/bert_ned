import time
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from os.path import isdir, join
from collections import Counter
from src.bert_model import BertBinaryClassification
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader


def format_time(elapsed):
    """
  Takes a time in seconds and returns a string hh:mm:ss
  """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as "d days, hh:mm:ss"
    return str(datetime.timedelta(seconds=elapsed_rounded))


def accuracy_over_entities(preds, labels, docs, entities):
    """
    Calculate the accuracy of a classification prediction
    over entities, using the argmax of the preds.

    This is achieved by grouping by entities, and choosing the strongest (if any)
    positive prediction as the only prediction of the True label.

    :param preds: logit predictions from the model
    :param labels: the correct true/false labels for each candidate/data point
    :param docs: list of the docs that the data points come from, to group data points
    :param entities: list of the entities that the data points belong to, 
        for grouping into accuracy over entities rather than candidates
    :returns: a scalar accuracy for this batch, averaged over entities
    """
    assert len(preds) == len(labels)
    assert len(preds) == len(docs)
    assert len(preds) == len(entities)

    # Counts number of data points for each candidate, 
    #  identified by doc ID + entity ID
    ctr = Counter(zip(docs, entities))
    n = len(ctr)
    
    # Iterate the lists entity by entity:
    tot_accuracy = 0
    i = 0
    while i < len(preds):
        key = (docs[i], entities[i])
        n_data_points = ctr[key]
        entity_preds = preds[i:i+n_data_points]
        entity_labels = labels[i:i+n_data_points]
        # TODO: Pick the top True, or pick all that are True?
        pred_true = np.argmax(entity_preds, axis=0)[1]
        pred = np.eye(n_data_points)[pred_true]
        entity_accuracy = np.sum(pred == entity_labels.flatten()) / n_data_points
        tot_accuracy += entity_accuracy
        i += n_data_points
    return tot_accuracy / n
    

def accuracy_over_candidates(preds, labels):
    """
    Calculate the accuracy of a classification prediction
    over candidates ("naïvely"), using the argmax of the preds.

    :param preds: logit predictions from the model
    :param labels: the correct true/false labels for each data point
    :returns: a scalar accuracy for this batch, averaged over candidates
    """
    # TODO: Make this operate on Tensors
    pred_classes = np.argmax(preds, axis=1).flatten()
    return np.sum(pred_classes == labels.flatten()) / len(labels)


def plot_training_stats(training_stats, save_to_dir: str = None):
    x_ = list(range(len(training_stats)))

    # Increase the size of the plot (set dots per inch)
    scale = 1.8
    fig_1, ax_1 = plt.subplots(dpi=scale * 72)

    tr_loss = [s['Training Loss'] for s in training_stats]
    ax_1.scatter(x_, tr_loss, s=64, marker='x')
    ax_1.plot(x_, tr_loss)

    val_loss = [s['Valid. Loss'] for s in training_stats]
    ax_1.scatter(x_, val_loss, s=64, marker='x')
    ax_1.plot(x_, val_loss)

    ax_1.grid()
    fig_1.legend(["Training loss", "Validation loss"])

    fig_2, ax_2 = plt.subplots(dpi=scale * 72)

    tr_loss = [s['Valid. Accur.'] for s in training_stats]
    ax_2.scatter(x_, tr_loss, s=64, marker='x')
    ax_2.plot(x_, tr_loss)
    ax_2.grid()

    if save_to_dir and isdir(save_to_dir):
        fig_1.savefig(join(save_to_dir, 'losses.png'))
        fig_2.savefig(join(save_to_dir, 'accuracy.png'))

    fig_1.show()
    fig_2.show()


class ModelTrainer:
    def __init__(self,
                 model: BertBinaryClassification,
                 device: torch.device,
                 train_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 epochs: int = 3):
        self.device = device
        self.model = model

        # Model is moved to device in-place, but tensors are not:
        # Source: https://discuss.pytorch.org/t/model-move-to-device-gpu/105620
        self.model.class_weights = self.model.class_weights.to(device)
        self.model.to(device)

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
        if type == 'train':
            # Put the model into training mode
            self.model.train()
            dataloader = self.train_dataloader
            torch.set_grad_enabled(True)
            find_accuracy = False
        elif type == 'validation':
            # Put the model into evaluation mode
            self.model.eval()
            dataloader = self.validation_dataloader
            # TODO: Test that this works (is persistent over the function)
            torch.set_grad_enabled(False)
            find_accuracy = True
        elif type == 'test':
            # Put the model into evaluation mode
            self.model.eval()
            dataloader = self.test_dataloader
            torch.set_grad_enabled(False)
            find_accuracy = True
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

    def train(self, train_update_freq: int = 50, valdation_update_freq: int = 50):
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
            avg_val_accuracy = accuracy(logits, labels, docs_entities)
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


    def test(self, dataset_to_docs, dataset_to_entities, test_update_freq: int = 50):
        """
        :param dataset_to_docs: A list of doc "IDs" 
            for the total (i.e. not split) dataset.
        :param dataset_to_entities: A list of entity "IDs" 
            for the total (i.e. not split) dataset.
        :param test_update_freq: how many batches to run before printing feedback
        """
        print("Running Testing...")

        total_loss, test_duration, logits, labels = self.run_epoch('test', test_update_freq)

        testdata_start = len(self.train_dataloader.dataset.indices) + \
                         len(self.validation_dataloader.dataset.indices)
        docs = dataset_to_docs[testdata_start:testdata_start+len(labels)]
        entities = dataset_to_entities[testdata_start:testdata_start+len(labels)]
        # Average accuracy over batches
        avg_accuracy = accuracy_over_entities(logits, labels, docs, entities)

        # Average loss over batches.
        avg_loss = total_loss / len(self.test_dataloader)

        test_duration = format_time(test_duration)

        print("\nTesting complete!")

        print(f"  Testing took {test_duration} (h:mm:ss)")
        print(f"  Test Loss: {avg_loss:.2f}")
        print(f"  Test accuracy: {avg_accuracy:.4f}")
