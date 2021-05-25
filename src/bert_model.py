"""
Author: Amund Faller Råheim

Custom BERT model, extending BertPreTrainedModel
in the same way as the transformers library

Requires:
A pretrained BERT model, or a reference to a pretrained Huggingface model
"""

from typing import List
from os.path import join, isdir
import time

import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BertBinaryClassification(BertPreTrainedModel):
    def __init__(self, config):
        """
        :param config: a transformers Config object
        """
        self.num_labels = 1
        super().__init__(config, num_labels=self.num_labels)

        # The BertModel without pooling layer
        self.bert = BertModel(config, add_pooling_layer=False)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Input dimension depending on number of tokens encodings
        input_dim = config.hidden_size * 3

        cls_layers = []
        cls_layers.append(nn.Linear(input_dim, config.hidden_size))
        cls_layers.append(nn.GELU())
        cls_layers.append(
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            )
        cls_layers.append(nn.Linear(config.hidden_size, self.num_labels))

        # The classifier:
        self.classifier = nn.Sequential(*cls_layers)

        # Used in BCEWithLogitsLoss function
        #  to counteract unbalanced training sets
        # Can be changed with self.set_class_weights
        self.class_weights = torch.ones([1])

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
            ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Use either pooled output or sequence output, depending on settings
        bert_output = outputs[0]

        # Get the state of the [CLS] token,
        # and the first token of the mention and candidate entity

        # Position of the first token of the candidate
        # (right after the [SEP] token)
        cand_pos = torch.argmax(token_type_ids, dim=1)

        # Get the embedding of the first token of the candidate over the batch
        cand_tensors = torch.cat(
            [t[i] for t, i in zip(bert_output, cand_pos)]
        ).reshape((bert_output.size(0), bert_output.size(-1)))

        # Flattened input of 3 * hidden_size features
        bert_output = torch.cat([bert_output[:, 0],
                                bert_output[:, 1],
                                cand_tensors], dim=1)

        bert_output = self.dropout(bert_output)
        logits = self.classifier(bert_output)

        loss = None
        if labels is not None:
            # Binary cross entropy loss with class weights and sigmoid
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
            loss = loss_fn(
                    logits.view(-1),
                    labels.view(-1).to(dtype=torch.float)
                )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def freeze_layers(self, param_idx: List):
        """
        Freeze layers at provided indices to not train them
        :param param_idx: list of indices of layers to be frozen
        """
        module_params = list(self.named_parameters())
        for param in (module_params[p] for p in param_idx):
            param[1].requires_grad = False

    def freeze_n_transformers(self, n: int = 11):
        """
        Freeze the provided number of encoders in the BERT architecture
         to not train them
        :param n: number of encoders to freeze
        """
        n = min(n, 12)
        n_emb_layers = 5
        n_layers_in_transformer = 12
        emb_layers = list(range(n_emb_layers))
        encoder_layers = list(
                range(
                        n_emb_layers,
                        n_emb_layers + n * n_layers_in_transformer
                    )
            )
        self.freeze_layers(emb_layers + encoder_layers)

    def freeze_bert(self):
        """
        Freezes all layers in BERT from training
        """
        for param in self.bert.named_parameters():
            param[1].requires_grad = False

    def set_class_weights(self, class_weights):
        """
        Set the self.class_weights used to penalize wrong prediction of
         minority class
        :param class_weights: a pytorch tensor with weights over the two
                              classes. Used by the CrossEntropyLoss.
        """
        self.class_weights = class_weights


def load_bert_from_file(model_path: str):
    """
    Load a BertBinaryClassification model from file
    :param model_path: path to directory with BertBinaryClassification
        model and config file
    :returns: BertBinaryClassification model
    """
    if not isdir(model_path):
        raise FileNotFoundError(f"No BERT model at directory {model_path}.")
    model = BertBinaryClassification.from_pretrained(model_path)
    return model


def save_bert_to_file(model: BertBinaryClassification, model_dir: str):
    """
    Write a BertBinaryClassification model to file
    :param model: a BertBinaryClassification model
    :param model_path: path to destination directory for the model
    """
    sub_dir = "saved_" + time.strftime('%Y%m%d_%H%M', time.gmtime(time.time()))
    model_dir = join(model_dir, sub_dir)
    print(f"Saving model to directory: {model_dir}")
    model.save_pretrained(model_dir)


def get_class_weights_tensor(neg, pos):
    """
    :param neg: number of negative labels in the dataset
    :param pos: number of positive labels in the dataset
    :returns: torch Tensor with class weights for unbalanced dataset,
     provide to BertBinaryClassification.set_class_weights function
    """
    # Weight by the frequency of the other label
    tot = neg + pos
    return torch.Tensor([pos / tot, neg / tot])
