import os.path
import torch
from torch import nn, Tensor
from transformers import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
    SequenceClassifierOutput,
)

from src.bug_prediction.BugPredictionArgs import model_class_of
from src.bug_prediction.FileEncoders import (
    RobertaForFileClassification,
    T5Pooler,
)


class BugPredictionModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        config: PretrainedConfig,
        encoder_type: str,  # line or file
        is_checkpoint: bool,
        pad_token_id: int,
        model_type: str,
        max_line_length: int,
        max_file_length: int,
        class_weight: torch.Tensor,
        handcrafted_dim: int = 32,  # [New] Parameter for handcrafted features dimension
    ):
        super(BugPredictionModel, self).__init__()

        model_class = model_class_of[model_type]
        print(f"{model_class=}")

        # token encoder: Used for token-level encoding of each line
        if is_checkpoint:
            self.token_encoder = model_class.line_encoder(config=config)
        else:
            self.token_encoder = model_class.line_encoder.from_pretrained(
                pretrained_model_name, config=config
            )

        # t5 special handling
        if model_type == "t5":
            self.t5_pooler = T5Pooler(config)

        # [New] Fusion Layer: Projects (Hidden + Handcrafted) back to Hidden size
        # Early Fusion Strategy: Concatenate -> Linear Projection -> Activation -> Dropout
        self.handcrafted_dim = handcrafted_dim
        self.fusion_projection = nn.Linear(
            config.hidden_size + handcrafted_dim, config.hidden_size
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # line/file encoder (Context Model)
        if encoder_type == "line":
            self.line_encoder = model_class.file_encoder(
                config, class_weight=class_weight
            )
        elif encoder_type == "file":
            assert model_type == "roberta"
            self.file_encoder = RobertaForFileClassification(
                config, class_weight=class_weight
            )
        else:
            raise ValueError(f"Invalid {encoder_type=}")

        # If it is a checkpoint, load the entire state_dict directly
        if is_checkpoint:
            self.load_state_dict(
                torch.load(os.path.join(pretrained_model_name, "pytorch_model.bin"))
            )

        self.config = config
        self.pad_token_id = pad_token_id
        self.model_type = model_type
        self.max_line_length = max_line_length
        self.max_file_length = max_file_length

    def get_roberta_vec(
        self,
        source_tensor: Tensor,
        label_tensor: Tensor,
        handcrafted_tensor: Tensor,  # [New] Input for handcrafted features
        output_attentions=False,
    ):
        """
        source_tensor: [B, max_file_length, max_line_length]
        label_tensor: [B, max_file_length] or [B]
        handcrafted_tensor: [B, max_file_length, 32]
        """
        token_encoder_outputs = []
        token_attentions = []

        # 1. Encode each line of code (Semantic Representation)
        for file_tensor in source_tensor:
            token_encoder_output: BaseModelOutputWithPoolingAndCrossAttentions = (
                self.token_encoder(
                    input_ids=file_tensor,
                    attention_mask=file_tensor.ne(self.pad_token_id),
                    output_attentions=output_attentions,
                )
            )

            # t5 uses custom pooler
            if self.model_type == "t5":
                token_encoder_output.pooler_output = self.t5_pooler(
                    token_encoder_output.last_hidden_state
                )

            token_encoder_outputs.append(token_encoder_output.pooler_output)

            if output_attentions:
                token_attentions.append(token_encoder_output.attentions)

        # line_tensor: [B, max_file_length, hidden_size]
        line_tensor = torch.stack(token_encoder_outputs)
        assert line_tensor.shape[1:] == (self.max_file_length, self.config.hidden_size)

        # [New] 2. Early Fusion
        if handcrafted_tensor is not None:
            # Check dimensions: [Batch, File_Len, 32]
            assert handcrafted_tensor.shape[1:] == (
                self.max_file_length,
                self.handcrafted_dim,
            ), f"Expected shape [B, {self.max_file_length}, {self.handcrafted_dim}], got {handcrafted_tensor.shape}"

            # Concatenate: [Batch, Len, Hidden] + [Batch, Len, 32] -> [Batch, Len, Hidden+32]
            fused_tensor = torch.cat([line_tensor, handcrafted_tensor], dim=-1)

            # Project back: [Batch, Len, Hidden+32] -> [Batch, Len, Hidden]
            line_tensor_fused = self.fusion_projection(fused_tensor)
            line_tensor_fused = torch.relu(line_tensor_fused)  # Activation
            line_tensor = self.dropout(line_tensor_fused)  # Dropout

        # 3. Context Modeling (Line/File Encoder)
        # attention mask for line encoder
        attention_mask = source_tensor[:, :, 0].ne(self.pad_token_id)
        if self.model_type == "t5":
            attention_mask = attention_mask[:, None, :]
        else:
            attention_mask = attention_mask[:, None, None, :]

        # Feed into line/file encoder
        if hasattr(self, "line_encoder"):
            encoder_output = self.line_encoder(
                hidden_states=line_tensor,  # fused tensor
                attention_mask=attention_mask,
                labels=label_tensor,
            )
        elif hasattr(self, "file_encoder"):
            encoder_output = self.file_encoder(
                hidden_states=line_tensor,  # fused tensor
                attention_mask=attention_mask,
                labels=label_tensor,
            )
        else:
            raise ValueError("line_encoder or file_encoder should be defined")

        if output_attentions:
            return (
                encoder_output.loss,
                encoder_output.logits,
                token_attentions,
            )
        return encoder_output.loss, encoder_output.logits

    def forward(
        self,
        source_tensor: Tensor,
        label_tensor: Tensor,
        handcrafted_tensor: Tensor = None,  # [New] Optional argument
        output_attentions: bool = False,
    ):
        """
        Returns:
            loss: scalar
            logits: [B,L] or [B]
            (attentions optional)
        """
        assert (
            (source_tensor.dim() == 3)
            and (source_tensor.size(1) == self.max_file_length)
            and (source_tensor.size(2) == self.max_line_length)
        ), source_tensor.size()

        # [New] Pass handcrafted_tensor to get_roberta_vec
        loss, logits, *token_attentions = self.get_roberta_vec(
            source_tensor,
            label_tensor,
            handcrafted_tensor=handcrafted_tensor,
            output_attentions=output_attentions,
        )

        # Logits are already probabilities [0,1], no softmax needed
        # No class dimension check performed here as per original code

        if output_attentions:
            return loss, logits, token_attentions[0]
        return loss, logits