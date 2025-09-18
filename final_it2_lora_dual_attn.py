### FINAL CODE: ONLY THIS FILE
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from transformers import GenerationConfig
import copy

print("Started")

# ----- Dataset -----
class MultimodalDataset(Dataset):
    def __init__(self, ids_path, src_path, tgt_path, feature_dir, tokenizer_src, tokenizer_tgt, max_len=128):
        with open(ids_path) as f:
            self.ids = [line.strip() for line in f]
        with open(src_path) as f:
            self.src = [line.strip() for line in f]
        with open(tgt_path) as f:
            self.tgt = [line.strip() for line in f]
        self.feature_dir = feature_dir
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_tgt.padding_side = 'right'  # Pad on the right
        self.tokenizer_tgt._switch_to_target_mode()

        self.max_len = max_len
        self.src_lang_tag = "eng_Latn"
        self.tgt_lang_tag = "hin_Deva"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Text data
        src_text = self.src[idx]
        tgt_text = self.tgt[idx]
        vid = self.ids[idx]

        # Load image features (assume features under "arr_0")
        feature_path = os.path.join(self.feature_dir, f"{vid}.npz")
        features_archive = np.load(feature_path,allow_pickle=True)
        features = features_archive['features']  # shape: (x, 768)
        features = torch.from_numpy(features).float()

        src_text_with_tags = f"{self.src_lang_tag} {self.tgt_lang_tag} {src_text.strip()}"
        tgt_text_with_tags = f"{tgt_text.strip()}"

        # Tokenize source and target text
        src_enc = self.tokenizer_src(
            src_text_with_tags, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt'
        )
        with self.tokenizer_tgt.as_target_tokenizer():
            tgt_enc = self.tokenizer_tgt(
                tgt_text_with_tags, truncation=True, max_length=self.max_len-1, return_tensors='pt', add_special_tokens=True
            )
        bos_token_id = self.tokenizer_tgt.bos_token_id

        input_ids = tgt_enc['input_ids']
        attention_mask = tgt_enc['attention_mask']
        input_ids = torch.cat([torch.tensor([[bos_token_id]] * input_ids.size(0)), input_ids[:, :]], dim=1)
        attention_mask = torch.cat([torch.ones_like(attention_mask[:, :1]), attention_mask[:, :]], dim=1)
        pad_len = self.max_len - input_ids.size(1)
        if pad_len > 0:
            input_ids = torch.cat([
                input_ids,
                torch.full(
                    (input_ids.size(0), pad_len),
                    self.tokenizer_tgt.pad_token_id,
                    dtype=torch.long,
                    device=input_ids.device
                )
            ], dim=1)

            attention_mask = torch.cat([
                attention_mask,
                torch.zeros((attention_mask.size(0), pad_len), dtype=torch.long, device=attention_mask.device)
            ], dim=1)

        # Final encoded result
        tgt_enc = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        return {
            'src_input_ids': src_enc['input_ids'].squeeze(0),           # (max_len)
            'src_attention_mask': src_enc['attention_mask'].squeeze(0), # (max_len)
            'tgt_input_ids': tgt_enc['input_ids'].squeeze(0),           # (max_len)
            'tgt_attention_mask': tgt_enc['attention_mask'].squeeze(0), # (max_len)
            'features': features                                         # (variable_len, 768)
        }

# ----- Model Components -----

class LoRALinear(nn.Module):
    """LoRA wrapper for an existing Linear layer."""
    def __init__(self, linear_module, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.linear = linear_module  # original nn.Linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_features = linear_module.in_features
        out_features = linear_module.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)

        # LoRA params
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Init
        nn.init.normal_(self.lora_A.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_output = self.linear(x)
        lora_output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return original_output + lora_output

class AdapterLayer(nn.Module):
    """Adapter layer with bottleneck architecture"""
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Initialize to near-identity
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x + residual

class DualCrossAttentionLayer(nn.Module):
    """
    Custom decoder layer with separate cross-attention for image and text
    """
    def __init__(self, original_layer, hidden_size):
        super().__init__()
        # Copy original layer components
        self.self_attn = original_layer.self_attn
        self.self_attn_layer_norm = original_layer.self_attn_layer_norm
        self.fc1 = original_layer.fc1
        self.fc2 = original_layer.fc2
        self.final_layer_norm = original_layer.final_layer_norm
        self.dropout = nn.Dropout(p=getattr(original_layer, 'dropout', 0.1))
        
        # Original cross attention (will be used for text)
        self.text_cross_attn = original_layer.encoder_attn
        self.encoder_attn_layer_norm = original_layer.encoder_attn_layer_norm
        
        # New cross attention for images
        # self.img_cross_attn = copy.deepcopy(original_layer.encoder_attn)
        # self.img_attn_layer_norm = copy.deepcopy(original_layer.encoder_attn_layer_norm)
        # num_heads = self.text_cross_attn.num_heads
        # dropout = self.text_cross_attn.dropout
        # self.img_cross_attn = nn.MultiheadAttention(
        #     embed_dim=hidden_size,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     batch_first=True
        # )

        self.img_cross_attn = copy.deepcopy(original_layer.encoder_attn)

        # re-init only key & value projections
        nn.init.xavier_uniform_(self.img_cross_attn.k_proj.weight)
        nn.init.xavier_uniform_(self.img_cross_attn.v_proj.weight)
        for p in self.text_cross_attn.parameters():
            p.requires_grad = True

        for p in self.img_cross_attn.parameters():
            p.requires_grad = True
        self.img_attn_layer_norm = nn.LayerNorm(hidden_size)
        
        # Fusion layer to combine image and text cross-attention outputs
        # self.fusion_layer = nn.Linear(2*hidden_size, hidden_size)
        # self.fusion_layer_norm = nn.LayerNorm(hidden_size)
    
    def _expand_mask(self, mask, dtype, tgt_len=None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_len, seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        
        inverted_mask = 1.0 - expanded_mask
        
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        
    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, 
                encoder_attention_mask=None, img_hidden_states=None, img_attention_mask=None, 
                layer_head_mask=None, cross_attn_layer_head_mask=None, past_key_value=None, 
                output_attentions=False, use_cache=False, **kwargs):
        
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Self attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Self-attention with proper signature
        self_attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = self_attn_outputs[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Cross attention outputs
        cross_attn_weights = None
        img_cross_attn_weights = None
        
        # Cross attention with text encoder
        text_cross_attn_output = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states_norm = self.encoder_attn_layer_norm(hidden_states)
            
            # Expand encoder attention mask to 4D if needed
            if encoder_attention_mask is not None and encoder_attention_mask.dim() == 2:
                encoder_attention_mask_4d = self._expand_mask(
                    encoder_attention_mask, 
                    hidden_states_norm.dtype, 
                    tgt_len=seq_len
                )
            else:
                encoder_attention_mask_4d = encoder_attention_mask
            
            text_cross_attn_outputs = self.text_cross_attn(
                hidden_states_norm,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask_4d,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=None,
                output_attentions=output_attentions,
            )
            text_cross_attn_output = text_cross_attn_outputs[0]
            text_cross_attn_output = self.dropout(text_cross_attn_output)
            
            if output_attentions:
                cross_attn_weights = text_cross_attn_outputs[1]
        
        # Cross attention with image encoder
        img_cross_attn_output = None
        if img_hidden_states is not None:
            hidden_states_norm = self.img_attn_layer_norm(hidden_states)
            
            # Expand image attention mask to 4D if needed
            if img_attention_mask is not None and img_attention_mask.dim() == 2:
                img_attention_mask_4d = self._expand_mask(
                    img_attention_mask, 
                    hidden_states_norm.dtype, 
                    tgt_len=seq_len
                )
            else:
                img_attention_mask_4d = img_attention_mask
            
            img_cross_attn_outputs = self.img_cross_attn(
                hidden_states_norm,
                key_value_states=img_hidden_states,
                attention_mask=img_attention_mask_4d,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=None,
                output_attentions=output_attentions,
            )
            img_cross_attn_output = img_cross_attn_outputs[0]
            img_cross_attn_output = self.dropout(img_cross_attn_output)
            
            if output_attentions:
                img_cross_attn_weights = img_cross_attn_outputs[1]
        
        # Combine cross-attention outputs
        if text_cross_attn_output is not None and img_cross_attn_output is not None:
            # Concatenate and fuse
            # combined = torch.cat([text_cross_attn_output, img_cross_attn_output], dim=-1)
            # fused_output = self.fusion_layer(combined)
            # fused_output = self.fusion_layer_norm(fused_output)
            hidden_states = residual + text_cross_attn_output + img_cross_attn_output
        elif text_cross_attn_output is not None:
            hidden_states = residual + text_cross_attn_output
        elif img_cross_attn_output is not None:
            hidden_states = residual + img_cross_attn_output
        
        # Feed forward
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_outputs[1], cross_attn_weights, img_cross_attn_weights)
        
        if use_cache:
            outputs += (self_attn_outputs[-1],)
        
        return outputs


class CustomDecoder(nn.Module):
    """
    Custom decoder that handles dual cross-attention
    """
    def __init__(self, original_decoder, hidden_size):
        super().__init__()
        self.embed_tokens = original_decoder.embed_tokens
        self.embed_positions = original_decoder.embed_positions
        self.embed_scale = original_decoder.embed_scale
        self.layerdrop = original_decoder.layerdrop
        self.padding_idx = original_decoder.padding_idx
        self.max_target_positions = original_decoder.max_target_positions
        self.embed_dim = 1024#original_decoder.embed_dim
        
        # Replace layers with dual cross-attention layers
        self.layers = nn.ModuleList([
            DualCrossAttentionLayer(layer, hidden_size) 
            for layer in original_decoder.layers
        ])
        
        if hasattr(original_decoder, 'layer_norm'):
            self.layer_norm = original_decoder.layer_norm
        if hasattr(original_decoder, 'output_projection'):
            self.output_projection = original_decoder.output_projection
    
    def forward(self, input_ids, encoder_hidden_states=None, encoder_attention_mask=None,
                img_hidden_states=None, img_attention_mask=None, output_attentions=False, 
                output_hidden_states=False, **kwargs):
        
        # Embedding
        positions = self.embed_positions(input_ids)
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale
        hidden_states = hidden_states + positions
        
        # Initialize storage for outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_text_cross_attentions = () if output_attentions else None
        all_img_cross_attentions = () if output_attentions else None
        
        # Pass through layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                img_hidden_states=img_hidden_states,
                img_attention_mask=img_attention_mask,
                output_attentions=output_attentions,
                **kwargs
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_text_cross_attentions = all_text_cross_attentions + (layer_outputs[2],)
                all_img_cross_attentions = all_img_cross_attentions + (layer_outputs[3],)
        
        # Final layer norm if exists
        if hasattr(self, 'layer_norm') and self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
            
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Create output object with all the necessary attributes
        class DecoderOutput:
            def __init__(self):
                self.last_hidden_state = hidden_states
                self.hidden_states = all_hidden_states
                self.attentions = all_self_attentions
                self.cross_attentions = all_text_cross_attentions  # Traditional cross attention (text)
                self.text_cross_attentions = all_text_cross_attentions  # Explicit text cross attention
                self.img_cross_attentions = all_img_cross_attentions   # Image cross attention
                
        return DecoderOutput()

class MultimodalIndicTrans2(nn.Module):
    def __init__(self, nmt_model_name, img_feat_dim=768, adapter_type="lora",
                 lora_rank=16, lora_alpha=32, adapter_size=512):
        super().__init__()
        # Load base NMT model
        self.nmt_model = AutoModelForSeq2SeqLM.from_pretrained(
            nmt_model_name, trust_remote_code=True
        )
        
        # Freeze all base model params
        for param in self.nmt_model.parameters():
            param.requires_grad = False

        # Get encoder hidden dim
        if hasattr(self.nmt_model.config, "d_model"):
            fusion_dim = self.nmt_model.config.d_model
        elif hasattr(self.nmt_model.config, "hidden_size"):
            fusion_dim = self.nmt_model.config.hidden_size
        else:
            fusion_dim = 1024
        self.fusion_dim = fusion_dim
        self.adapter_type = adapter_type

        # Image projection & encoder
        self.img_proj = nn.Linear(img_feat_dim, fusion_dim)
        self.img_encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=8, batch_first=True
        )
        self.img_encoder = nn.TransformerEncoder(self.img_encoder_layer, num_layers=1)
        self.img_layernorm = nn.LayerNorm(fusion_dim)

        # Replace decoder with custom dual cross-attention decoder
        self.original_decoder = self.nmt_model.model.decoder
        self.custom_decoder = CustomDecoder(self.original_decoder, fusion_dim)
        self.nmt_model.model.decoder = self.custom_decoder
        
        # Add adapters
        self._add_adapters_to_decoder(adapter_type, lora_rank, lora_alpha, adapter_size)

    def _add_adapters_to_decoder(self, adapter_type, lora_rank, lora_alpha, adapter_size):
        if adapter_type == "lora":
            for layer in self.custom_decoder.layers:
                # Self-attention
                if hasattr(layer.self_attn, 'q_proj'):
                    layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, lora_rank, lora_alpha)
                if hasattr(layer.self_attn, 'v_proj'):
                    layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, lora_rank, lora_alpha)

                # Text cross-attention
                if hasattr(layer.text_cross_attn, 'q_proj'):
                    layer.text_cross_attn.q_proj = LoRALinear(layer.text_cross_attn.q_proj, lora_rank, lora_alpha)
                if hasattr(layer.text_cross_attn, 'v_proj'):
                    layer.text_cross_attn.v_proj = LoRALinear(layer.text_cross_attn.v_proj, lora_rank, lora_alpha)
                    
                # Image cross-attention
                if hasattr(layer.img_cross_attn, 'q_proj'):
                    layer.img_cross_attn.q_proj = LoRALinear(layer.img_cross_attn.q_proj, lora_rank, lora_alpha)
                if hasattr(layer.img_cross_attn, 'v_proj'):
                    layer.img_cross_attn.v_proj = LoRALinear(layer.img_cross_attn.v_proj, lora_rank, lora_alpha)

                # Feed-forward
                if hasattr(layer, 'fc1'):
                    layer.fc1 = LoRALinear(layer.fc1, lora_rank, lora_alpha)
                if hasattr(layer, 'fc2'):
                    layer.fc2 = LoRALinear(layer.fc2, lora_rank, lora_alpha)

        elif adapter_type == "bottleneck":
            self.bottleneck_adapters = nn.ModuleList([
                AdapterLayer(self.fusion_dim, adapter_size) for _ in self.custom_decoder.layers
            ])
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

    def forward(self, input_ids, attention_mask, image_features, img_attention_mask, labels=None, 
                output_attentions=False, output_hidden_states=False):
        """
        input_ids: (B, L) - Source token IDs
        attention_mask: (B, L)
        image_features: (B, V, D) - Visual features (V = number of patches, D = img_feat_dim)
        img_attention_mask: (B, V) - Image attention mask
        labels: (B, T) - Target token IDs (for training)
        """
        B = input_ids.size(0)

        # Encode text
        encoder_outputs = self.nmt_model.get_encoder()(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        text_embeds = encoder_outputs.last_hidden_state  # (B, L, fusion_dim)

        # Encode image
        img_proj = self.img_proj(image_features)         # (B, V, fusion_dim)
        img_key_padding_mask = (img_attention_mask == 0)
        img_encoded = self.img_encoder(img_proj, src_key_padding_mask=img_key_padding_mask)
        img_encoded = self.img_layernorm(img_encoded)    # (B, V, fusion_dim)

        # Convert attention masks for decoder
        img_mask = img_attention_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
        text_mask = attention_mask
        
        # Create extended attention mask (for compatibility)
        extended_attention_mask = torch.cat([img_mask, text_mask], dim=1)
        
        # Create modified encoder outputs (for compatibility)
        concat_embeds = torch.cat([img_encoded, text_embeds], dim=1)
        modified_encoder_outputs = BaseModelOutput(
            last_hidden_state=concat_embeds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions if hasattr(encoder_outputs, 'attentions') else None
        )

        if labels is not None:
            # Training mode
            decoder_input_ids = labels[:, :-1]
            decoder_target = labels[:, 1:]

            # Use custom decoder with dual cross-attention
            decoder_outputs = self.custom_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=text_embeds,
                encoder_attention_mask=text_mask,
                img_hidden_states=img_encoded,
                img_attention_mask=img_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )

            logits = self.nmt_model.lm_head(decoder_outputs.last_hidden_state)  # (B, T, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                decoder_target.reshape(-1),
                ignore_index=self.nmt_model.config.pad_token_id
            )
            return {
                "loss": loss, 
                "logits": logits, 
                "decoder_outputs": decoder_outputs,
                "encoder_outputs": modified_encoder_outputs, 
                "attention_mask": extended_attention_mask
            }
        else:
            # Inference mode - return encoder outputs for generation
            # Store image embeddings for generation
            self._img_encoded = img_encoded
            self._img_mask = img_mask
            
            return {
                "encoder_outputs": modified_encoder_outputs,
                "attention_mask": extended_attention_mask
            }
    
    def generate(self, input_ids, attention_mask, image_features, img_attention_mask, **generation_kwargs):
        """
        Custom beam search generation with dual cross-attention (text + image).
        """

        # ==== Encode text ====
        encoder_outputs = self.nmt_model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False
        )
        text_embeds = encoder_outputs.last_hidden_state  # (B, L, d_model)

        # ==== Encode image ====
        img_proj = self.img_proj(image_features)          # (B, V, d_model)
        img_key_padding_mask = (img_attention_mask == 0)
        img_encoded = self.img_encoder(img_proj, src_key_padding_mask=img_key_padding_mask)
        img_encoded = self.img_layernorm(img_encoded)     # (B, V, d_model)

        # Params
        max_length = generation_kwargs.get("max_length", 128)
        max_new_tokens = generation_kwargs.get("max_new_tokens", None)
        num_beams = generation_kwargs.get("num_beams", 1)
        do_sample = generation_kwargs.get("do_sample", False)
        temperature = generation_kwargs.get("temperature", 1.0)
        pad_token_id = generation_kwargs.get("pad_token_id", self.nmt_model.config.pad_token_id)
        eos_token_id = generation_kwargs.get("eos_token_id", self.nmt_model.config.eos_token_id)
        bos_token_id = generation_kwargs.get("bos_token_id", self.nmt_model.config.bos_token_id)

        batch_size = input_ids.size(0)
        device = input_ids.device
        vocab_size = self.nmt_model.config.vocab_size
        # ==== Expand encoder states for beams ====
        text_embeds = text_embeds.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        img_encoded = img_encoded.repeat_interleave(num_beams, dim=0)
        img_attention_mask = img_attention_mask.repeat_interleave(num_beams, dim=0)

        # ==== Init sequences ====
        generated_ids = torch.full(
            (batch_size * num_beams, max_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        generated_ids[:, 0] = bos_token_id
        seq_lengths = torch.ones(batch_size * num_beams, dtype=torch.long, device=device)

        beam_scores = torch.zeros((batch_size, num_beams), device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # flatten

        finished = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)

        # ==== Beam search loop ====
        past_key_values = None
        for step in range(1, max_length):
            # Current input
            current_ids = generated_ids[:, :step]

            # Decode step
            decoder_outputs = self.custom_decoder(
                input_ids=current_ids,
                attention_mask=None,  # decoder will handle causal mask internally
                encoder_hidden_states=text_embeds,
                encoder_attention_mask=attention_mask,
                img_hidden_states=img_encoded,
                img_attention_mask=img_attention_mask,
                use_cache=False,
            )

            logits = self.nmt_model.lm_head(decoder_outputs.last_hidden_state)  # (B*num_beams, step, V)
            next_token_logits = logits[:, -1, :]  # (B*num_beams, V)

            # Sampling vs greedy
            if do_sample:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                log_probs = torch.log(probs + 1e-12)
            else:
                log_probs = F.log_softmax(next_token_logits, dim=-1)

            # Add beam scores
            next_scores = log_probs + beam_scores[:, None]
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)

            # Select top beams
            next_scores, next_tokens = torch.topk(next_scores, k=num_beams, dim=1)
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # Reorder
            beam_idx = (
                torch.arange(batch_size, device=device)[:, None] * num_beams + next_indices
            ).view(-1)

            generated_ids = generated_ids[beam_idx]
            seq_lengths = seq_lengths[beam_idx]
            finished = finished[beam_idx]

            # Append token
            generated_ids[:, step] = next_tokens.view(-1)
            seq_lengths += (~finished).long()
            beam_scores = next_scores.view(-1)

            # Mark finished
            eos_in_batch = next_tokens.view(-1) == eos_token_id
            finished |= eos_in_batch

            # Mask finished beams so they don't compete further
            beam_scores = beam_scores.masked_fill(finished, -1e9)

            # Early stop if all beams finished
            if finished.view(batch_size, num_beams).all(dim=1).all():
                break

        # ==== Select best beam ====
        beam_scores = beam_scores.view(batch_size, num_beams)
        seq_lengths = seq_lengths.view(batch_size, num_beams)

        # Apply length penalty
        # length_penalty = seq_lengths.float().pow(length_penalty)
        # norm_scores = beam_scores / length_penalty

        best = beam_scores.argmax(dim=1)
        best_sequences = generated_ids.view(batch_size, num_beams, -1)[
            torch.arange(batch_size, device=device), best
        ]

        return best_sequences

    def get_encoder(self):
        """Return encoder for compatibility"""
        return self.nmt_model.get_encoder()

    def get_decoder(self):
        """Return custom decoder"""
        return self.custom_decoder
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters"""
        trainable_params = 0
        total_params = 0
        
        for param in self.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        return trainable_params, total_params

    def print_trainable_parameters(self):
        """Print trainable parameter statistics"""
        trainable_params, total_params = self.get_trainable_parameters()
        print(f"Trainable params: {trainable_params:,} || "
              f"Total params: {total_params:,} || "
              f"Trainable%: {100 * trainable_params / total_params:.2f}%")

# ----- Loading Tokenizers -----
tokenizer_src = AutoTokenizer.from_pretrained("/home/cfiltlab/24m0741/indic_sample/it2_model/models--ai4bharat--indictrans2-en-indic-1b/snapshots/10e65a9951a1e922cd109a95e8aba9357b62144b", trust_remote_code=True)
tokenizer_tgt = AutoTokenizer.from_pretrained("/home/cfiltlab/24m0741/indic_sample/it2_model/models--ai4bharat--indictrans2-en-indic-1b/snapshots/10e65a9951a1e922cd109a95e8aba9357b62144b", trust_remote_code=True)
print("Tokenizers loaded")

# ----- Custom Collate Function for Padding Image Features -----
def collate_fn(batch):
    src_input_ids = torch.stack([item['src_input_ids'] for item in batch])
    src_attention_mask = torch.stack([item['src_attention_mask'] for item in batch])
    tgt_input_ids = torch.stack([item['tgt_input_ids'] for item in batch])
    tgt_attention_mask = torch.stack([item['tgt_attention_mask'] for item in batch])

    features = [item['features'] for item in batch]
    max_image_seq_len = max(f.size(0) for f in features)
    padded_features = []
    img_attention_masks = []
    for f in features:
        seq_len = f.size(0)
        if seq_len < max_image_seq_len:
            padding = torch.ones(max_image_seq_len - seq_len, f.size(1), dtype=f.dtype)
            padded_features.append(torch.cat([f, padding], dim=0))
            img_attention_masks.append(torch.cat([torch.ones(seq_len), torch.zeros(max_image_seq_len - seq_len)]))
        else:
            padded_features.append(f)
            img_attention_masks.append(torch.ones(seq_len))
    padded_features = torch.stack(padded_features)
    img_attention_masks = torch.stack(img_attention_masks).long()

    return {
        'src_input_ids': src_input_ids,
        'src_attention_mask': src_attention_mask,
        'tgt_input_ids': tgt_input_ids,
        'tgt_attention_mask': tgt_attention_mask,
        'features': padded_features,
        'img_attention_mask': img_attention_masks
    }

# ----- Create Datasets and DataLoaders -----

feature_dir = "/home/cfiltlab/24m0741/ViT_features/train_final/"
amb_feature_dir = "/home/cfiltlab/24m0741/ViT_features/test/"
base_dir = "/home/cfiltlab/24m0741/subtitles/"
train_dataset = MultimodalDataset(
    base_dir+"cluster_train.id", base_dir+"cluster_train.en", base_dir+"cluster_train.hi", feature_dir, tokenizer_src, tokenizer_tgt
)
valid_dataset = MultimodalDataset(
    base_dir+"cluster_valid.id", base_dir+"cluster_valid.en", base_dir+"cluster_valid.hi", feature_dir, tokenizer_src, tokenizer_tgt
)
test_dataset = MultimodalDataset(
    base_dir+"cluster_test.id", base_dir+"cluster_test.en", base_dir+"cluster_test.hi", feature_dir, tokenizer_src, tokenizer_tgt
)
amb_test_dataset = MultimodalDataset(
    base_dir+"amb_annote.id", base_dir+"amb_annote.en", base_dir+"amb_annote.hi", amb_feature_dir, tokenizer_src, tokenizer_tgt
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
amb_test_loader = DataLoader(amb_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
# ----- Training Setup -----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the MultimodalIndicTrans2 model
model = MultimodalIndicTrans2(nmt_model_name="/home/cfiltlab/24m0741/indic_sample/it2_model/models--ai4bharat--indictrans2-en-indic-1b/snapshots/10e65a9951a1e922cd109a95e8aba9357b62144b",
        adapter_type="lora",
        adapter_size=512,
        lora_rank=16,
        lora_alpha=32)

model_path = "multimodal_indictrans2_best_lora_r_16_a_32_mod_dual_attn.pth"
print("Model:", model)
# model.load_state_dict(torch.load(model_path))

model.to(device)
print("model loaded")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# ----- Training Loop -----

def train_epoch():
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['src_input_ids'].to(device)
        attention_mask = batch['src_attention_mask'].to(device)
        image_feats = batch['features'].to(device)
        img_attention_mask = batch['img_attention_mask'].to(device)
        labels = batch['tgt_input_ids'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       image_features=image_feats, img_attention_mask=img_attention_mask,labels=labels)

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Train loss: {avg_loss:.4f}")


def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['src_input_ids'].to(device)
            attention_mask = batch['src_attention_mask'].to(device)
            image_feats = batch['features'].to(device)
            img_attention_mask = batch['img_attention_mask'].to(device)
            labels = batch['tgt_input_ids'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        image_features=image_feats, img_attention_mask=img_attention_mask,labels=labels)

            loss = outputs["loss"]
            total_loss += loss.item()

    avg_loss = total_loss / len(valid_loader)
    print(f"Valid loss: {avg_loss:.4f}")
    return avg_loss


# ----- Run Training -----

num_epochs = 50
patience = 11   # stop if no improvement for 3 epochs
best_val_loss = float('inf')
epochs_no_improve = 0

for ep in tqdm(range(num_epochs)):
    print(f"===== Epoch {ep+1} / {num_epochs} =====", flush=True)
    train_epoch()
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        #save_path = "multimodal_indictrans2_best_lora_r_16_a_32_mod_dual_attn.pth"
        save_path = "multimodal_indictrans2_best_dual_attn.pth"
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved best model to {save_path}.")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= patience:
        print(f"⏹ Early stopping triggered after {ep+1} epochs.")
        break


def translate_test_set(model_path, output_path="output_test.hi", test_loader=test_loader, device='cuda'):
    """
    Fixed testing function with proper error handling
    """
    print("===== Running Test-Time Inference =====")

    # Load best model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Aborting inference.")
        return

    model.eval()
    outputs_list = []
    failed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Translating")):
            try:
                input_ids = batch['src_input_ids'].to(device)
                attention_mask = batch['src_attention_mask'].to(device)
                image_feats = batch['features'].to(device)
                img_attention_mask = batch['img_attention_mask'].to(device)

                # Generate translations
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_features=image_feats,
                    img_attention_mask=img_attention_mask,
                    max_length=128,
                    num_beams=2,  
                    do_sample=False,
                    temperature=0.9,
                    pad_token_id=tokenizer_tgt.pad_token_id,
                    eos_token_id=tokenizer_tgt.eos_token_id,
                    bos_token_id=tokenizer_tgt.bos_token_id
                )

                # Decode generated sequences
                decoded = tokenizer_tgt.batch_decode(generated_ids, skip_special_tokens=True)
                outputs_list.extend(decoded)
                
                # Progress logging
                if batch_idx % 1 == 0:
                    print(f"Processed batch {batch_idx}/{len(test_loader)}", flush=True)
                    if decoded:
                        print(f"Sample translation: {decoded[0][:100]}...", flush=True)

            except Exception as e:
                print(f"❌ Failed to process batch {batch_idx}: {e}", flush=True)
                failed_batches += 1
                # Add empty translations for failed batches to maintain alignment
                batch_size = batch['src_input_ids'].size(0)
                outputs_list.extend([''] * batch_size)
                continue

    # Save outputs
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in outputs_list:
                f.write(line.strip() + '\n')
        
        print(f"✅ Saved {len(outputs_list)} translations to {output_path}")
        if failed_batches > 0:
            print(f"⚠️  {failed_batches} batches failed during generation")
            
    except Exception as e:
        print(f"❌ Failed to save outputs: {e}")

    # Save outputs
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in outputs_list:
            f.write(line.strip() + '\n')

    print(f"Saved {len(outputs_list)} translations to {output_path}")

print("Starting test-time inference...")
# ----- Run Test-Time Inference -----
translate_test_set("multimodal_indictrans2_best_dual_attn.pth", output_path="output_test_dual_attn.hi")
# ----- Run Ambiguous Test-Time Inference -----
translate_test_set("/home/cfiltlab/24m0741/multimodal_indictrans2_best_dual_attn.pth", output_path="output_amb_test_annote_dual_attn.hi", test_loader=amb_test_loader)
