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
from transformers import GenerationConfig


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



# ----- Model -----


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


# ===== Bottleneck Adapter =====

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


class MultimodalIndicTrans2(nn.Module):
    def __init__(self, nmt_model_name, img_feat_dim=768, adapter_type="lora",lora_rank=16, lora_alpha=32, adapter_size=512,
                 contrastive_weight=1.0, contrastive_temperature=0.005,use_dual_ctr=False):
        super().__init__()
        # Load base NMT model
        self.nmt_model = AutoModelForSeq2SeqLM.from_pretrained(
            nmt_model_name, trust_remote_code=True
        )
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

        # Contrastive settings
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.use_dual_ctr = use_dual_ctr

        # Image projection & encoder
        self.img_proj = nn.Linear(img_feat_dim, fusion_dim)
        self.img_encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=8, batch_first=True
        )
        self.img_encoder = nn.TransformerEncoder(self.img_encoder_layer, num_layers=1)
        self.img_layernorm = nn.LayerNorm(fusion_dim)

        # Adapters
        self.adapter_type = adapter_type
        self._add_adapters_to_decoder(adapter_type, lora_rank, lora_alpha, adapter_size)
        self.text_proj_head = nn.Sequential(nn.Linear(fusion_dim, fusion_dim),nn.ReLU(),nn.Linear(fusion_dim, fusion_dim))
        self.img_proj_head = nn.Sequential(nn.Linear(fusion_dim, fusion_dim),nn.ReLU(),nn.Linear(fusion_dim, fusion_dim))

    def _add_adapters_to_decoder(self, adapter_type, lora_rank, lora_alpha, adapter_size):
        decoder = self.nmt_model.model.decoder

        if adapter_type == "lora":
            for layer in decoder.layers:
                # Self-attention
                if hasattr(layer.self_attn, 'q_proj'):
                    layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, lora_rank, lora_alpha)
                if hasattr(layer.self_attn, 'v_proj'):
                    layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, lora_rank, lora_alpha)

                # Cross-attention
                if hasattr(layer, 'encoder_attn') and layer.encoder_attn is not None:
                    if hasattr(layer.encoder_attn, 'q_proj'):
                        layer.encoder_attn.q_proj = LoRALinear(layer.encoder_attn.q_proj, lora_rank, lora_alpha)
                    if hasattr(layer.encoder_attn, 'v_proj'):
                        layer.encoder_attn.v_proj = LoRALinear(layer.encoder_attn.v_proj, lora_rank, lora_alpha)

                # Feed-forward
                if hasattr(layer, 'fc1'):
                    layer.fc1 = LoRALinear(layer.fc1, lora_rank, lora_alpha)
                if hasattr(layer, 'fc2'):
                    layer.fc2 = LoRALinear(layer.fc2, lora_rank, lora_alpha)

        elif adapter_type == "bottleneck":
            self.bottleneck_adapters = nn.ModuleList([
                AdapterLayer(self.fusion_dim, adapter_size) for _ in decoder.layers
            ])
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

    def forward(self, input_ids, attention_mask, image_features, img_attention_mask, labels=None):
        B = input_ids.size(0)

        # Encode text
        encoder_outputs = self.nmt_model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = encoder_outputs.last_hidden_state  # (B, L, D)

        # Encode image
        img_proj = self.img_proj(image_features)         
        img_key_padding_mask = (img_attention_mask == 0)
        img_encoded = self.img_encoder(img_proj, src_key_padding_mask=img_key_padding_mask) # (B, V, D)
        img_encoded = self.img_layernorm(img_encoded)

        # Fuse image + text for decoder
        concat_embeds = torch.cat([img_encoded, text_embeds], dim=1)
        modified_encoder_outputs = BaseModelOutput(
            last_hidden_state=concat_embeds,
            hidden_states=encoder_outputs.hidden_states
        )
        img_mask = img_attention_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
        extended_attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        # Compute contrastive features
        # mean pool image + text
        text_repr = self.text_proj_head(text_embeds.mean(dim=1))   # (B, D)
        img_repr  = self.img_proj_head(img_encoded.mean(dim=1))    # (B, D)


        # Normalize (cosine sim works better with L2 norm)
        text_repr = F.normalize(text_repr, dim=-1)
        img_repr = F.normalize(img_repr, dim=-1)

        # Contrastive loss
        contrastive_loss = self.compute_contrastive_loss(text_repr, img_repr)

        if labels is not None:
            # Training mode
            decoder_input_ids = labels[:, :-1]
            decoder_target = labels[:, 1:]

            decoder_outputs = self.nmt_model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=concat_embeds,
                encoder_attention_mask=extended_attention_mask
            )

            logits = self.nmt_model.lm_head(decoder_outputs.last_hidden_state)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                decoder_target.reshape(-1),
                ignore_index=self.nmt_model.config.pad_token_id
            )

            total_loss = ce_loss + self.contrastive_weight * contrastive_loss
            return {"loss": total_loss, "ce_loss": ce_loss, "contrastive_loss": contrastive_loss, "logits": logits}
        else:
            return {"encoder_outputs": modified_encoder_outputs, "attention_mask": extended_attention_mask}

    def compute_contrastive_loss(self, text_repr, img_repr):
        """
        InfoNCE contrastive loss
        """
        batch_size, hidden_size = text_repr.size()
        logits = text_repr @ img_repr.t()   # (B, B) similarity
        logits /= self.contrastive_temperature

        if self.use_dual_ctr:
            loss_text = F.cross_entropy(logits, torch.arange(batch_size, device=logits.device))
            loss_img = F.cross_entropy(logits.t(), torch.arange(batch_size, device=logits.device))
            loss = (loss_text + loss_img) / 2
        else:
            # text→image only
            loss = F.cross_entropy(logits, torch.arange(batch_size, device=logits.device))
        return loss

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

feature_dir = "/home/cfiltlab/24m0741/ViT_features/video_features_clip/"
amb_feature_dir = "/home/cfiltlab/24m0741/ViT_features/test/"
base_dir = "/home/cfiltlab/24m0741/subtitles/"
typ = "cluster_clip_" #"cluster_"  #""
train_dataset = MultimodalDataset(
    base_dir+typ+"train.id", base_dir+typ+"train.en", base_dir+typ+"train.hi", feature_dir, tokenizer_src, tokenizer_tgt
)
valid_dataset = MultimodalDataset(
    base_dir+typ+"valid.id", base_dir+typ+"valid.en", base_dir+typ+"valid.hi", feature_dir, tokenizer_src, tokenizer_tgt
)
test_dataset = MultimodalDataset(
    base_dir+typ+"test.id", base_dir+typ+"test.en", base_dir+typ+"test.hi", feature_dir, tokenizer_src, tokenizer_tgt
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
        img_feat_dim = 512,
        adapter_size=512,
        lora_rank=16,
        lora_alpha=32)

# model_path = "multimodal_indictrans2_best_lora_r_16_a_32_contra_loss_lam_1_clip.pth"
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
        save_path = "multimodal_indictrans2_best_lora_r_16_a_32_contra_loss_lam_1_cluster_clip.pth"
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved best model to {save_path}.")
    else:
        if ep < 19:
            continue
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

    if ep > 15 and epochs_no_improve >= patience:
        print(f"⏹ Early stopping triggered after {ep+1} epochs.")
        break


def translate_test_set(model_path, output_path="output_test.hi", test_loader=test_loader):
    print("===== Running Test-Time Inference =====")

    # Load best model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    outputs_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['src_input_ids'].to(device)
            attention_mask = batch['src_attention_mask'].to(device)
            image_feats = batch['features'].to(device)
            img_attention_mask = batch['img_attention_mask'].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        image_features=image_feats, img_attention_mask=img_attention_mask)


            # Generation config (adjust as needed)
            generation_config = GenerationConfig(
                max_new_tokens=128,
                num_beams=2,
                early_stopping=True
            )

            generated_ids = model.nmt_model.generate(
                input_ids=None,
                attention_mask=out["attention_mask"],
                encoder_outputs=out["encoder_outputs"],
                generation_config=generation_config,
                use_cache=False
            )

            decoded = tokenizer_tgt.batch_decode(generated_ids, skip_special_tokens=True)
            outputs_list.extend(decoded)

    # Save to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in outputs_list:
            f.write(line.strip() + '\n')

    print(f"Saved {len(outputs_list)} translations to {output_path}")

# ----- Run Test-Time Inference -----
translate_test_set("multimodal_indictrans2_best_lora_r_16_a_32_contra_loss_lam_1_cluster_clip.pth", output_path="output_amb_test_lora_r_16_a_32_contra_loss_lam_1_cluster_clip.hi")
# ----- Run Ambiguous Test-Time Inference -----
translate_test_set("/home/cfiltlab/24m0741/multimodal_indictrans2_best_lora_r_16_a_32_contra_loss_lam_1_cluster_clip.pth", output_path="output_amb_test_lora_r_16_a_32_contra_loss_annote_cluster_clip.hi", test_loader=amb_test_loader)
