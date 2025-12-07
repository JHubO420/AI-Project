import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re # For parsing the summary file

# --- Model Architecture (Copied from project_claude.py) ---
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, 1, bias=False) # Corrected: W_v projects to 1 for attention scores

    def forward(self, query, keys, values, mask=None):
        batch_size = keys.size(0)
        seq_len = keys.size(1)

        query = query.unsqueeze(1).expand(batch_size, seq_len, self.hidden_size)

        scores = self.W_v(torch.tanh(self.W_q(query) + self.W_k(keys)))
        scores = scores.squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), values) # Values are typically encoder_outputs
        context = context.squeeze(1)

        return context, attention_weights

class ResidualEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super(ResidualEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        self.projection = nn.Linear(hidden_size * 2, hidden_size) # For bidirectional output
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        # 이 부분이 중요합니다!
        self.embedded_proj = nn.Linear(embed_size, hidden_size)  # For residual connection matching dimensions


    def forward(self, x, lengths=None): # lengths not used in the provided project_claude.py forward
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        lstm_out, (hidden, cell) = self.lstm(embedded)
        lstm_out_projected = self.projection(lstm_out) # Project bidirectional LSTM output to hidden_size

        # Residual connection: project embedding to match LSTM output dimension
        embedded_residual = self.embedded_proj(embedded) # 여기서 self.embedded_proj 사용

        output = self.layer_norm(lstm_out_projected + embedded_residual)

        # Combine bidirectional hidden and cell states
        # (num_layers * num_directions, batch, hidden_size) -> (num_layers, batch, hidden_size * num_directions)
        # Then sum or average directions. project_claude.py sums them.
        hidden = hidden.view(self.num_layers, 2, x.size(0), self.hidden_size)
        hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]

        cell = cell.view(self.num_layers, 2, x.size(0), self.hidden_size)
        cell = cell[:, 0, :, :] + cell[:, 1, :, :]

        return output, (hidden, cell)

class ResidualAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super(ResidualAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size)
        # Decoder LSTM input: embedding_dim + context_vector_dim (which is hidden_size from encoder)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)

        # For residual connection: projects LSTM input to hidden_size to match LSTM output
        self.input_projection = nn.Linear(embed_size + hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size) # Projects LSTM output to vocab size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, encoder_mask=None):
        # input_token shape: (batch_size, 1) for single time step
        embedded = self.embedding(input_token) # (batch_size, 1, embed_size)
        embedded = self.dropout(embedded)

        # Query for attention is the previous hidden state (last layer)
        # hidden shape: (num_layers, batch_size, hidden_size)
        query = hidden[-1] # Take the hidden state of the last layer

        context, attention_weights = self.attention(query, encoder_outputs, encoder_outputs, encoder_mask)
        # context shape: (batch_size, hidden_size)

        # Concatenate embedded input token and context vector
        # embedded is (batch_size, 1, embed_size), context is (batch_size, hidden_size)
        # Squeeze embedded to (batch_size, embed_size) before cat
        lstm_input = torch.cat([embedded.squeeze(1), context], dim=1).unsqueeze(1)
        # lstm_input shape: (batch_size, 1, embed_size + hidden_size)

        lstm_out, (new_hidden, new_cell) = self.lstm(lstm_input, (hidden, cell))
        # lstm_out shape: (batch_size, 1, hidden_size)

        # Residual connection
        residual_input_projected = self.input_projection(lstm_input.squeeze(1)) # Project to hidden_size
        output_sum = lstm_out.squeeze(1) + residual_input_projected
        output_normalized = self.layer_norm(output_sum)

        output_logits = self.output_projection(output_normalized) # (batch_size, vocab_size)

        return output_logits, new_hidden, new_cell, attention_weights

class ResidualAttentionSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.1):
        super(ResidualAttentionSeq2Seq, self).__init__()
        self.encoder = ResidualEncoder(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = ResidualAttentionDecoder(tgt_vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size # Store for convenience

    def forward(self, src, tgt, teacher_forcing_ratio=0.5): # Only for training, not used in translate
        # This forward method is typically used for training.
        # For inference, we usually call encoder and then decode step-by-step.
        # The translate() function below handles the step-by-step decoding.
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        encoder_outputs, (hidden, cell) = self.encoder(src)
        encoder_mask = (src != 0).float() # Assuming 0 is padding_idx

        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size).to(src.device)
        input_token = tgt[:, 0].unsqueeze(1) # <sos> token

        for t in range(1, tgt_len):
            output, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs, encoder_mask
            )
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs
# --- End of Model Architecture ---

# --- Vocabulary and Model Loading Utilities ---
def load_vocab(path):
    """Loads a vocabulary from a JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {path}")
        print("Please ensure you have saved src_vocab.json and tgt_vocab.json from project_claude.py.")
        exit()

def find_best_model_path(summary_dir="best_models", summary_filename="models_summary.txt"):
    """Parses the summary file to find the path of the best model (Rank 1)."""
    summary_path = os.path.join(summary_dir, summary_filename)
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "Rank 1:" in line:
                # Search for the model path in the lines following "Rank 1:"
                for subsequent_line_idx in range(i + 1, min(i + 6, len(lines))):
                    model_path_match = re.search(r"- Model Path:\s*(.*)", lines[subsequent_line_idx])
                    if model_path_match:
                        # Ensure the path is treated as relative to the script or an absolute path
                        path = model_path_match.group(1).strip()
                        return path
    except FileNotFoundError:
        print(f"Info: Summary file {summary_path} not found. Will use default model path if specified, or ask.")
    except Exception as e:
        print(f"Error parsing summary file: {e}")
    return None

# --- Translation Function (Copied and adapted from project_claude.py) ---
def translate(model, sentence, src_vocab, tgt_vocab, inv_tgt_vocab, device, max_length=50):
    model.eval()

    # Tokenize sentence
    tokens = sentence.lower().split() # Basic whitespace tokenizer
    indices = [src_vocab.get(tok, src_vocab['<unk>']) for tok in tokens]
    src_tensor = torch.tensor([src_vocab['<sos>']] + indices + [src_vocab['<eos>']]).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor)
        encoder_mask = (src_tensor != src_vocab['<pad>']).float().to(device) # Create mask based on pad token

        input_token = torch.tensor([[tgt_vocab['<sos>']]], device=device) # Start with <sos>
        translated_ids = []

        for _ in range(max_length):
            output_logits, hidden, cell, attention_weights = model.decoder(
                input_token, hidden, cell, encoder_outputs, encoder_mask
            )

            predicted_id = output_logits.argmax(1).item()

            if predicted_id == tgt_vocab['<eos>']:
                break

            translated_ids.append(predicted_id)
            input_token = torch.tensor([[predicted_id]], device=device) # Use predicted token as next input

    translated_words = [inv_tgt_vocab.get(str(idx), inv_tgt_vocab.get(idx, '<unk>')) for idx in translated_ids] # Handle int vs str keys
    return ' '.join(translated_words)

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    SRC_VOCAB_PATH = 'src_vocab.json'
    TGT_VOCAB_PATH = 'tgt_vocab.json'
    MODEL_DIR = 'best_models' # Directory where models (and summary) are saved
    # Attempt to find the best model automatically, otherwise fallback or ask
    # You can hardcode a specific model path here if you prefer:
    # DEFAULT_MODEL_PATH = 'best_models/model_epoch_8_bleu_0.7234.pth'
    DEFAULT_MODEL_PATH = None


    # Model hyperparameters (must match the trained model from project_claude.py)
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.1

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocabularies
    print(f"Loading source vocabulary from {SRC_VOCAB_PATH}...")
    src_vocab = load_vocab(SRC_VOCAB_PATH)
    print(f"Loading target vocabulary from {TGT_VOCAB_PATH}...")
    tgt_vocab = load_vocab(TGT_VOCAB_PATH)

    # Create inverse target vocabulary (ID to word)
    # JSON keys are strings, so ensure lookup handles this if IDs were saved as int
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    # If vocab values (IDs) are integers, ensure they are correctly mapped back
    # This handles if tgt_vocab was {'word': 0} and inv_tgt_vocab needs {0: 'word'}
    # The translate function handles string conversion for lookup if needed.


    print(f"출처 어휘 크기: {len(src_vocab)}")
    print(f"목표 어휘 크기: {len(tgt_vocab)}")


    # Initialize model
    model = ResidualAttentionSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    # Load the best model's state_dict
    model_path_to_load = DEFAULT_MODEL_PATH
    if not model_path_to_load:
        print(f"요약에서 최적의 모델 찾기 시도 '{MODEL_DIR}'...")
        model_path_to_load = find_best_model_path(summary_dir=MODEL_DIR)

    if not model_path_to_load:
        print(f"자동으로 최적의 모델을 찾을 수 없습니다.")
        model_path_to_load = input(f"모델 파일의 경로를 입력해 주세요 (e.g., {MODEL_DIR}/model_epoch_X_bleu_Y.pth): ")

    if not os.path.exists(model_path_to_load):
        print(f"오류: 모델 파일을 찾을 수 없습니다 {model_path_to_load}")
        print("모델 경로가 정확하고 파일이 존재하는지 확인해 주세요.")
        exit()

    print(f"모델 상태 로드 위치: {model_path_to_load}")
    try:
        # project_claude.py saves only the state_dict
        model.load_state_dict(torch.load(model_path_to_load, map_location=device))
    except RuntimeError as e:
        print(f"모델 state_dict 로드 중 오류 발생: {e}")
        print("이 스크립트의 모델 아키텍처가 저장된 모델과 일치하지 않을 경우 이런 일이 발생할 수 있습니다,")
        print("또는 .pth 파일에 state_dict가 직접 포함되어 있지 않은 경우(예: 체크포인트 사전).")
        print("project.py 스크립트는 'torch.save(model.state_dict(), PATH'로 저장해야 합니다'.")
        exit()

    model.eval() # Set model to evaluation mode

    print("\n모델이 성공적으로 로드되었습니다. 번역 준비 완료.")
    print("스페인어로 번역할 문장을 영어로 입력합니다.")
    print("'#'을 입력하고 Enter 키를 눌러 종료합니다.")

    while True:
        input_sentence = input("\n영어: ")
        if input_sentence.strip() == "#":
            print("번역가 종료 중.")
            break

        if not input_sentence.strip():
            continue

        try:
            translation = translate(model, input_sentence, src_vocab, tgt_vocab, inv_tgt_vocab, device)
            print(f"스페인어: {translation}")
        except Exception as e:
            print(f"번역 중 오류가 발생했습니다: {e}")
            # You might want to add more robust error handling here
