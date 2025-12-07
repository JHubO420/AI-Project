import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import random
import time
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import json  # 이미 있다면 추가하지 않아도 됨

# 데이터 전처리 부분 (기존 코드 유지)
data_path = r'C:\Code\Python\data\spa-eng\spa.txt'

# 데이터 불러오기
with open(data_path, encoding='utf-8') as f:
    lines = f.readlines()

# 전처리 (탭으로 구분된 문장쌍 필터링)
pairs = [line.strip().lower().split('\t') for line in lines if '\t' in line]
pairs = [(eng, spa) for items in pairs if len(items) >= 2 for eng, spa in [(items[0], items[1])]]

# 긴 문장 처리를 위한 필터링 (최대 50단어)
pairs = [(eng, spa) for eng, spa in pairs if len(eng.split()) <= 50 and len(spa.split()) <= 50]

# 셔플 후 분할
random.seed(42)
random.shuffle(pairs)
total = len(pairs)
train_size = int(0.8 * total)
val_size = int(0.1 * total)
train_data = pairs[:train_size]
val_data = pairs[train_size:train_size+val_size]
test_data = pairs[train_size+val_size:]

# 단어 사전 구축 함수 정의를 먼저 위치시킵니다.
def build_vocab(sentences, min_freq=2): #
    counter = Counter() #
    for s in sentences: #
        counter.update(s.split()) #
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3} #
    for word, freq in counter.items(): #
        if freq >= min_freq: #
            vocab[word] = len(vocab) #
    return vocab #

# project_claude.py에서, 어휘 사전 구축 후:
src_vocab = build_vocab([eng for eng, _ in train_data])
tgt_vocab = build_vocab([spa for _, spa in train_data])

# 다음 줄을 추가하여 저장:
with open('src_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(src_vocab, f, ensure_ascii=False, indent=4)
with open('tgt_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(tgt_vocab, f, ensure_ascii=False, indent=4)

inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()} # 이 부분은 다시 만들 수 있습니다.

# 문장을 텐서로 변환
def sentence_to_tensor(sentence, vocab):
    tokens = sentence.split()
    indices = [vocab.get(tok, vocab['<unk>']) for tok in tokens]
    return torch.tensor([vocab['<sos>']] + indices + [vocab['<eos>']])

# Dataset 정의
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return sentence_to_tensor(src, self.src_vocab), sentence_to_tensor(tgt, self.tgt_vocab)

# collate function (패딩)
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=src_vocab['<pad>'], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab['<pad>'], batch_first=True)
    return src_batch, tgt_batch

# Attention 메커니즘 구현
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, query, keys, values, mask=None):
        batch_size = keys.size(0)
        seq_len = keys.size(1)
        
        query = query.unsqueeze(1).expand(batch_size, seq_len, self.hidden_size)
        
        scores = self.W_v(torch.tanh(self.W_q(query) + self.W_k(keys)))
        scores = scores.squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), values)
        context = context.squeeze(1)
        
        return context, attention_weights

# Residual Connection이 적용된 인코더
class ResidualEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super(ResidualEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.embedded_proj = nn.Linear(embed_size, hidden_size)
        
    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        lstm_out_projected = self.projection(lstm_out)
        
        # embedded_proj = nn.Linear(embedded.size(-1), self.hidden_size).to(embedded.device)
        embedded_residual = self.embedded_proj(embedded)
        
        output = self.layer_norm(lstm_out_projected + embedded_residual)
        
        batch_size = hidden.size(1)
        hidden = hidden.view(self.num_layers, 2, x.size(0), self.hidden_size)
        hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]

        cell = cell.view(self.num_layers, 2, x.size(0), self.hidden_size)
        cell = cell[:, 0, :, :] + cell[:, 1, :, :]

        return output, (hidden, cell)

# Attention과 Residual Connection이 적용된 디코더
class ResidualAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        super(ResidualAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        
        self.input_projection = nn.Linear(embed_size + hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden, cell, encoder_outputs, encoder_mask=None):
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)
        
        query = hidden[-1]
        context, attention_weights = self.attention(query, encoder_outputs, encoder_outputs, encoder_mask)
        
        lstm_input = torch.cat([embedded.squeeze(1), context], dim=1).unsqueeze(1)
        
        lstm_out, (new_hidden, new_cell) = self.lstm(lstm_input, (hidden, cell))
        
        residual_input = self.input_projection(lstm_input)
        output = self.layer_norm(lstm_out + residual_input)
        
        output = self.output_projection(output.squeeze(1))
        
        return output, new_hidden, new_cell, attention_weights

# 전체 Seq2Seq 모델
class ResidualAttentionSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.1):
        super(ResidualAttentionSeq2Seq, self).__init__()
        self.encoder = ResidualEncoder(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = ResidualAttentionDecoder(tgt_vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.hidden_size = hidden_size
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.vocab_size
        
        encoder_outputs, (hidden, cell) = self.encoder(src)
        encoder_mask = (src != 0).float()
        
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        input_token = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            output, hidden, cell, attention_weights = self.decoder(
                input_token, hidden, cell, encoder_outputs, encoder_mask
            )
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs

# BLEU 스코어 계산 함수
def calculate_bleu(predicted, target, vocab):
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    
    for pred, tar in zip(predicted, target):
        pred_words = [vocab.get(idx.item(), '<unk>') for idx in pred if idx.item() not in [0, 1, 2]]
        tar_words = [vocab.get(idx.item(), '<unk>') for idx in tar if idx.item() not in [0, 1, 2]]
        
        if len(pred_words) > 0 and len(tar_words) > 0:
            score = sentence_bleu([tar_words], pred_words, smoothing_function=smoothie)
            bleu_scores.append(score)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0

# 모델 저장 클래스
class ModelCheckpoint:
    def __init__(self, save_dir='best_models', top_k=5):
        self.save_dir = save_dir
        self.top_k = top_k
        self.best_models = []  # (bleu_score, epoch, model_path, metadata) 형태로 저장
        
        # 저장 디렉토리 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def update(self, model, bleu_score, epoch, val_loss):
        model_info = {
            'bleu_score': bleu_score,
            'epoch': epoch,
            'val_loss': val_loss,
            'model_path': f"{self.save_dir}/model_epoch_{epoch}_bleu_{bleu_score:.4f}.pth",
            'metadata_path': f"{self.save_dir}/model_epoch_{epoch}_bleu_{bleu_score:.4f}_meta.json"
        }
        
        # 모델 가중치만 저장 (안전한 방식)
        torch.save(model.state_dict(), model_info['model_path'])
        
        # 메타데이터는 별도 JSON 파일로 저장
        import json
        metadata = {
            'bleu_score': bleu_score,
            'epoch': epoch,
            'val_loss': val_loss
        }
        with open(model_info['metadata_path'], 'w') as f:
            json.dump(metadata, f)
        
        # best_models 리스트에 추가
        self.best_models.append(model_info)
        
        # BLEU 스코어 기준으로 정렬 (내림차순)
        self.best_models.sort(key=lambda x: x['bleu_score'], reverse=True)
        
        # top_k개만 유지하고 나머지는 삭제
        if len(self.best_models) > self.top_k:
            removed_model = self.best_models.pop()  # 마지막(가장 낮은 점수) 제거
            if os.path.exists(removed_model['model_path']):
                os.remove(removed_model['model_path'])
            if os.path.exists(removed_model['metadata_path']):
                os.remove(removed_model['metadata_path'])
    
    def get_best_models(self):
        return self.best_models
    
    def load_model_with_metadata(self, model, model_path):
        """모델과 메타데이터를 함께 로드하는 헬퍼 함수"""
        # 모델 가중치 로드
        model.load_state_dict(torch.load(model_path, weights_only=True))
        
        # 메타데이터 로드
        metadata_path = model_path.replace('.pth', '_meta.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            return None
    
    def save_summary(self):
        summary_path = f"{self.save_dir}/models_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== TOP 5 BEST MODELS ===\n\n")
            for i, model_info in enumerate(self.best_models, 1):
                f.write(f"Rank {i}:\n")
                f.write(f"  - Epoch: {model_info['epoch']}\n")
                f.write(f"  - BLEU Score: {model_info['bleu_score']:.4f}\n")
                f.write(f"  - Validation Loss: {model_info['val_loss']:.4f}\n")
                f.write(f"  - Model Path: {model_info['model_path']}\n")
                f.write(f"  - Metadata Path: {model_info['metadata_path']}\n")
                f.write("-" * 50 + "\n")

# 훈련 함수 (모델 저장 기능 추가)
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.7)
    
    # 모델 체크포인트 객체 생성
    checkpoint = ModelCheckpoint()
    
    train_losses = []
    val_losses = []
    val_bleu_scores = []
    best_bleu = 0.0
    patience_counter = 0
    early_stopping_patience = 3
    
    print(f"Training on {device}")
    
    for epoch in range(num_epochs):
        # 훈련
        model.train()
        total_train_loss = 0
        train_start_time = time.time()
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt)
            
            output = output[:, 1:].reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_time = time.time() - train_start_time
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 검증
        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                
                output = model(src, tgt, teacher_forcing_ratio=0)
                
                output_loss = output[:, 1:].reshape(-1, output.size(-1))
                tgt_loss = tgt[:, 1:].reshape(-1)
                loss = criterion(output_loss, tgt_loss)
                total_val_loss += loss.item()
                
                predictions = output.argmax(dim=-1)
                all_predictions.extend(predictions.cpu())
                all_targets.extend(tgt.cpu())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        bleu_score = calculate_bleu(all_predictions, all_targets, inv_tgt_vocab)
        val_bleu_scores.append(bleu_score)
        
        # 모델 체크포인트 업데이트
        checkpoint.update(model, bleu_score, epoch+1, avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'  BLEU Score: {bleu_score:.4f}, Train Time: {train_time:.2f}s')
        print('-' * 50)
        
        # Early stopping
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break
    
    # 최종 요약 저장
    checkpoint.save_summary()
    
    print("\n🏆 TOP 5 BEST MODELS:")
    print("=" * 60)
    for i, model_info in enumerate(checkpoint.get_best_models(), 1):
        print(f"Rank {i}: Epoch {model_info['epoch']}, BLEU: {model_info['bleu_score']:.4f}, Val Loss: {model_info['val_loss']:.4f}")
        print(f"         Path: {model_info['model_path']}")
    
    return train_losses, val_losses, val_bleu_scores, checkpoint

# 번역 함수 (개선된 형식)
def translate(model, sentence, src_vocab, tgt_vocab, inv_tgt_vocab, device, max_length=50):
    model.eval()
    
    tokens = sentence.lower().split()
    indices = [src_vocab.get(tok, src_vocab['<unk>']) for tok in tokens]
    src_tensor = torch.tensor([src_vocab['<sos>']] + indices + [src_vocab['<eos>']]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor)
        encoder_mask = (src_tensor != 0).float()
        
        input_token = torch.tensor([[tgt_vocab['<sos>']]]).to(device)
        translated = []
        
        for _ in range(max_length):
            output, hidden, cell, attention_weights = model.decoder(
                input_token, hidden, cell, encoder_outputs, encoder_mask
            )
            
            predicted_id = output.argmax(1).item()
            
            if predicted_id == tgt_vocab['<eos>']:
                break
                
            translated.append(inv_tgt_vocab.get(predicted_id, '<unk>'))
            input_token = torch.tensor([[predicted_id]]).to(device)
    
    return ' '.join(translated)

# 테스트 데이터로 번역 평가 (새로운 형식)
def evaluate_on_test_data(model, test_loader, src_vocab, tgt_vocab, inv_tgt_vocab, device, num_samples=10):
    model.eval()
    results = []
    
    # 테스트 데이터에서 샘플 추출
    test_samples = []
    for src_batch, tgt_batch in test_loader:
        for i in range(min(src_batch.size(0), num_samples - len(test_samples))):
            src_tensor = src_batch[i]
            tgt_tensor = tgt_batch[i]
            
            # 텐서를 문장으로 변환
            src_sentence = ' '.join([list(src_vocab.keys())[list(src_vocab.values()).index(idx.item())] 
                                   for idx in src_tensor if idx.item() not in [0, 1, 2]])
            tgt_sentence = ' '.join([inv_tgt_vocab.get(idx.item(), '<unk>') 
                                   for idx in tgt_tensor if idx.item() not in [0, 1, 2]])
            
            test_samples.append((src_sentence, tgt_sentence))
            
            if len(test_samples) >= num_samples:
                break
        if len(test_samples) >= num_samples:
            break
    
    print("\n📝 번역 결과 (형식: 입력 → 정답 → 번역):")
    print("=" * 80)
    
    for i, (input_sentence, target_sentence) in enumerate(test_samples, 1):
        translation = translate(model, input_sentence, src_vocab, tgt_vocab, inv_tgt_vocab, device)
        
        print(f"Sample {i}:")
        print(f"입력 문장: {input_sentence}")
        print(f"정답 문장: {target_sentence}")
        print(f"번역 문장: {translation}")
        print("-" * 40)
        
        results.append({
            'input': input_sentence,
            'target': target_sentence,
            'translation': translation
        })
    
    return results

# 모델 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResidualAttentionSeq2Seq(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    embed_size=256,
    hidden_size=512,
    num_layers=2,
    dropout=0.1
)

# DataLoader 생성
train_loader = DataLoader(TranslationDataset(train_data, src_vocab, tgt_vocab), 
                         batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(TranslationDataset(val_data, src_vocab, tgt_vocab), 
                       batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(TranslationDataset(test_data, src_vocab, tgt_vocab), 
                        batch_size=64, shuffle=False, collate_fn=collate_fn)

print(f"학습 샘플: {len(train_loader.dataset)}, 검증 샘플: {len(val_loader.dataset)}, 테스트 샘플: {len(test_loader.dataset)}")
print(f"소스 어휘 크기: {len(src_vocab)}, 타겟 어휘 크기: {len(tgt_vocab)}")
print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# 훈련 실행
if __name__ == "__main__":
    print("모델 훈련 시작...")
    print(f"예상 소요 시간: {'1.5-2시간 (GPU)' if torch.cuda.is_available() else '10-25시간 (CPU)'}")
    
    QUICK_TEST = False
    epochs = 5 if QUICK_TEST else 10
    
    if QUICK_TEST:
        print("⚡ 빠른 테스트 모드 - 5 epochs로 단축")
        train_data_subset = train_data[:5000]
        val_data_subset = val_data[:1000]
        train_loader = DataLoader(TranslationDataset(train_data_subset, src_vocab, tgt_vocab), 
                                 batch_size=64, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(TranslationDataset(val_data_subset, src_vocab, tgt_vocab), 
                               batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    start_time = time.time()
    train_losses, val_losses, val_bleu_scores, checkpoint = train_model(model, train_loader, val_loader, num_epochs=epochs)
    total_time = time.time() - start_time
    
    print(f"\n🎉 총 훈련 시간: {total_time/3600:.2f}시간 ({total_time/60:.1f}분)")
    
    # 학습 결과 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(val_bleu_scores)
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('Validation BLEU Score')
    
    plt.subplot(1, 3, 3)
    plt.plot(range(len(train_losses)), [f"Epoch {i+1}" for i in range(len(train_losses))])
    plt.xticks(rotation=45)
    plt.title('Training Progress')
    
    plt.tight_layout()
    plt.show()
    
    # 최고 성능 모델로 번역 테스트 (수정된 부분)
    best_model_info = checkpoint.get_best_models()[0]
    print(f"\n🏆 최고 성능 모델 로드 (Epoch {best_model_info['epoch']}, BLEU: {best_model_info['bleu_score']:.4f})")
    
    # 새로운 안전한 방식으로 모델 로드
    metadata = checkpoint.load_model_with_metadata(model, best_model_info['model_path'])
    if metadata:
        print(f"로드된 모델 정보 - Epoch: {metadata['epoch']}, BLEU: {metadata['bleu_score']:.4f}, Val Loss: {metadata['val_loss']:.4f}")
    
    # 테스트 데이터로 평가
    test_results = evaluate_on_test_data(model, test_loader, src_vocab, tgt_vocab, inv_tgt_vocab, device, num_samples=15)
    
    # 추가 번역 테스트
    custom_test_sentences = [
        "i love you",
        "how are you today",
        "the weather is beautiful",
        "i want to learn spanish",
        "this is a very long sentence that tests the model capability"
    ]
    
    print("\n🔤 추가 번역 테스트:")
    print("=" * 80)
    for i, sentence in enumerate(custom_test_sentences, 1):
        translation = translate(model, sentence, src_vocab, tgt_vocab, inv_tgt_vocab, device)
        print(f"Test {i}:")
        print(f"입력 문장: {sentence}")
        print(f"번역 문장: {translation}")
        print("-" * 40)
