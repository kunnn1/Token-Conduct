import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    
    model.eval()
  
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

    token_embeddings = torch.stack(hidden_states, dim=0)

    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    token_embeddings = token_embeddings.permute(1,0,2)

    token_vecs_sum = []
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)

    sentence_embedding = torch.mean(torch.stack(token_vecs_sum), dim=0)

    return sentence_embedding.numpy()
