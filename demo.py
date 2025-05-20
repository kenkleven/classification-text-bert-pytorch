import torch
import gradio as gr
from transformers import AutoTokenizer, BertModel
from bertclassifiactiontext import NewsClassifier

model_path = "saved_model_bert"
num_classes = 2  

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = NewsClassifier(num_classes=num_classes)
model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=torch.device('cpu')))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

labels_map = {
    0: "Politics News",
    1: "World News"
}

def predict(title, content):
    combined_text = f"{title} [SEP] {content}"
    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probs = torch.softmax(logits, dim=1).squeeze()
    predicted_class = torch.argmax(probs).item()
    predicted_label = labels_map[predicted_class]
    confidence = probs[predicted_class].item()

    return f"{predicted_label} — confiance : {confidence:.2%}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Titre de l'article", placeholder="Ex: Inflation en hausse"),
        gr.Textbox(label="Contenu de l'article", lines=5, placeholder="Ex: Le taux d'inflation..."),
    ],
    outputs=gr.Textbox(label="Prédiction"),
    title="Démo BERT : Classification d'Articles",
    description="Entrez un titre et un texte pour prédire s'il s'agit de 'Politics News' ou 'World News'."
)

if __name__ == "__main__":
    demo.launch(share=True) 
