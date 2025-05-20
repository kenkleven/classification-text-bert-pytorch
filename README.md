
# 📄 Classification de textes avec BERT (Politics vs World News)

## 🧠 Objectif

Ce projet a pour but de classer automatiquement des articles de presse en deux catégories :
- `Politics News` (Classe 0)
- `World News` (Classe 1)

Le modèle utilise un BERT pré-entraîné, affiné (`fine-tuned`) sur un jeu de données d'articles labellisés, puis intégré dans une application interactive via **Gradio**.

---

## 📁 Structure du projet

```
.
├── bertclassifiactiontext.ipynb   # Entrainement et évaluation du modèle NewsClassifier
├── bertclassifiactiontext.py   # Définition du modèle NewsClassifier
├── demo.py                     # Interface Gradio pour la prédiction
├── saved_model_bert/          # Modèle entraîné et tokenizer sauvegardés
│   ├── config.json
│   ├── model.pt
│   └── vocab.txt
└── requirements.txt                   # Dépendances du projet
└── README.md                   # Fichier d’explication (ce fichier)
```

---

## 🏗️ Étapes de réalisation

### 1. 🔨 Entraînement du modèle (`bertclassifiactiontext.ipynb`)
Le modèle est basé sur la classe `BertModel` de Hugging Face. Il comprend :

- Une couche BERT pré-entraînée
- Une couche `Dropout`
- Une couche linéaire pour la classification

```python
class NewsClassifier(nn.Module):
    def __init__(self, num_classes):
        super(NewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config.model)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(self.drop(pooled_output))
```

---

### 2. ✅ Chargement et prédiction (`demo.py`)

- Chargement du tokenizer et du modèle sauvegardé
- Prédiction via `torch.softmax` pour obtenir les probabilités
- Retour de l’étiquette prédite et du score de confiance

Les champs **titre** et **contenu** sont saisis séparément dans l’interface, mais **ne sont pas fusionnés**. Seul le champ "contenu" est utilisé pour la prédiction :

```python
def predict(content):
    inputs = tokenizer(content, return_tensors="pt", truncation=True, padding=True, max_length=512)
    ...
```

---

### 3. 🖼️ Interface Gradio

```python
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Titre de l'article"),
        gr.Textbox(label="Contenu de l'article", lines=5),
    ],
    outputs=gr.Textbox(label="Prédiction"),
    title="Démo BERT : Classification d'Articles",
    description="Entrez un titre et un texte pour prédire la catégorie"
)
```

Ajoutez `share=True` pour générer un lien public :

```python
demo.launch(share=True)
```

---

## 🧪 Exemple

Entrée :
- **Titre** : “La présidente signe un nouveau décret”
- **Contenu** : “Dans un discours au parlement, la présidente a annoncé...”

Sortie :
> `Politics News — confiance : 91.25%`

---

## 📦 Installation des dépendances

```bash
pip install torch transformers gradio
pip install -r requirements.txt
```

---

## 🚀 Lancer l’application

```bash
python demo.py
```

---

## 📌 Notes

- Pour une classification multi-classes, augmentez la valeur de `num_classes`.
- Le tokenizer et le modèle doivent être sauvegardés dans le dossier `saved_model_bert/`.
- Le modèle ne fusionne pas les champs "titre" et "contenu" pour la prédiction.
