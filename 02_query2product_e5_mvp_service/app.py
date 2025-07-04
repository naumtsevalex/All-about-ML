import streamlit as st
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from time import perf_counter
import matplotlib.pyplot as plt
import seaborn as sns

# Это очень плохо так писать, не бейти
class E5Model(torch.nn.Module):
    def __init__(self, model_name='intfloat/e5-small'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]

    def forward(self, anchor_ids, anchor_mask, pos_ids, pos_mask, neg_ids, neg_mask):
        anchor_emb = self.encode(anchor_ids, anchor_mask)
        pos_emb = self.encode(pos_ids, pos_mask)
        neg_emb = self.encode(neg_ids, neg_mask)
        return anchor_emb, pos_emb, neg_emb


# --- Константы ---
PRETRAIN_MODEL_PATH = "saved_e5_model"
# FT_MODEL_PATH = "checkpoints/e5_train_20250703_170142.pt" # lr=2e-5
FT_MODEL_PATH = "checkpoints/e5_train_20250703_173139.pt"  # lr=1e-4 + 3 epochs
# FT_MODEL_PATH = "checkpoints/e5_train_20250703_175235.pt"  # lr=1e-4 + 10 epochs


PRODUCTS_PATH = "product_titles.csv"
# EMBEDS_PATH = "small_product_embeddings.npy"
EMBEDS_PATH = "product_embeddings.npy"
STATS_PATH = "product_stats.csv"  # CSV с колонками ['product_title', 'views']



# --- Выбор модели ---
st.sidebar.markdown("🧠 **Выбор модели**")
model_choice = st.sidebar.selectbox(
    "Модель для поиска:",
    options=["Дообученная (saved_e5_model)", "Предобученная (intfloat/e5-small)"]
)


# --- Загрузка модели и данных (кешируется) ---
@st.cache_resource
def load_model_and_data(model_choice):
    if model_choice == "Дообученная (saved_e5_model)":
        model = E5Model(model_name="intfloat/e5-small")  # init из того же архетипа
        model.load_state_dict(torch.load(FT_MODEL_PATH))
        tokenizer = model.tokenizer
        model = model.encoder.eval().cuda()  # достаём encoder
    else:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL_PATH)
        model = AutoModel.from_pretrained(PRETRAIN_MODEL_PATH).eval().cuda()
        
    product_titles = pd.read_csv(PRODUCTS_PATH).squeeze().tolist()
    product_embs = np.load(EMBEDS_PATH)

    # Загрузка статистики
    product_stats_df = pd.read_csv(STATS_PATH)
    product_stats = dict(zip(product_stats_df['product_title'], product_stats_df['views']))
    return tokenizer, model, product_titles, product_embs, product_stats

tokenizer, model, product_titles, product_embs, product_stats = load_model_and_data(model_choice=model_choice)
total_products = len(product_titles)

# --- Функция кодирования запроса ---
def encode_query(query):
    inputs = tokenizer([f"query: {query}"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda()
        )
    return outputs.last_hidden_state[:, 0].cpu().numpy()[0]

# --- Интерфейс ---
st.title("🔎 Поиск по товарам (E5)")

st.markdown(f"🛍️ **Всего товаров в базе:** `{total_products}`")

query = st.text_input("Введите запрос:", value="wireless headphones")
top_k = st.slider("Сколько результатов показать:", 1, 30, 10)
    
if query:
    with st.status("🔄 Обработка запроса... Пожалуйста, подождите.", expanded=True) as status:

        # --- Этап 1: Кодирование запроса ---
        t1 = perf_counter()
        query_emb = encode_query(query)
        query_time = perf_counter() - t1

        query_emb /= np.linalg.norm(query_emb)
        prod_embs_norm = product_embs / np.linalg.norm(product_embs, axis=1, keepdims=True)

        # --- Этап 2: Поиск по базе ---
        t2 = perf_counter()
        scores = np.dot(prod_embs_norm, query_emb)
        search_time = perf_counter() - t2

        # --- Этап 3: Top-K ранжирование ---
        top_idx = np.argsort(scores)[-top_k:][::-1]
        scores_top = scores[top_idx]

        # --- Интерфейс: две колонки ---
        col1, col2 = st.columns([2, 1])

        # --- Левая колонка: выдача товаров ---
        with col1:
            st.subheader("📋 Результаты:")
            for i in top_idx:
                title = product_titles[i]
                score = scores[i]
                views = product_stats.get(title, 0)

                st.markdown(
                    f"""
                    <div style='
                        background: #1c1c1c; 
                        padding: 10px 15px; 
                        margin: 8px 0; 
                        border-left: 4px solid #52c41a; 
                        border-radius: 6px;
                    '>
                        <div style='font-size: 16px; font-weight: bold; color: #ffffff;'>{title}</div>
                        <div style='margin-top: 4px; color: #cccccc; font-size: 14px;'>
                            🔍 <span style='color:#52c41a;'>score: {score:.3f}</span> &nbsp;&nbsp; 
                            👁️ <span style='color:#f4d35e;'>{views:,} показов</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown(f"⏱️ Время кодирования запроса: `{query_time:.4f} сек`")
            st.markdown(f"📡 Время поиска по базе: `{search_time:.4f} сек`")

        # --- Правая колонка: анализ ---
        with col2:
            st.markdown("### 📊 Анализ")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(scores, bins=50, kde=True, ax=ax, color='skyblue', label='Все товары')
            ax.axvline(scores_top.min(), color='green', linestyle='--', label='Min Top-K')
            ax.axvline(scores_top.max(), color='orange', linestyle='--', label='Max Top-K')
            ax.set_xlabel("Score")
            ax.set_title("Распределение")
            ax.legend()
            st.pyplot(fig)

            st.markdown("### 📐 Статистика")
            stats_dict = {
                "Min": float(np.min(scores)),
                "Max": float(np.max(scores)),
                "Mean": float(np.mean(scores)),
                "Median": float(np.median(scores)),
                "Std": float(np.std(scores)),
                "5%": float(np.percentile(scores, 5)),
                "25%": float(np.percentile(scores, 25)),
                "75%": float(np.percentile(scores, 75)),
                "95%": float(np.percentile(scores, 95)),

            }
            stats_df = pd.DataFrame(stats_dict, index=["Score"]).T
            st.dataframe(stats_df.style.format("{:.4f}"))

        status.update(label="✅ Готово!", state="complete", expanded=True)


# Для запуска:
# streamlit run app.py