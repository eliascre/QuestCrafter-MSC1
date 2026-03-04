import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

st.set_page_config(
    page_title="QuestCrafter DM", 
    page_icon="🐉",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling for a professional RPG look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
        border-color: #ff2b2b;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.4);
    }
    .quest-box {
        background-color: #1e2532;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        color: #e0e6ed;
        font-family: 'Georgia', serif;
        line-height: 1.6;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    adapter_path = os.path.join(os.path.dirname(__file__), "models", "tuned_questcrafter")
    
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        st.warning(f"No tuned model found at {adapter_path}. Using base distilgpt2.")
        model = base_model
        
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")
    return generator

st.title("🐉 QuestCrafter: AI Dungeon Master")
st.markdown("### Generate short fantasy quests with customized settings, levels, and tones.")
st.markdown("---")

generator = load_model()

col1, col2, col3 = st.columns(3)
with col1:
    level = st.selectbox("🛡️ Quest Level", [1, 5, 10])
with col2:
    setting = st.selectbox("🏰 Setting", ["forest", "desert", "cyberpunk", "medieval city", "dark cave", "abandoned space station"])
with col3:
    tone = st.selectbox("🎭 Tone", ["epic", "humorous", "dark", "casual", "mysterious"])

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Generate Quest ✨"):
    # Using the control token format from Week 4
    prompt = f"Create a level-{level} quest in a {setting} space with a {tone} tone.\nQuest:"
    
    with st.spinner("The Dungeon Master is weaving a tale..."):
        outputs = generator(prompt, max_new_tokens=150, num_return_sequences=1, truncation=True)
        quest_text = outputs[0]['generated_text'].replace(prompt, "").strip()
        
        st.markdown("<br><h3>📜 Your Quest Overview:</h3>", unsafe_allow_html=True)
        st.markdown(f'<div class="quest-box">{quest_text}</div>', unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Powered by `distilgpt2` + LoRA fine-tuning on TinyStories dataset.")

