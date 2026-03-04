# Interactive Interface (`script/demoapp.py`)

## Purpose
A web-based graphical interface constructed using `streamlit` to seamlessly test and utilize the fine-tuned AI Dungeon Master model visually.

## Tasks Performed
1. **Caching Resources**: Utilizes `@st.cache_resource` to mount the loaded parameters efficiently into memory, blocking redundant loading instances upon refresh.
2. **Controls**: Provides Dropdown/Selectbox fields matching the control token structures parameterized in Week 4:
   - `Level`: Target length and complexity.
   - `Setting`: Base scene descriptions.
   - `Tone`: Humorous, Dark, Serious configurations.
3. **Generation Event**: Wraps the Hugging Face text pipeline triggering inference and streaming it into the markdown rendering window.

## How to Run
```bash
streamlit run script/demoapp.py
```
