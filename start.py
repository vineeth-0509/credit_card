import os
os.system(f"streamlit run app.py --server.port {os.getenv('PORT', '8501')}")