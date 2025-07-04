import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import re
import os
from typing import Dict, Optional, Any

# Tentar importar OpenAI
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    if OPENAI_AVAILABLE:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except:
    OPENAI_AVAILABLE = False
    client = None

# Tentar importar spaCy
# Novo Bloco (para colocar no lugar)
try:
    st.write("DEBUG: Tentando importar spacy...")
    import spacy
    st.write("DEBUG: spaCy importado com sucesso!")

    st.write("DEBUG: Tentando carregar o modelo de linguagem 'pt_core_news_sm'...")
    nlp = spacy.load("pt_core_news_sm")
    st.write("DEBUG: Modelo spaCy carregado com sucesso!")

    SPACY_AVAILABLE = True
except Exception as e:
    # ISSO IRÁ IMPRIMIR O ERRO REAL NA TELA
    st.error(f"ERRO DETALHADO DO SPACY: {e}")
    SPACY_AVAILABLE = False
    nlp = None

# Configuração da página
st.set_page_config(
    page_title="Kickstarter Success Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurações
API_URL = os.getenv("KICKSTARTER_API_URL", "https://api-case-6sy7.onrender.com")
st.write(f"DEBUG: Tentando conectar na API em: {API_URL}")

# Base de dados de usuários (requisito do case)
USERS_DATABASE = {
    "joao@example.com": {
        "nome": "João Silva",
        "cargo": "Gerente de Projetos",
        "experiencia_anos": 5,
        "projetos_historico": 15,
        "taxa_sucesso_pessoal": 0.80,
        "categorias_experiencia": ["Technology", "Design"],
        "projetos_detalhes": [
            {"nome": "Smart Home App", "categoria": "Technology", "sucesso": True, "meta": 25000},
            {"nome": "Eco Design Kit", "categoria": "Design", "sucesso": True, "meta": 15000},
            {"nome": "AI Assistant", "categoria": "Technology", "sucesso": False, "meta": 50000}
        ]
    },
    "maria@example.com": {
        "nome": "Maria Santos", 
        "cargo": "Analista de Projetos",
        "experiencia_anos": 3,
        "projetos_historico": 10,
        "taxa_sucesso_pessoal": 0.65,
        "categorias_experiencia": ["Games", "Art"],
        "projetos_detalhes": [
            {"nome": "Board Game Adventure", "categoria": "Games", "sucesso": True, "meta": 10000},
            {"nome": "Digital Art Gallery", "categoria": "Art", "sucesso": True, "meta": 8000},
            {"nome": "Mobile Game RPG", "categoria": "Games", "sucesso": False, "meta": 30000}
        ]
    },
    "pedro@example.com": {
        "nome": "Pedro Oliveira",
        "cargo": "Coordenador de Projetos", 
        "experiencia_anos": 8,
        "projetos_historico": 25,
        "taxa_sucesso_pessoal": 0.90,
        "categorias_experiencia": ["Film & Video", "Music", "Publishing"],
        "projetos_detalhes": [
            {"nome": "Documentary Series", "categoria": "Film & Video", "sucesso": True, "meta": 40000},
            {"nome": "Music Album", "categoria": "Music", "sucesso": True, "meta": 12000}
        ]
    },
    "default": {
        "nome": "Novo Usuário",
        "cargo": "Criador de Projetos",
        "experiencia_anos": 0,
        "projetos_historico": 0,
        "taxa_sucesso_pessoal": 0.0,
        "categorias_experiencia": [],
        "projetos_detalhes": []
    }
}

# CSS customizado
st.markdown("""
<style>
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .danger-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        text-align: left;
        margin-right: 20%;
    }
    .chat-header-split {
        background: #1f77b4;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
    }
    .top-chat-container {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .user-profile-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #a5d6a7;
        margin: 10px 0;
    }
    .extraction-method {
        font-size: 0.8em;
        color: #666;
        font-style: italic;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar session state
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'project_data' not in st.session_state:
    st.session_state.project_data = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = USERS_DATABASE["default"]
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'extraction_method' not in st.session_state:
    st.session_state.extraction_method = None

# Verificar se API está online
@st.cache_data(ttl=60)
def check_api_health():
    """Verifica se a API está online"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=60)
        return response.status_code == 200
    except:
        return False

# Carregar categorias
@st.cache_data(ttl=300)
def load_categories():
    """Carrega categorias disponíveis da API"""
    try:
        response = requests.get(f"{API_URL}/info/categories")
        if response.status_code == 200:
            data = response.json()
            return {cat['value']: cat for cat in data['categories']}
    except:
        pass
    
    # Fallback se API não responder
    return {
        'Film & Video': {'description': 'Filmes, documentários, vídeos', 'avg_success': '42%'},
        'Music': {'description': 'Álbuns, shows, instrumentos', 'avg_success': '53%'},
        'Publishing': {'description': 'Livros, revistas, e-books', 'avg_success': '35%'},
        'Games': {'description': 'Jogos de tabuleiro, card games, RPG', 'avg_success': '44%'},
        'Technology': {'description': 'Gadgets, apps, hardware', 'avg_success': '24%'},
        'Design': {'description': 'Produtos, móveis, acessórios', 'avg_success': '42%'},
        'Art': {'description': 'Pinturas, esculturas, instalações', 'avg_success': '45%'},
        'Comics': {'description': 'HQs, graphic novels, mangás', 'avg_success': '59%'},
        'Theater': {'description': 'Peças, musicais, performances', 'avg_success': '64%'},
        'Food': {'description': 'Restaurantes, produtos alimentícios', 'avg_success': '28%'},
        'Photography': {'description': 'Projetos fotográficos, livros de fotos', 'avg_success': '34%'},
        'Fashion': {'description': 'Roupas, calçados, acessórios', 'avg_success': '28%'},
        'Dance': {'description': 'Espetáculos, workshops, vídeos', 'avg_success': '65%'},
        'Journalism': {'description': 'Reportagens, documentários jornalísticos', 'avg_success': '24%'},
        'Crafts': {'description': 'Artesanato, DIY, kits', 'avg_success': '27%'}
    }

# Países disponíveis
COUNTRIES = {
    'US': 'Estados Unidos',
    'GB': 'Reino Unido',
    'CA': 'Canadá',
    'AU': 'Austrália',
    'DE': 'Alemanha',
    'FR': 'França',
    'IT': 'Itália',
    'ES': 'Espanha',
    'NL': 'Países Baixos',
    'SE': 'Suécia',
    'BR': 'Brasil',
    'JP': 'Japão',
    'MX': 'México'
}

# Categorias válidas
VALID_CATEGORIES = {
    'Film & Video', 'Music', 'Publishing', 'Games', 'Technology',
    'Design', 'Art', 'Comics', 'Theater', 'Food', 'Photography',
    'Fashion', 'Dance', 'Journalism', 'Crafts'
}

# Mapeamento de categorias em português
CATEGORY_MAPPING = {
    'filme': 'Film & Video',
    'vídeo': 'Film & Video',
    'video': 'Film & Video',
    'música': 'Music',
    'musica': 'Music',
    'publicação': 'Publishing',
    'publicacao': 'Publishing',
    'livro': 'Publishing',
    'jogos': 'Games',
    'jogo': 'Games',
    'game': 'Games',
    'tecnologia': 'Technology',
    'tech': 'Technology',
    'design': 'Design',
    'arte': 'Art',
    'quadrinhos': 'Comics',
    'hq': 'Comics',
    'teatro': 'Theater',
    'comida': 'Food',
    'alimentação': 'Food',
    'fotografia': 'Photography',
    'foto': 'Photography',
    'moda': 'Fashion',
    'dança': 'Dance',
    'danca': 'Dance',
    'jornalismo': 'Journalism',
    'artesanato': 'Crafts'
}

# Adicionar estas funções ao código do app_streamlit_hybrid_completo.py

def preprocess_message(message: str) -> str:
    """
    Pré-processa a mensagem para corrigir erros comuns
    """
    # Converter para lowercase para comparações
    message_lower = message.lower()
    
    # Dicionário de correções comuns
    corrections = {
        # Erros de digitação comuns
        'categria': 'categoria',
        'categorai': 'categoria',
        'catgoria': 'categoria',
        'categora': 'categoria',
        
        # Variações de palavras
        'dolar': 'dollar',
        'dolares': 'dollars',
        'reais': 'dollars',  # Assumir conversão
        
        # Abreviações de valores
        'k ': '000 ',
        'mil ': '000 ',
        
        # Países
        'brasil': 'BR',
        'estados unidos': 'US',
        'eua': 'US',
        'usa': 'US',
        
        # Categorias em português
        'tecnologia': 'Technology',
        'jogos': 'Games',
        'música': 'Music',
        'musica': 'Music',
        'arte': 'Art',
        'filmes': 'Film & Video',
        'filme': 'Film & Video',
        'video': 'Film & Video',
        'vídeo': 'Film & Video',
        'design': 'Design',
        'comida': 'Food',
        'teatro': 'Theater',
        'dança': 'Dance',
        'danca': 'Dance',
        'fotografia': 'Photography',
        'moda': 'Fashion',
        'artesanato': 'Crafts',
        'publicação': 'Publishing',
        'publicacao': 'Publishing',
        'quadrinhos': 'Comics',
        'jornalismo': 'Journalism'
    }
    
    # Aplicar correções
    result = message
    for wrong, correct in corrections.items():
        # Usar regex para substituir palavras completas
        import re
        pattern = r'\b' + re.escape(wrong) + r'\b'
        result = re.sub(pattern, correct, result, flags=re.IGNORECASE)
    
    return result

def extract_with_spacy_improved(message: str) -> Optional[Dict[str, Any]]:
    """
    Versão melhorada do extrator spaCy com pré-processamento
    """
    if not SPACY_AVAILABLE:
        return None
    
    try:
        # Pré-processar mensagem
        processed_message = preprocess_message(message)
        print(f"Mensagem original: {message}")
        print(f"Mensagem processada: {processed_message}")
        
        # Padrões regex expandidos
        patterns = {
            'nome': [
                # Padrões estruturados
                r'nome:\s*([^\n,]+?)(?=\s+categoria:|$)',
                r'projeto:\s*([^\n,]+?)(?=\s+categoria:|$)',
                r'título:\s*([^\n,]+?)(?=\s+categoria:|$)',
                
                # Padrões mais flexíveis
                r'meu projeto (?:é|e|se chama)\s+([^\n,]+?)(?=\s+categoria|$)',
                r'projeto\s+([^\n,]+?)\s+(?:categoria|da categoria)',
                r'analise?\s+(?:o\s+)?(?:meu\s+)?projeto\s+([^\n,]+?)\s+categoria',
                
                # Padrão mais genérico (última tentativa)
                r'(?:projeto|nome)\s*:?\s*([a-zA-Z0-9\s\-\_]+?)(?=\s*(?:categoria|tipo|meta|$))'
            ],
            'categoria': [
                # Padrões estruturados
                r'categoria:\s*([^\n,]+?)(?=\s+meta:|$)',
                r'tipo:\s*([^\n,]+?)(?=\s+meta:|$)',
                r'category:\s*([^\n,]+?)(?=\s+meta:|$)',
                
                # Padrões flexíveis
                r'(?:da\s+)?categoria\s+([^\n,]+?)(?=\s+meta|$)',
                r'é\s+(?:um|uma)\s+([^\n,]+?)(?=\s+meta|com|$)',
                
                # Categorias entre aspas ou parênteses
                r'categoria[:\s]+["\']([^"\']+)["\']',
                r'categoria[:\s]+\(([^)]+)\)',
                
                # Padrão genérico
                r'categoria\s*:?\s*([a-zA-Z\s&]+?)(?=\s*(?:meta|valor|$))'
            ],
            'meta': [
                # Valores monetários estruturados
                r'meta:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'objetivo:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'goal:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'valor:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                
                # Valores com k/mil
                r'meta\s*:?\s*(\d+)\s*(?:k|mil)',
                r'(\d+)\s*(?:k|mil)\s*(?:dólares|dolares|reais|dollars)',
                
                # Valores entre símbolos
                r'\$\s*([\d,]+(?:\.\d{2})?)',
                r'R\$\s*([\d,]+(?:\.\d{2})?)',
                
                # Padrão genérico
                r'meta\s*:?\s*(?:de\s+)?\$?\s*([\d,\.]+)'
            ],
            'pais': [
                r'país:\s*([A-Za-z]{2})',
                r'pais:\s*([A-Za-z]{2})',
                r'country:\s*([A-Za-z]{2})',
                r'local:\s*([A-Za-z]{2})',
                r'de\s+([A-Za-z]{2})(?:\s|$)'
            ],
            'inicio': [
                r'início:\s*(\d{4}-\d{2}-\d{2})',
                r'inicio:\s*(\d{4}-\d{2}-\d{2})',
                r'começa:\s*(\d{4}-\d{2}-\d{2})',
                r'lançamento:\s*(\d{4}-\d{2}-\d{2})',
                r'start:\s*(\d{4}-\d{2}-\d{2})',
                r'data\s+(?:de\s+)?início:\s*(\d{4}-\d{2}-\d{2})'
            ],
            'fim': [
                r'fim:\s*(\d{4}-\d{2}-\d{2})',
                r'término:\s*(\d{4}-\d{2}-\d{2})',
                r'termino:\s*(\d{4}-\d{2}-\d{2})',
                r'deadline:\s*(\d{4}-\d{2}-\d{2})',
                r'end:\s*(\d{4}-\d{2}-\d{2})',
                r'data\s+(?:de\s+)?fim:\s*(\d{4}-\d{2}-\d{2})',
                r'até:\s*(\d{4}-\d{2}-\d{2})'
            ]
        }
        
        # Extrair dados usando regex
        extracted_data = {}
        
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, processed_message, re.IGNORECASE)
                if match:
                    extracted_data[field] = match.group(1).strip()
                    break
        
        # Processar valores especiais
        if 'meta' in extracted_data:
            meta_str = extracted_data['meta']
            
            # Verificar se tem 'k' ou 'mil'
            if 'k' in message.lower() or 'mil' in message.lower():
                # Extrair apenas números
                numbers = re.findall(r'(\d+)', meta_str)
                if numbers:
                    extracted_data['meta'] = str(int(numbers[0]) * 1000)
        
        # Validar se temos dados mínimos
        if not all(key in extracted_data for key in ['nome', 'categoria', 'meta']):
            print(f"Dados incompletos. Extraídos: {extracted_data}")
            return None
        
        # Converter e validar dados
        try:
            # Limpar e converter meta
            meta_str = extracted_data['meta'].replace(',', '')
            if '.' in meta_str and len(meta_str.split('.')[-1]) == 2:
                meta = float(meta_str)
            else:
                meta_str = meta_str.replace('.', '')
                meta = float(meta_str)
            
            # Normalizar categoria
            categoria = normalize_category(extracted_data['categoria'])
            
            # Preparar dados finais
            project_data = {
                "name": extracted_data['nome'],
                "main_category": categoria,
                "country": extracted_data.get('pais', 'US').upper(),
                "usd_goal_real": meta,
                "launched": extracted_data.get('inicio', datetime.now().strftime("%Y-%m-%d")),
                "deadline": extracted_data.get('fim', (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
            }
            
            # Validar datas
            launched_date = datetime.strptime(project_data['launched'], "%Y-%m-%d")
            deadline_date = datetime.strptime(project_data['deadline'], "%Y-%m-%d")
            
            if deadline_date <= launched_date:
                project_data['deadline'] = (launched_date + timedelta(days=30)).strftime("%Y-%m-%d")
            
            return project_data
            
        except Exception as e:
            print(f"Erro na conversão: {e}")
            return None
            
    except Exception as e:
        print(f"Erro geral: {e}")
        return None

# Modificar a função extract_project_info_from_message para usar a versão melhorada
def extract_project_info_from_message(message):
    """Extrai informações do projeto da mensagem do usuário"""
    # Primeiro tenta com spaCy melhorado
    if SPACY_AVAILABLE:
        project_data = extract_with_spacy_improved(message)
        if project_data:
            st.session_state.extraction_method = "spaCy (local/gratuito)"
            return project_data 
    
    # Se spaCy falhar e OpenAI estiver disponível
    if OPENAI_AVAILABLE and client:
        try:
            prompt = f"""
            Extraia as informações do projeto Kickstarter desta mensagem.
            Retorne APENAS um JSON válido com os campos:
            - name: nome do projeto
            - main_category: categoria (deve ser uma das válidas: Technology, Games, Art, Music, Film & Video, Design, Comics, Theater, Food, Photography, Fashion, Dance, Journalism, Crafts, Publishing)
            - country: código do país (2 letras)
            - usd_goal_real: meta em dólares (número)
            - launched: data de início (YYYY-MM-DD)
            - deadline: data fim (YYYY-MM-DD)
            
            Se algum campo não for mencionado, use valores padrão razoáveis.
            
            Mensagem: {message}
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1
            )
            
            json_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                st.session_state.extraction_method = "OpenAI GPT-3.5 (fallback)"
                return extracted_data
        except Exception as e:
            print(f"Erro com OpenAI: {e}")
    
    return None

# BOTÃO TEMPORÁRIO PARA TREINAR MODELO - ADICIONE ANTES DAS BOAS-VINDAS
# Verificar se o modelo precisa ser treinado
try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    health_data = health_response.json() if health_response.status_code == 200 else {}
    model_loaded = health_data.get('model_loaded', False)
except:
    model_loaded = False

if not model_loaded:
    st.markdown("---")
    col1_train, col2_train, col3_train = st.columns([1, 2, 1])
    with col2_train:
        st.markdown("### 🚨 CONFIGURAÇÃO INICIAL DO MODELO")
        st.warning("⚠️ O modelo ainda não foi treinado na API. É necessário treinar o modelo antes de usar o sistema.")
        
        if st.button("🚀 TREINAR MODELO NA API", type="primary", use_container_width=True):
            with st.spinner("Iniciando treinamento do modelo... Isso pode levar 2-5 minutos."):
                try:
                    response = requests.post(f"{API_URL}/train", timeout=30)
                    result = response.json()
                    
                    if response.status_code == 200:
                        st.success(f"✅ {result.get('message', 'Treinamento iniciado com sucesso!')}")
                        st.info("📊 O treinamento demora cerca de 2-5 minutos. Use o botão abaixo para verificar o status.")
                        st.balloons()
                    else:
                        st.error(f"❌ Erro ao iniciar treinamento: {result}")
                except Exception as e:
                    st.error(f"❌ Erro ao conectar com a API: {str(e)}")
        
        # Botão para verificar status
        if st.button("🔍 Verificar Status do Modelo", use_container_width=True):
            try:
                response = requests.get(f"{API_URL}/health", timeout=10)
                health_data = response.json()
                
                if health_data.get('model_loaded'):
                    st.success("✅ Modelo carregado e pronto para uso! 🎉")
                    st.info("🔄 Recarregue a página para começar a usar o sistema.")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.warning("⏳ Modelo ainda está sendo treinado. Aguarde mais um pouco...")
                    st.json(health_data)
            except Exception as e:
                st.error(f"❌ Erro ao verificar status: {str(e)}")
        
        # Mostrar informações sobre o processo
        with st.expander("ℹ️ Sobre o processo de treinamento"):
            st.markdown("""
            **O que acontece durante o treinamento:**
            1. 📥 Download automático dos dados do Kickstarter
            2. 🧹 Limpeza e preparação dos dados
            3. 🤖 Treinamento do modelo de Machine Learning
            4. 💾 Salvamento do modelo treinado
            5. ✅ Disponibilização para uso
            
            **Tempo estimado:** 2-5 minutos
            
            **Nota:** Este processo só precisa ser feito uma vez.
            """)
    
    st.markdown("---")
    st.stop()  # Para a execução aqui se o modelo não estiver carregado

# Seção de Boas-Vindas e Instruções
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h1 style="text-align: center; margin-bottom: 20px;">🎯 Bem-vindo ao Kickstarter Success Predictor!</h1>
    <p style="text-align: center; font-size: 1.1em; margin-bottom: 30px;">
        Sou seu assistente de IA especializado em prever o sucesso de projetos no Kickstarter.<br>
        Posso analisar seu projeto e dar recomendações personalizadas!
    </p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.user_email or st.session_state.user_email == "default":
    # Container para login rápido
    st.markdown("""
    <div style="background: #f8f9fa; border: 2px dashed #dee2e6; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h3 style="text-align: center; color: #495057;">🚀 Experimente com um Usuário Demo</h3>
        <p style="text-align: center; color: #6c757d;">Clique em um dos perfis abaixo para testar o sistema com histórico real</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Três colunas para os botões
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    
    with demo_col1:
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h4>👨‍💼 João Silva</h4>
            <p style="font-size: 0.9em; color: #666;">
                Gerente de Projetos<br>
                5 anos experiência<br>
                Taxa sucesso: 80%<br>
                <span style="color: #28a745;">✓ Technology ✓ Design</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Entrar como João", key="btn_joao", use_container_width=True, type="primary"):
            st.session_state.user_email = "joao@example.com"
            st.session_state.user_data = USERS_DATABASE["joao@example.com"]
            st.success("✅ Logado como João Silva!")
            time.sleep(1)
            st.rerun()
    
    with demo_col2:
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h4>👩‍💻 Maria Santos</h4>
            <p style="font-size: 0.9em; color: #666;">
                Analista de Projetos<br>
                3 anos experiência<br>
                Taxa sucesso: 65%<br>
                <span style="color: #17a2b8;">✓ Games ✓ Art</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Entrar como Maria", key="btn_maria", use_container_width=True):
            st.session_state.user_email = "maria@example.com"
            st.session_state.user_data = USERS_DATABASE["maria@example.com"]
            st.success("✅ Logada como Maria Santos!")
            time.sleep(1)
            st.rerun()
    
    with demo_col3:
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h4>👨‍🎨 Pedro Oliveira</h4>
            <p style="font-size: 0.9em; color: #666;">
                Coordenador<br>
                8 anos experiência<br>
                Taxa sucesso: 90%<br>
                <span style="color: #6610f2;">✓ Film ✓ Music ✓ Publishing</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Entrar como Pedro", key="btn_pedro", use_container_width=True):
            st.session_state.user_email = "pedro@example.com"
            st.session_state.user_data = USERS_DATABASE["pedro@example.com"]
            st.success("✅ Logado como Pedro Oliveira!")
            time.sleep(1)
            st.rerun()
    
    # Opção de continuar sem login
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("➡️ Continuar sem login", key="btn_anonimo", use_container_width=True):
        st.info("Você pode usar o sistema, mas não terá análises personalizadas baseadas em histórico.")
        time.sleep(1.5)
        st.rerun()

else:
    # Se já está logado, mostrar card de boas-vindas personalizado
    user = st.session_state.user_data
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0;">
        <h3>👋 Bem-vindo de volta, {user['nome']}!</h3>
        <p>
            <strong>{user['cargo']}</strong> | 
            {user['experiencia_anos']} anos de experiência | 
            {user['projetos_historico']} projetos | 
            Taxa de sucesso: {user['taxa_sucesso_pessoal']:.0%}
        </p>
        <p>Especialista em: {', '.join(user['categorias_experiencia'])}</p>
    </div>
    """, unsafe_allow_html=True)

# Container de instruções expandível
with st.expander("📝 **Como usar o Assistente de IA** (Clique para ver exemplos)", expanded=True):
    col_inst1, col_inst2 = st.columns(2)
    
    with col_inst1:
        st.markdown("""
        ### ✨ Formato Recomendado
        ```
        Analise meu projeto: 
        Nome: [nome] 
        Categoria: [categoria] 
        Meta: $[valor] 
        País: [código] 
        Início: YYYY-MM-DD 
        Fim: YYYY-MM-DD
        ```
        """)
    
    with col_inst2:
        st.markdown("""
        ### ✅ Exemplos Funcionais
        ```
        Analise meu projeto: Nome: power 
        Categoria: Games Meta: $10,000 
        País: US Início: 2025-07-03 
        Fim: 2025-08-02
        ```
        
        ```
        Sou joao@example.com. Analise meu projeto: 
        Nome: power Categoria: Technology 
        Meta: $10000 País: US 
        Início: 2025-07-04 Fim: 2025-08-03
        ```
        """)  

# Linha divisória estilizada
st.markdown("""
<div style="height: 2px; background: linear-gradient(to right, transparent, #667eea, #764ba2, transparent); margin: 20px 0;"></div>
""", unsafe_allow_html=True)

def get_initial_chat_message():
    """Retorna mensagem inicial com exemplos para o usuário"""
    return """
Como posso ajudar com seu projeto hoje?
"""

# Modificar a resposta de erro para ser mais útil
def get_error_response():
    """Retorna mensagem de erro útil com exemplos"""
    return """
❌ **Não consegui entender os dados do seu projeto.**

📝 **Por favor, use um destes formatos:**

**Formato completo (recomendado):**
```
Analise meu projeto: 
Nome: [nome do projeto]
Categoria: [categoria]
Meta: $[valor]
País: [código de 2 letras]
Início: YYYY-MM-DD
Fim: YYYY-MM-DD
```

**Formato simplificado:**
```
Nome: [projeto] Categoria: [categoria] Meta: $[valor]
```

**Exemplos reais que funcionam:**
- `Analise meu projeto: Nome: SmartHome Categoria: Technology Meta: $15,000 País: US`
- `Nome: BoardGame Fun Categoria: Games Meta: $8,000`
- `projeto EcoBottle design 5000 dolares`

**Categorias válidas:**
- **Tecnologia**: Technology
- **Jogos**: Games
- **Arte**: Art
- **Música**: Music
- **Filme/Vídeo**: Film & Video
- **Design**: Design
- **Outras**: Comics, Theater, Food, Photography, Fashion, Dance, Journalism, Crafts, Publishing

💡 **Dicas:**
- Escreva valores sem vírgulas: $10000 (não $10,000)
- Use códigos de país: US, BR, GB, etc.
- Posso entender português: "jogos" → Games

Tente novamente! Estou aqui para ajudar 😊
"""

def normalize_category(category: str) -> str:
    """Normaliza categoria para formato válido"""
    # Primeiro tenta match direto
    if category in VALID_CATEGORIES:
        return category
    
    # Tenta mapear do português
    category_lower = category.lower().strip()
    if category_lower in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[category_lower]
    
    # Tenta match parcial
    for key, value in CATEGORY_MAPPING.items():
        if key in category_lower or category_lower in key:
            return value
    
    # Se não encontrar, retorna Technology como padrão
    return "Technology"

def extract_with_spacy(message: str) -> Optional[Dict[str, Any]]:
    """
    Extrai informações do projeto usando spaCy e regex.
    Retorna None se não conseguir extrair informações suficientes.
    """
    if not SPACY_AVAILABLE:
        return None
    
    try:
        # Converter para lowercase para facilitar matching
        message_lower = message.lower()
        
        # Padrões regex para cada campo - CORRIGIDOS
        patterns = {
            'nome': [
                r'nome:\s*([^\n,]+?)(?=\s+categoria:|$)',
                r'projeto:\s*([^\n,]+?)(?=\s+categoria:|$)',
                r'título:\s*([^\n,]+?)(?=\s+categoria:|$)',
                r'nome\s+(?:é|e)\s+([^\n,]+?)(?=\s+categoria:|$)'
            ],
            'categoria': [
                r'categoria:\s*([^\n,]+?)(?=\s+meta:|$)',
                r'tipo:\s*([^\n,]+?)(?=\s+meta:|$)',
                r'category:\s*([^\n,]+?)(?=\s+meta:|$)'
            ],
            'meta': [
                # Padrões mais específicos para capturar valores monetários completos
                r'meta:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'objetivo:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'goal:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'\$\s*([\d,]+(?:\.\d{2})?)'
            ],
            'pais': [
                r'país:\s*([A-Za-z]{2})',
                r'pais:\s*([A-Za-z]{2})',
                r'country:\s*([A-Za-z]{2})',
                r'local:\s*([A-Za-z]{2})'
            ],
            'inicio': [
                r'início:\s*(\d{4}-\d{2}-\d{2})',
                r'inicio:\s*(\d{4}-\d{2}-\d{2})',
                r'começa:\s*(\d{4}-\d{2}-\d{2})',
                r'lançamento:\s*(\d{4}-\d{2}-\d{2})',
                r'start:\s*(\d{4}-\d{2}-\d{2})'
            ],
            'fim': [
                r'fim:\s*(\d{4}-\d{2}-\d{2})',
                r'término:\s*(\d{4}-\d{2}-\d{2})',
                r'termino:\s*(\d{4}-\d{2}-\d{2})',
                r'deadline:\s*(\d{4}-\d{2}-\d{2})',
                r'end:\s*(\d{4}-\d{2}-\d{2})'
            ]
        }
        
        # Debug: imprimir a mensagem recebida
        print(f"Mensagem recebida: {message}")
        
        # Extrair dados usando regex
        extracted_data = {}
        
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    extracted_data[field] = match.group(1).strip()
                    print(f"Extraído {field}: {extracted_data[field]}")
                    break
        
        # Validar se temos dados mínimos
        if not all(key in extracted_data for key in ['nome', 'categoria', 'meta']):
            print(f"Dados incompletos. Extraídos: {extracted_data}")
            return None
        
        # Converter e validar dados
        try:
            # Limpar e converter meta - CORREÇÃO PRINCIPAL
            meta_str = extracted_data['meta'].replace(',', '')
            # Não remover o ponto se for decimal
            if '.' in meta_str and len(meta_str.split('.')[-1]) == 2:
                # É um valor decimal (ex: 10000.00)
                meta = float(meta_str)
            else:
                # É um valor inteiro (ex: 10000 ou 10,000)
                meta_str = meta_str.replace('.', '')
                meta = float(meta_str)
            
            print(f"Meta convertida: {meta}")
            
            # Normalizar categoria
            categoria = normalize_category(extracted_data['categoria'])
            
            # Preparar dados finais
            project_data = {
                "name": extracted_data['nome'],
                "main_category": categoria,
                "country": extracted_data.get('pais', 'US').upper(),
                "usd_goal_real": meta,
                "launched": extracted_data.get('inicio', datetime.now().strftime("%Y-%m-%d")),
                "deadline": extracted_data.get('fim', (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
            }
            
            # Validar datas
            launched_date = datetime.strptime(project_data['launched'], "%Y-%m-%d")
            deadline_date = datetime.strptime(project_data['deadline'], "%Y-%m-%d")
            
            if deadline_date <= launched_date:
                # Ajustar deadline se inválido
                project_data['deadline'] = (launched_date + timedelta(days=30)).strftime("%Y-%m-%d")
            
            print(f"Dados finais: {project_data}")
            return project_data
            
        except Exception as e:
            print(f"Erro na conversão: {e}")
            return None
            
    except Exception as e:
        print(f"Erro geral: {e}")
        return None

# Funções do Chatbot
def make_prediction_from_chat(project_info):
    """Faz predição através do chat"""
    try:
        # Preparar dados para API
        project_data = {
            "name": project_info.get('name', 'My Kickstarter Project'),
            "main_category": project_info.get('category', 'Technology'),
            "country": project_info.get('country', 'US'),
            "usd_goal_real": float(project_info.get('goal', 10000)),
            "launched": project_info.get('launched', datetime.now().strftime("%Y-%m-%d")),
            "deadline": project_info.get('deadline', (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
        }
        
        # Fazer requisição para API
        response = requests.post(f"{API_URL}/predict", json=project_data)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}



def get_chat_response(user_message, context=None):
    """Gera resposta do chatbot usando OpenAI ou respostas predefinidas"""
    try:
        # Verificar se o usuário quer fazer uma predição
        prediction_keywords = ['predict', 'prever', 'chance', 'probabilidade', 'analisar projeto', 'analyze', 'analise']
        wants_prediction = any(keyword in user_message.lower() for keyword in prediction_keywords)
        
        # Se quiser predição e tiver dados estruturados
        if wants_prediction:
            # Tentar extrair dados
            project_info = extract_project_info_from_message(user_message)
            
            if project_info:
                # CORREÇÃO: Os dados já vêm padronizados do extract_with_spacy
                # Apenas garantir que os campos estejam corretos
                project_data_for_api = {
                    "name": project_info.get('name', 'My Project'),
                    "main_category": project_info.get('main_category', 'Technology'),
                    "country": project_info.get('country', 'US'),
                    "usd_goal_real": float(project_info.get('usd_goal_real', 10000)),
                    "launched": project_info.get('launched', datetime.now().strftime("%Y-%m-%d")),
                    "deadline": project_info.get('deadline', (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
                }
                
                # Fazer predição real com a API diretamente
                try:
                    # Linha corrigida
                    response = requests.post(f"{API_URL}/predict", json=project_data_for_api, timeout=90)
                    
                    if response.status_code == 200:
                        prediction_result = response.json()
                        
                        # Salvar no contexto com dados padronizados
                        st.session_state.project_data = project_data_for_api
                        st.session_state.prediction_result = prediction_result
                        
                        # Criar resposta formatada
                        duration_days = (pd.to_datetime(project_data_for_api['deadline']) - pd.to_datetime(project_data_for_api['launched'])).days
                        
                        # Carregar categorias para mostrar taxa média
                        categories = load_categories()
                        
                        # Emoji baseado na probabilidade
                        if prediction_result['success_probability'] >= 0.6:
                            emoji_result = "🟢"
                            status_msg = "ALTA CHANCE DE SUCESSO!"
                        elif prediction_result['success_probability'] >= 0.5:
                            emoji_result = "🟡"
                            status_msg = "CHANCE MODERADA"
                        elif prediction_result['success_probability'] >= 0.3:
                            emoji_result = "🟠"
                            status_msg = "CHANCE BAIXA - PRECISA MELHORAR"
                        else:
                            emoji_result = "🔴"
                            status_msg = "ALTO RISCO DE FRACASSO!"
                        
                        # Adicionar análise personalizada baseada no usuário
                        user_analysis = ""
                        if st.session_state.user_email and st.session_state.user_email != "default":
                            user_data = st.session_state.user_data
                            
                            # Comparar com histórico pessoal
                            if user_data['taxa_sucesso_pessoal'] > 0:
                                diff = prediction_result['success_probability'] - user_data['taxa_sucesso_pessoal']
                                if diff > 0:
                                    user_analysis += f"\n\n📊 **Análise Personalizada para {user_data['nome']}:**\n"
                                    user_analysis += f"✅ Este projeto tem {diff*100:.1f}% mais chance que sua média histórica ({user_data['taxa_sucesso_pessoal']:.0%})!"
                                else:
                                    user_analysis += f"\n\n📊 **Análise Personalizada para {user_data['nome']}:**\n"
                                    user_analysis += f"⚠️ Este projeto está {abs(diff)*100:.1f}% abaixo da sua média histórica ({user_data['taxa_sucesso_pessoal']:.0%})"
                            
                            # Verificar experiência na categoria
                            if project_data_for_api['main_category'] in user_data['categorias_experiencia']:
                                user_analysis += f"\n✅ Você tem experiência em {project_data_for_api['main_category']}. Isso é um diferencial!"
                            else:
                                user_analysis += f"\n💡 Primeira vez em {project_data_for_api['main_category']}? Considere buscar mentoria nesta área."
                        
                        # Adicionar método de extração
                        extraction_info = ""
                        if st.session_state.extraction_method:
                            extraction_info = f"\n\n<p class='extraction-method'>📝 Dados extraídos via: {st.session_state.extraction_method}</p>"
                        
                        return f"""
{emoji_result} **{status_msg}**

🎯 **Análise do Projeto: {project_data_for_api['name']}**

📊 **TAXA DE SUCESSO: {prediction_result['success_probability']:.1%}**
🔮 **PREDIÇÃO: {prediction_result['prediction'].upper()}**
💪 **CONFIANÇA: {prediction_result['confidence']}**

**📋 Detalhes do Projeto:**
- 🎬 Categoria: {project_data_for_api['main_category']} (Taxa média de sucesso: {categories.get(project_data_for_api['main_category'], {}).get('avg_success', '42%')})
- 💰 Meta: ${project_data_for_api['usd_goal_real']:,.0f}
- 🌍 País: {project_data_for_api['country']}
- 📅 Duração: {duration_days} dias
- 🚀 Período: {project_data_for_api['launched']} até {project_data_for_api['deadline']}

**🎲 Threshold do modelo: {prediction_result['threshold_used']:.1%}**
Sua probabilidade está {(prediction_result['success_probability'] - prediction_result['threshold_used'])*100:.1f}% {'acima' if prediction_result['success_probability'] > prediction_result['threshold_used'] else 'abaixo'} do threshold.

**💡 Recomendações Personalizadas:**
{chr(10).join(f"- {rec}" for rec in prediction_result['recommendations'])}
{user_analysis}

**📈 Próximos Passos:**
{'✅ Você está no caminho certo! Foque na execução e marketing.' if prediction_result['success_probability'] >= 0.5 else '⚠️ Recomendo ajustar alguns aspectos antes de lançar.'}

Quer que eu:
- 📝 Sugira títulos melhores?
- 💰 Analise se a meta está adequada?
- 📅 Crie um cronograma de campanha?
- 🎁 Monte estrutura de recompensas?
{extraction_info}
"""
                    else:
                        return f"❌ Erro ao fazer predição: API retornou status {response.status_code}"
                        
                except Exception as e:
                    return f"❌ Erro ao fazer predição: {str(e)}\n\nPor favor, use o formulário na aba '🔮 Predictor' para análise precisa."
            else:
                return get_error_response()
        
        # Se não for predição ou não tiver OpenAI, usar respostas padrão
        if not OPENAI_AVAILABLE:
            # Respostas predefinidas para casos sem OpenAI
            message_lower = user_message.lower()
            
            # Verificar se é primeira mensagem/saudação
            if any(word in message_lower for word in ['oi', 'olá', 'hello', 'hi', 'início', 'começ', 'ajud']):
                return get_initial_chat_message()
            elif 'categoria' in message_lower or 'categories' in message_lower:
                categories = load_categories()
                return f"""
**Categorias disponíveis no Kickstarter:**

{chr(10).join(f"- {cat} ({info['avg_success']} sucesso)" for cat, info in categories.items())}

As categorias com maior taxa de sucesso são Dance, Theater e Comics!
"""
            else:
                return """
Desculpe, não entendi sua pergunta. 

**Posso ajudar com:**
- Prever sucesso do seu projeto
- Listar categorias disponíveis
- Dar dicas para melhorar suas chances

Para fazer uma predição, envie os dados do projeto no formato estruturado.
"""
        
        # Se tiver OpenAI, usar para respostas gerais
        system_message = """Você é um consultor especialista em crowdfunding do Kickstarter com 10 anos de experiência.
        
        REGRA CRÍTICA: Você NUNCA deve inventar taxas de sucesso ou probabilidades. 
        Se o usuário pedir uma predição, você DEVE usar a função de predição real que retorna a probabilidade exata do modelo.
        NUNCA diga coisas como "aproximadamente 75%" ou invente números.
        
        Você tem acesso a:
        - Um modelo preditivo treinado com 300,000+ projetos (AUC-ROC: 0.733)
        - Dados estatísticos sobre taxas de sucesso por categoria
        - Base de dados de usuários com histórico de projetos
        - Capacidade de fazer predições REAIS quando o usuário fornecer dados do projeto
        
        IMPORTANTE: Quando fizer uma predição, SEMPRE:
        1. Use os dados REAIS retornados pela API
        2. Mostre a taxa EXATA de sucesso
        3. Considere o histórico do usuário se disponível
        4. Seja direto e objetivo
        
        Se o usuário quiser fazer uma predição, extraia os dados e faça a chamada real para o modelo."""
        
        messages = [{"role": "system", "content": system_message}]
        
        # Adicionar contexto se disponível
        if context:
            context_message = f"""
            Contexto atual do projeto:
            - Nome: {context.get('name', 'Não definido')}
            - Categoria: {context.get('main_category', 'Não definida')}
            - Meta: ${context.get('usd_goal_real', 0):,.2f}
            - País: {context.get('country', 'Não definido')}
            - Duração: {context.get('campaign_days', 'Não definida')} dias
            
            Resultados da predição (se disponível):
            {json.dumps(st.session_state.prediction_result, indent=2) if st.session_state.prediction_result else 'Nenhuma predição feita ainda'}
            """
            messages.append({"role": "system", "content": context_message})
        
        # Adicionar informações do usuário se disponível
        if st.session_state.user_email and st.session_state.user_data:
            user_context = f"""
            Informações do usuário atual:
            - Nome: {st.session_state.user_data['nome']}
            - Cargo: {st.session_state.user_data['cargo']}
            - Experiência: {st.session_state.user_data['experiencia_anos']} anos
            - Projetos anteriores: {st.session_state.user_data['projetos_historico']}
            - Taxa de sucesso pessoal: {st.session_state.user_data['taxa_sucesso_pessoal']:.0%}
            - Experiência em categorias: {', '.join(st.session_state.user_data['categorias_experiencia'])}
            
            Use essas informações para personalizar suas recomendações.
            """
            messages.append({"role": "system", "content": user_context})
        
        # Adicionar histórico de conversa
        for msg in st.session_state.chat_messages[-10:]:  # Últimas 10 mensagens
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Adicionar mensagem atual
        messages.append({"role": "user", "content": user_message})
        
        # Fazer chamada para OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Desculpe, houve um erro ao processar sua mensagem: {str(e)}"

def analyze_project_with_ai(project_data, prediction_result):
    """Análise detalhada do projeto usando AI"""
    if not OPENAI_AVAILABLE:
        return "⚠️ OpenAI não está configurado. Configure OPENAI_API_KEY no arquivo .env para usar esta funcionalidade."
    
    user_context = ""
    if st.session_state.user_data and st.session_state.user_data != USERS_DATABASE["default"]:
        user_context = f"""
        Considere também o perfil do usuário:
        - {st.session_state.user_data['nome']} ({st.session_state.user_data['cargo']})
        - {st.session_state.user_data['experiencia_anos']} anos de experiência
        - Taxa de sucesso histórica: {st.session_state.user_data['taxa_sucesso_pessoal']:.0%}
        - Experiência em: {', '.join(st.session_state.user_data['categorias_experiencia'])}
        """
    
    prompt = f"""
    Analise este projeto Kickstarter e forneça insights detalhados:
    
    Dados do Projeto:
    {json.dumps(project_data, indent=2)}
    
    Resultado da Predição:
    {json.dumps(prediction_result, indent=2)}
    
    {user_context}
    
    Por favor, forneça:
    1. Análise dos pontos fortes e fracos
    2. 3 sugestões específicas para melhorar as chances
    3. Comparação com projetos bem-sucedidos na mesma categoria
    4. Estratégia de lançamento recomendada personalizada para este usuário
    
    Seja específico e prático.
    """
    
    return get_chat_response(prompt)

def generate_title_suggestions(current_title, category):
    """Gera sugestões de títulos melhores"""
    if not OPENAI_AVAILABLE:
        return """
**Sugestões de títulos baseadas em padrões de sucesso:**

1. **[Adjetivo] + [Produto] + [Benefício]**
   - Ex: "Revolutionary Solar Charger for Travelers"
   
2. **[Problema] + [Solução] + [Diferencial]**
   - Ex: "Never Lose Keys Again - Smart Bluetooth Tracker"
   
3. **[Público] + [Necessidade] + [Inovação]**
   - Ex: "Gamers Ultimate Wireless Controller Experience"

**Dicas:**
- Use 4-7 palavras
- Seja específico sobre o que faz
- Inclua um diferencial claro
- Evite jargões técnicos
"""
    
    prompt = f"""
    O título atual do projeto é: "{current_title}"
    Categoria: {category}
    
    Sugira 3 títulos melhores que:
    1. Sejam mais atrativos e descritivos
    2. Incluam palavras-chave relevantes para SEO
    3. Tenham entre 4-7 palavras
    4. Comuniquem claramente o valor do projeto
    
    Para cada sugestão, explique brevemente por que é melhor.
    """
    
    return get_chat_response(prompt)

def optimize_campaign_strategy(project_data, prediction_result):
    """Gera estratégia otimizada de campanha"""
    if not OPENAI_AVAILABLE:
        duration = (pd.to_datetime(project_data['deadline']) - pd.to_datetime(project_data['launched'])).days
        return f"""
**Estratégia de Campanha para {duration} dias:**

**🚀 Pré-Lançamento (7 dias antes):**
- Criar lista de e-mail com interessados
- Preparar conteúdo visual (vídeo + imagens)
- Engajar comunidade nas redes sociais
- Definir recompensas early bird (25% desconto)

**📈 Semana 1 - Momentum Inicial:**
- Objetivo: 30% da meta
- Ativar lista de e-mail no dia 1
- Postar em grupos relevantes
- Atualização diária nas primeiras 48h

**🎯 Semanas 2-3 - Manutenção:**
- Objetivo: 70% da meta
- Atualizações 2x por semana
- Adicionar stretch goals se > 50%
- Engajar apoiadores como embaixadores

**🏁 Última Semana - Sprint Final:**
- Objetivo: 100%+ da meta
- Campanha "últimas horas"
- Oferecer bônus limitados
- Live/AMA com criadores

**📊 Métricas para acompanhar:**
- Taxa de conversão de visitantes
- Ticket médio por apoiador
- Origem do tráfego
- Engajamento nas atualizações
"""
    
    user_context = ""
    if st.session_state.user_data and st.session_state.user_data != USERS_DATABASE["default"]:
        user_context = f"""
        Considere o perfil do usuário:
        - {st.session_state.user_data['nome']} tem {st.session_state.user_data['experiencia_anos']} anos de experiência
        - Taxa de sucesso histórica: {st.session_state.user_data['taxa_sucesso_pessoal']:.0%}
        - Já trabalhou com: {', '.join(st.session_state.user_data['categorias_experiencia'])}
        """
    
    prompt = f"""
    Crie um plano estratégico de 30 dias para maximizar o sucesso desta campanha:
    
    Projeto: {project_data['name']}
    Categoria: {project_data['main_category']}
    Meta: ${project_data['usd_goal_real']:,.2f}
    Probabilidade atual: {prediction_result['success_probability']:.1%}
    
    {user_context}
    
    Inclua:
    1. Cronograma detalhado (pré-lançamento, lançamento, meio, final)
    2. Metas de arrecadação por semana
    3. Estratégias de marketing específicas
    4. Momentos-chave para atualizações
    5. Táticas para manter momentum
    
    Seja prático e específico, considerando a experiência do usuário.
    """
    
    return get_chat_response(prompt)
