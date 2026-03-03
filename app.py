import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import gdown

# ==========================================================
# CONFIGURAÇÃO INICIAL
# ==========================================================
st.set_page_config(page_title="Dashboard Anatel SCM", layout="wide")

# ==========================================================
# FUNÇÕES DE DOWNLOAD E CARREGAMENTO DE DADOS (CACHE)
# ==========================================================
@st.cache_data
def baixar_do_drive(file_id, output_name):
    if not os.path.exists(output_name):
        url = f'https://drive.google.com/uc?id={file_id}'
        # fuzzy=True resolve problemas de redirecionamento e arquivos grandes
        gdown.download(url, output_name, quiet=False, fuzzy=True)
    return output_name

@st.cache_data
def carregar_regioes(sheet_id):
    """Lê a planilha do Google Sheets via link de exportação (Resolve Erro 500)."""
    url_export = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        df_regioes = pd.read_csv(url_export)
        
        df_regioes.replace('', np.nan, inplace=True)
        df_regioes['Nome da Região Imediata'] = df_regioes['Nome da Região Imediata'].ffill().str.strip().str.upper()
        df_regioes['Nome do Município'] = df_regioes['Nome do Município'].str.strip().str.upper()
        df_regioes['UF'] = df_regioes['UF'].str.strip().str.upper()

        pop_limpa = df_regioes['Número de Habitantes'].astype(str).str.replace('.', '', regex=False).str.strip()
        df_regioes['Pop_Num'] = pd.to_numeric(pop_limpa, errors='coerce')

        pop_regiao = df_regioes.groupby('Nome da Região Imediata')['Pop_Num'].sum().reset_index()
        return df_regioes, pop_regiao
    except Exception as e:
        st.error(f"Erro ao acessar a Planilha de Regiões: {e}")
        return None, None

@st.cache_data
def processar_base_otimizada(file_id, output_name, df_regioes):
    """Baixa e processa as bases pesadas da Anatel."""
    caminho_local = baixar_do_drive(file_id, output_name)
    
    colunas = ['AnoMês', 'UF', 'Cidade', 'Serviço', 'Assunto', 'Marca']
    
    # Tenta UTF-8, se falhar tenta Latin1 (comum em CSVs brasileiros)
    try:
        df = pd.read_csv(caminho_local, sep=';', usecols=colunas, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(caminho_local, sep=';', usecols=colunas, encoding='latin1')

    # Filtragem SCM
    df['Serviço'] = df['Serviço'].astype(str).str.strip().str.upper()
    df = df[df['Serviço'] == 'SCM'].copy()

    # Cruzamento Geográfico
    df['Cidade'] = df['Cidade'].astype(str).str.strip().str.upper()
    df['UF'] = df['UF'].astype(str).str.strip().str.upper()
    
    df = df.merge(df_regioes[['UF', 'Nome do Município', 'Nome da Região Imediata']],
                  left_on=['UF', 'Cidade'], right_on=['UF', 'Nome do Município'], how='inner')

    # Filtro de Assuntos de Rede
    df['Assunto'] = df['Assunto'].astype(str).str.strip().str.upper()
    mask_rede = df['Assunto'].str.contains('QUALIDADE|INSTALAÇÃO|ATIVAÇÃO|FUNCIONAMENTO|REPARO', na=False)

    df_rede = df[mask_rede].copy()
    return df_rede

# ==========================================================
# FUNÇÕES DE APOIO E GRÁFICOS (DRY)
# ==========================================================
def preparar_analise_regiao(df_rede, pop_regiao):
    vol_rede = df_rede.groupby('Nome da Região Imediata').size().reset_index(name='Qtd')
    analise = vol_rede.merge(pop_regiao, on='Nome da Região Imediata', how='inner')
    analise['Taxa'] = (analise['Qtd'] / analise['Pop_Num']) * 1000
    analise['Z_Score'] = stats.zscore(analise['Taxa'])
    return analise

def plot_serie_temporal(df_rede, titulo):
    tendencia = df_rede.groupby('AnoMês').size().reset_index(name='Qtd').sort_values('AnoMês')
    tendencia['Media_Movel_3M'] = tendencia['Qtd'].rolling(window=3).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=tendencia, x='AnoMês', y='Qtd', label='Volume', alpha=0.5, marker='o', ax=ax)
    sns.lineplot(data=tendencia, x='AnoMês', y='Media_Movel_3M', color='red', label='Tendência (3M)', linewidth=2, ax=ax)
    plt.xticks(rotation=90)
    plt.title(titulo)
    plt.tight_layout()
    return fig

def plot_boxplot(analise, titulo):
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(x=analise['Taxa'], color='lightgreen', ax=ax)
    plt.title(titulo)
    plt.xlabel("Taxa por Mil Habitantes")
    plt.tight_layout()
    return fig

def plot_pareto(df_rede, titulo):
    pareto = df_rede['Marca'].value_counts().reset_index()
    pareto.columns = ['Marca', 'Qtd']
    pareto['%_Acumulada'] = pareto['Qtd'].cumsum() / pareto['Qtd'].sum() * 100

    fig, ax1 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=pareto.head(10), x='Marca', y='Qtd', color='C0', ax=ax1)
    ax2 = ax1.twinx()
    sns.lineplot(data=pareto.head(10), x='Marca', y='%_Acumulada', color='C1', marker='D', ax=ax2)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax2.set_ylim(0, 110)
    plt.title(titulo)
    plt.tight_layout()
    return fig

def plot_top50(analise, titulo):
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.barplot(data=analise.sort_values('Taxa', ascending=False).head(50), 
                x='Nome da Região Imediata', y='Taxa', palette='magma', ax=ax)
    plt.xticks(rotation=90)
    plt.title(titulo)
    plt.tight_layout()
    return fig

# ==========================================================
# APP STREAMLIT PRINCIPAL (COM PERSISTÊNCIA DE DADOS)
# ==========================================================
def main():
    st.title("📊 Dashboard Analítico - Reclamações de Rede ANATEL (SCM)")
    
    # 1. Inicializar o estado da sessão para os dados
    if 'dados_prontos' not in st.session_state:
        st.session_state['dados_prontos'] = False

    st.sidebar.header("Configurações de Dados")
    
    id_regioes = st.sidebar.text_input("ID - Base de Regiões", "1Ko5cTsY6uaF2BU8ErFGHTo2aUkHAY4dj3qAZmNe3ckM")
    id_base_1 = st.sidebar.text_input("ID - Base 2015-2020", "1DqWhXBilLdSv9bsS7wgpvEnxLbmYSd-k")
    id_base_2 = st.sidebar.text_input("ID - Base 2020-2025", "1v3fJelLAo3Afg7crBC6kB0hyQQiDCqte")

    # Botão de Carregamento
    if st.sidebar.button("Baixar e Carregar Dados"):
        try:
            with st.spinner("Conectando ao Google Drive e processando bases..."):
                df_regioes, pop_regiao = carregar_regioes(id_regioes)
                
                if df_regioes is not None:
                    # Carregar e processar
                    df_rede_1 = processar_base_otimizada(id_base_1, 'base_15_20.csv', df_regioes)
                    df_rede_2 = processar_base_otimizada(id_base_2, 'base_20_25.csv', df_regioes)

                    # Salvar no session_state para persistir entre cliques
                    st.session_state['df_regioes'] = df_regioes
                    st.session_state['pop_regiao'] = pop_regiao
                    st.session_state['df_rede_1'] = df_rede_1
                    st.session_state['df_rede_2'] = df_rede_2
                    st.session_state['analise_1'] = preparar_analise_regiao(df_rede_1, pop_regiao)
                    st.session_state['analise_2'] = preparar_analise_regiao(df_rede_2, pop_regiao)
                    
                    st.session_state['dados_prontos'] = True
                    st.sidebar.success("Dados carregados com sucesso!")
                else:
                    st.error("Falha ao carregar base de regiões.")
        except Exception as e:
            st.error(f"Ocorreu um erro no processamento: {e}")

    # Se os dados não foram carregados ainda, para a execução aqui
    if not st.session_state['dados_prontos']:
        st.info("👈 Clique em 'Baixar e Carregar Dados' para iniciar.")
        return

    # A partir daqui, usamos os dados salvos no session_state
    df_regioes = st.session_state['df_regioes']
    pop_regiao = st.session_state['pop_regiao']
    df_rede_1 = st.session_state['df_rede_1']
    df_rede_2 = st.session_state['df_rede_2']
    analise_1 = st.session_state['analise_1']
    analise_2 = st.session_state['analise_2']

    # --- Navegação ---
    aba_selecionada = st.sidebar.radio("Selecione a Análise", 
        ["Análise Visual", "Estatísticas e Rankings", "Sazonalidade (Litoral)", "Análise Macro-Regional e Bayesiana"])

    # --- ABA 1 ---
    if aba_selecionada == "Análise Visual":
        periodo = st.selectbox("Selecione o Período", ["2020-2025", "2015-2020"])
        
        # Lógica de seleção de dados baseada no dropdown
        df_atual = df_rede_2 if periodo == "2020-2025" else df_rede_1
        analise_atual = analise_2 if periodo == "2020-2025" else analise_1

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_serie_temporal(df_atual, f"Série Temporal de Rede ({periodo})"))
            st.pyplot(plot_boxplot(analise_atual, f"Distribuição das Taxas ({periodo})"))
        with col2:
            st.pyplot(plot_pareto(df_atual, f"Pareto Operadoras ({periodo})"))
        
        st.pyplot(plot_top50(analise_atual, f"Top 50 - Taxa de Problemas de Rede por Mil Hab. ({periodo})"))

    # --- ABA 2 (Estatísticas) ---
    elif aba_selecionada == "Estatísticas e Rankings":
        st.header("Análise Estatística Avançada")
        # Pearson e CV (usando analise_2 como padrão)
        r, p_val_r = stats.pearsonr(analise_2['Pop_Num'], analise_2['Qtd'])
        cv = analise_2['Taxa'].std() / analise_2['Taxa'].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Correlação Pearson", f"{r:.4f}", f"p-val: {p_val_r:.4f}")
        col2.metric("Coef. Variação (CV)", f"{cv:.2%}")

        comp_temporal = analise_1[['Nome da Região Imediata', 'Taxa']].merge(
            analise_2[['Nome da Região Imediata', 'Taxa']], on='Nome da Região Imediata', suffixes=('_1', '_2')
        )
        comp_temporal['Delta'] = comp_temporal['Taxa_2'] - comp_temporal['Taxa_1']
        t_stat, p_val_t = stats.ttest_rel(comp_temporal['Taxa_1'], comp_temporal['Taxa_2'])
        col3.metric("Teste t Pareado", f"t = {t_stat:.2f}", f"p-val: {p_val_t:.4f}")

        st.subheader("Rankings Críticos")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 15 Regiões Mais Críticas (Z-Score)**")
            st.dataframe(analise_2[analise_2['Z_Score'] > 2].sort_values('Z_Score', ascending=False).head(15), use_container_width=True)
        with c2:
            st.markdown("**Top 15 Regiões que Mais Pioraram (Delta)**")
            st.dataframe(comp_temporal.sort_values('Delta', ascending=False).head(15), use_container_width=True)

    # --- ABA 3 ---
    elif aba_selecionada == "Sazonalidade (Litoral)":
        litoral = ['CARAGUATATUBA - UBATUBA - SÃO SEBASTIÃO', 'ANGRA DOS REIS', 'CABO FRIO', 'SANTOS', 'FLORIANÓPOLIS']
        df_litoral = df_rede_2[df_rede_2['Nome da Região Imediata'].isin(litoral)].copy()
        df_litoral['Mes'] = df_litoral['AnoMês'].astype(str).str[-2:]
        sazonalidade = df_litoral.groupby(['Mes', 'Nome da Região Imediata']).size().unstack()

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=sazonalidade, markers=True, ax=ax)
        plt.title("Sazonalidade: Regiões Litorâneas")
        plt.grid(alpha=0.3)
        st.pyplot(fig)

    # --- ABA 4 ---
    elif aba_selecionada == "Análise Macro-Regional e Bayesiana":
        st.header("Análise Macro-Regional e Bayesiana")
        mapa_regioes = {
            'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
            'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
            'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
            'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
            'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
        }
        
        mapa_uf = df_regioes[['Nome da Região Imediata', 'UF']].drop_duplicates()
        analise_reg = analise_2.merge(mapa_uf, on='Nome da Região Imediata', how='left')
        analise_reg['Regiao'] = analise_reg['UF'].map(mapa_regioes)

        # Gráfico Macro
        df_macro = analise_reg.groupby('Regiao').agg({'Pop_Num': 'sum', 'Qtd': 'sum'}).reset_index()
        df_macro['Taxa_Regiao'] = (df_macro['Qtd'] / df_macro['Pop_Num']) * 1000
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df_macro.sort_values('Taxa_Regiao', ascending=False), x='Regiao', y='Taxa_Regiao', palette='viridis', ax=ax)
        st.pyplot(fig)

        # Bayes
        st.subheader("Suavização Bayesiana (Nível Municipal)")
        vol_mun = df_rede_2.groupby('Nome do Município').size().reset_index(name='Qtd_Mun')
        pop_mun = df_regioes.groupby(['Nome do Município', 'UF'])['Pop_Num'].sum().reset_index()
        analise_mun = vol_mun.merge(pop_mun, on='Nome do Município', how='inner')
        analise_mun['Taxa_Mun'] = (analise_mun['Qtd_Mun'] / analise_mun['Pop_Num']) * 1000
        analise_mun['Regiao'] = analise_mun['UF'].map(mapa_regioes)

        taxa_global = (analise_mun['Qtd_Mun'].sum() / analise_mun['Pop_Num'].sum()) * 1000
        m = analise_mun['Pop_Num'].median()
        peso_c = analise_mun['Pop_Num'] / (analise_mun['Pop_Num'] + m)
        analise_mun['Taxa_Bayesiana'] = (peso_c * analise_mun['Taxa_Mun']) + ((1 - peso_c) * taxa_global)
        
        top_bayes = analise_mun.sort_values('Taxa_Bayesiana', ascending=False).groupby('Regiao').head(5)
        st.dataframe(top_bayes[['Regiao', 'UF', 'Nome do Município', 'Taxa_Mun', 'Taxa_Bayesiana']].round(2), use_container_width=True)

if __name__ == "__main__":
    main()