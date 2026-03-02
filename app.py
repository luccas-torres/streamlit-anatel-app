import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==========================================================
# CONFIGURAÇÃO INICIAL
# ==========================================================
st.set_page_config(page_title="Dashboard Anatel SCM", layout="wide")

# ==========================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS (CACHE)
# ==========================================================
@st.cache_data
def carregar_regioes(caminho_csv_regioes):
    """Carrega e limpa a base de regiões. (Requer um CSV local exportado do Sheets)"""
    df_regioes = pd.read_csv(caminho_csv_regioes)
    df_regioes.replace('', np.nan, inplace=True)
    df_regioes['Nome da Região Imediata'] = df_regioes['Nome da Região Imediata'].ffill().str.strip().str.upper()
    df_regioes['Nome do Município'] = df_regioes['Nome do Município'].str.strip().str.upper()
    df_regioes['UF'] = df_regioes['UF'].str.strip().str.upper()

    pop_limpa = df_regioes['Número de Habitantes'].astype(str).str.replace('.', '', regex=False).str.strip()
    df_regioes['Pop_Num'] = pd.to_numeric(pop_limpa, errors='coerce')

    pop_regiao = df_regioes.groupby('Nome da Região Imediata')['Pop_Num'].sum().reset_index()
    return df_regioes, pop_regiao

@st.cache_data
def processar_base_otimizada(caminho, df_regioes):
    """Processa as bases da Anatel, retornando apenas os dados de rede filtrados."""
    colunas = ['AnoMês', 'UF', 'Cidade', 'Serviço', 'Assunto', 'Marca']
    df = pd.read_csv(caminho, sep=';', usecols=colunas, encoding='utf-8')

    df['Serviço'] = df['Serviço'].astype(str).str.strip().str.upper()
    df = df[df['Serviço'] == 'SCM'].copy()

    df['Cidade'] = df['Cidade'].astype(str).str.strip().str.upper()
    df['UF'] = df['UF'].astype(str).str.strip().str.upper()
    
    df = df.merge(df_regioes[['UF', 'Nome do Município', 'Nome da Região Imediata']],
                  left_on=['UF', 'Cidade'], right_on=['UF', 'Nome do Município'], how='inner')

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
# APP STREAMLIT PRINCIPAL
# ==========================================================
def main():
    st.title("📊 Dashboard Analítico - Reclamações de Rede ANATEL (SCM)")
    
    st.sidebar.header("Configurações de Dados")
    # Para o Streamlit, o ideal é ter os arquivos em uma pasta data/ local ou fazer upload
    # Caso use uploaders, pode substituir essas strings por st.sidebar.file_uploader()
    caminho_regioes = st.sidebar.text_input("CSV de Regiões", "regioes_geograficas.csv")
    caminho_1 = st.sidebar.text_input("CSV Anatel (2015-2020)", "reclamacoes_ANATEL_2015_2020-002.csv")
    caminho_2 = st.sidebar.text_input("CSV Anatel (2020-2025)", "reclamacoes_ANATEL_2020_2025.csv")

    if not st.sidebar.button("Carregar Dados"):
        st.info("👈 Verifique os caminhos dos arquivos e clique em 'Carregar Dados' na barra lateral para iniciar.")
        return

    try:
        with st.spinner("Carregando bases... Isso pode levar um tempo."):
            df_regioes, pop_regiao = carregar_regioes(caminho_regioes)
            df_rede_1 = processar_base_otimizada(caminho_1, df_regioes)
            df_rede_2 = processar_base_otimizada(caminho_2, df_regioes)
            
            analise_1 = preparar_analise_regiao(df_rede_1, pop_regiao)
            analise_2 = preparar_analise_regiao(df_rede_2, pop_regiao)
            
        st.sidebar.success("Bases carregadas com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return

    # Navegação
    aba_selecionada = st.sidebar.radio("Selecione a Análise", 
        ["Análise Visual", "Estatísticas e Rankings", "Sazonalidade (Litoral)", "Análise Macro-Regional e Bayesiana"])

    # --------------------------------------------------
    # ABA 1: ANÁLISE VISUAL (Reaproveitamento de código)
    # --------------------------------------------------
    if aba_selecionada == "Análise Visual":
        periodo = st.selectbox("Selecione o Período", ["2020-2025", "2015-2020"])
        
        df_atual = df_rede_2 if periodo == "2020-2025" else df_rede_1
        analise_atual = analise_2 if periodo == "2020-2025" else analise_1

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_serie_temporal(df_atual, f"Série Temporal de Rede ({periodo})"))
            st.pyplot(plot_boxplot(analise_atual, f"Distribuição das Taxas ({periodo})"))
        with col2:
            st.pyplot(plot_pareto(df_atual, f"Pareto Operadoras ({periodo})"))
        
        st.pyplot(plot_top50(analise_atual, f"Top 50 - Taxa de Problemas de Rede por Mil Hab. ({periodo})"))

    # --------------------------------------------------
    # ABA 2: ESTATÍSTICAS E RANKINGS
    # --------------------------------------------------
    elif aba_selecionada == "Estatísticas e Rankings":
        st.header("Análise Estatística Avançada e Comparações Temporais")
        
        # Pearson e CV
        r, p_val_r = stats.pearsonr(analise_2['Pop_Num'], analise_2['Qtd'])
        cv = analise_2['Taxa'].std() / analise_2['Taxa'].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Correlação Pearson (Pop x Volume)", f"{r:.4f}", f"p-val: {p_val_r:.4f}")
        col2.metric("Coef. Variação (CV) Taxas", f"{cv:.2%}")

        # Teste T
        comp_temporal = analise_1[['Nome da Região Imediata', 'Taxa']].merge(
            analise_2[['Nome da Região Imediata', 'Taxa']], on='Nome da Região Imediata', suffixes=('_1', '_2')
        )
        comp_temporal['Delta'] = comp_temporal['Taxa_2'] - comp_temporal['Taxa_1']
        t_stat, p_val_t = stats.ttest_rel(comp_temporal['Taxa_1'], comp_temporal['Taxa_2'])
        col3.metric("Teste t Pareado", f"t = {t_stat:.2f}", f"p-val: {p_val_t:.4f}")

        # Tabelas de Ranking
        st.subheader("Rankings Críticos")
        col_rank1, col_rank2 = st.columns(2)
        
        with col_rank1:
            st.markdown("**Top 15 Regiões Mais Críticas (Z-Score > 2)**")
            top_criticas = analise_2[analise_2['Z_Score'] > 2].sort_values('Z_Score', ascending=False).head(15)
            st.dataframe(top_criticas[['Nome da Região Imediata', 'Taxa', 'Z_Score']].round(2), use_container_width=True)

        with col_rank2:
            st.markdown("**Top 15 Regiões que Mais Pioraram (Delta Taxa)**")
            top_piora = comp_temporal.sort_values('Delta', ascending=False).head(15)
            st.dataframe(top_piora[['Nome da Região Imediata', 'Taxa_1', 'Taxa_2', 'Delta']].round(2), use_container_width=True)

    # --------------------------------------------------
    # ABA 3: SAZONALIDADE
    # --------------------------------------------------
    elif aba_selecionada == "Sazonalidade (Litoral)":
        st.header("Análise de Sazonalidade (Cidades Litorâneas)")
        litoral = ['CARAGUATATUBA - UBATUBA - SÃO SEBASTIÃO', 'ANGRA DOS REIS', 'CABO FRIO', 'SANTOS', 'FLORIANÓPOLIS']
        
        df_litoral = df_rede_2[df_rede_2['Nome da Região Imediata'].isin(litoral)].copy()
        df_litoral['Mes'] = df_litoral['AnoMês'].astype(str).str[-2:]
        sazonalidade = df_litoral.groupby(['Mes', 'Nome da Região Imediata']).size().unstack()

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=sazonalidade, markers=True, dashes=False, ax=ax)
        plt.title("Sazonalidade de Reclamações: Regiões Litorâneas")
        plt.xlabel("Mês do Ano")
        plt.ylabel("Volume de Reclamações de Rede")
        plt.grid(alpha=0.3)
        plt.legend(title='Região', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Distribuição Mensal Percentual")
        dist_pct = (sazonalidade.sum(axis=1) / sazonalidade.sum().sum()) * 100
        st.dataframe(dist_pct.round(2).to_frame(name="% do Volume Anual").T)

    # --------------------------------------------------
    # ABA 4: MACRO-REGIONAL E BAYES
    # --------------------------------------------------
    elif aba_selecionada == "Análise Macro-Regional e Bayesiana":
        st.header("Análise por Macro-Regiões do Brasil")
        
        mapa_regioes = {
            'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
            'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
            'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
            'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
            'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
        }

        # Resgatar UF a partir do df_regioes para fazer o merge
        mapa_uf = df_regioes[['Nome da Região Imediata', 'UF']].drop_duplicates()
        analise_reg = analise_2.merge(mapa_uf, on='Nome da Região Imediata', how='left')
        analise_reg['Regiao'] = analise_reg['UF'].map(mapa_regioes)

        df_macro = analise_reg.groupby('Regiao').agg({'Pop_Num': 'sum', 'Qtd': 'sum'}).reset_index()
        df_macro['Taxa_Regiao'] = (df_macro['Qtd'] / df_macro['Pop_Num']) * 1000

        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=df_macro.sort_values('Taxa_Regiao', ascending=False), x='Regiao', y='Taxa_Regiao', palette='viridis', ax=ax)
            plt.title("Taxa de Reclamações de Rede por Grande Região (2020-2025)")
            plt.ylabel("Taxa por Mil Habitantes")
            st.pyplot(fig)

        with col2:
            st.markdown("**Coeficiente de Variação Interno por Região**")
            cv_interno = analise_reg.groupby('Regiao')['Taxa'].apply(lambda x: (x.std() / x.mean()) * 100).reset_index(name='CV_Interno_%')
            st.dataframe(cv_interno.sort_values('CV_Interno_%', ascending=False), use_container_width=True)

        st.markdown("---")
        st.subheader("Suavização Bayesiana em Nível Municipal")
        
        # Corrigindo o DataFrame analise_mun que faltava no script original
        with st.spinner("Processando Nível Municipal..."):
            vol_mun = df_rede_2.groupby('Nome do Município').size().reset_index(name='Qtd_Mun')
            pop_mun = df_regioes.groupby(['Nome do Município', 'UF'])['Pop_Num'].sum().reset_index()
            
            analise_mun = vol_mun.merge(pop_mun, on='Nome do Município', how='inner')
            analise_mun['Taxa_Mun'] = (analise_mun['Qtd_Mun'] / analise_mun['Pop_Num']) * 1000
            analise_mun['Regiao'] = analise_mun['UF'].map(mapa_regioes)

            # Cálculo Bayesiano
            taxa_global = (analise_mun['Qtd_Mun'].sum() / analise_mun['Pop_Num'].sum()) * 1000
            m = analise_mun['Pop_Num'].median()
            
            peso_cidade = analise_mun['Pop_Num'] / (analise_mun['Pop_Num'] + m)
            peso_global = m / (analise_mun['Pop_Num'] + m)
            
            analise_mun['Taxa_Bayesiana'] = (peso_cidade * analise_mun['Taxa_Mun']) + (peso_global * taxa_global)
            analise_mun['Z_Score_Bayes'] = stats.zscore(analise_mun['Taxa_Bayesiana'])

            top_bayes_regiao = analise_mun.sort_values(['Regiao', 'Z_Score_Bayes'], ascending=[True, False]).groupby('Regiao').head(5)

        st.write(f"**Taxa Global de Referência:** {taxa_global:.2f} por mil hab. | **População Mediana (M):** {m:.0f} hab.")

        # Gerar guias para visualização rápida no Streamlit
        tabs = st.tabs(list(top_bayes_regiao['Regiao'].unique()))
        for i, regiao in enumerate(top_bayes_regiao['Regiao'].unique()):
            with tabs[i]:
                df_reg = top_bayes_regiao[top_bayes_regiao['Regiao'] == regiao]
                st.dataframe(df_reg[['UF', 'Nome do Município', 'Pop_Num', 'Qtd_Mun', 'Taxa_Mun', 'Taxa_Bayesiana', 'Z_Score_Bayes']].round(3), use_container_width=True)

if __name__ == "__main__":
    main()