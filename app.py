import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import libpysal
import esda
from splot.esda import moran_scatterplot, lisa_cluster

# ==========================================================
# CONFIGURAÇÃO INICIAL
# ==========================================================
st.set_page_config(page_title="Dashboard Anatel - Telefonia Móvel (SMP)", layout="wide")

# ==========================================================
# FUNÇÕES DE CARREGAMENTO (CACHE)
# ==========================================================
@st.cache_data
def carregar_mapa_smp():
    # Lê o arquivo diretamente da raiz do projeto (Git)
    gdp = gpd.read_file("mapa_smp_completo.geojson")
    gdp['codigo_ibge'] = gdp['codigo_ibge'].astype(int)
    # Define o índice como código IBGE para alinhar com a matriz espacial
    gdp = gdp.set_index('codigo_ibge', drop=False) 
    return gdp

@st.cache_data
def gerar_matriz_espacial(_gdp):
    # Calcula a matriz Queen diretamente da geometria na hora! (Evita erros de indexação)
    w = libpysal.weights.Queen.from_dataframe(_gdp)
    w.transform = 'r'
    return w

# ==========================================================
# APP STREAMLIT PRINCIPAL
# ==========================================================
def main():
    st.title("📊 Dashboard Analítico - Telefonia Móvel (SMP)")
    st.write("Análise Espacial de Reclamações de Rede e Qualidade (2020 - 2025)")

    # Tenta carregar os dados automaticamente da raiz
    try:
        with st.spinner("Carregando base espacial..."):
            mapa_smp = carregar_mapa_smp()
            w_hist = gerar_matriz_espacial(mapa_smp)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo local. Verifique se 'mapa_smp_completo.geojson' está na mesma pasta do app.py. Detalhe: {e}")
        return

    anos = ['2020', '2021', '2022', '2023', '2024', '2025']

    st.sidebar.markdown("### Controles de Visualização")
    # Menu atualizado com a opção de Taxa Bruta
    visao_smp = st.sidebar.radio(
        "Tipo de Análise", 
        ["Mapas de Calor (Suavizados)", "Mapas de Calor (Taxa Bruta)", "Mapas de Clusters (LISA)", "Diagramas de Dispersão (Moran)", "Tabela de Indicadores"]
    )
    modo_exibicao = st.sidebar.radio("Modo de Exibição", ["Ano Específico", "Todos os Anos (Comparativo)"])
    
    # Dropdown de seleção de ano
    ano_selecionado = st.sidebar.selectbox("Ano de Referência", options=anos, index=5)

    # Definir quebras fixas (bins) para manter as cores padronizadas nos mapas de calor suavizados
    colunas_suavizadas = [f'taxa_suav_{a}' for a in anos]
    vmax = np.ceil(mapa_smp[colunas_suavizadas].max().max())
    bins_manuais = [0.5, 1.0, 2.0, 5.0, vmax]

    # ==========================================
    # 1. MAPAS DE CALOR (SUAVIZADOS)
    # ==========================================
    if visao_smp == "Mapas de Calor (Suavizados)":
        if modo_exibicao == "Ano Específico":
            coluna_suav = f'taxa_suav_{ano_selecionado}'
            st.subheader(f"Mapa de Calor - Taxa Suavizada Bayesiana ({ano_selecionado})")
            
            fig, ax = plt.subplots(figsize=(7, 5))
            mapa_smp.plot(
                column=coluna_suav, 
                cmap='YlOrRd', 
                scheme='UserDefined',
                classification_kwds={'bins': bins_manuais}, 
                legend=True, 
                ax=ax,
                missing_kwds={'color': 'lightgrey', 'label': 'Sem dados'},
                legend_kwds={'loc': 'lower right', 'fontsize': 8}
            )
            ax.set_title(f"Reclamações Anatel SMP: {ano_selecionado}", fontsize=12)
            ax.axis('off')
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig)

        else:
            st.subheader("Comparativo Histórico - Todos os Anos (2020-2025)")

            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
            axes = axes.flatten()
            
            for i, ano in enumerate(anos):
                ax = axes[i]
                exibir_legenda = (i == 5)
                
                mapa_smp.plot(
                    column=f'taxa_suav_{ano}', 
                    cmap='YlOrRd', 
                    scheme='UserDefined',
                    classification_kwds={'bins': bins_manuais}, 
                    legend=exibir_legenda, 
                    ax=ax,
                    missing_kwds={'color': 'lightgrey'},
                    legend_kwds={'loc': 'center left', 'bbox_to_anchor': (1.05, 0.5), 'fontsize': 9} if exibir_legenda else None
                )
                ax.set_title(f"SMP: {ano}", fontsize=10)
                ax.axis('off')
            
            plt.subplots_adjust(right=0.85, wspace=0.1, hspace=0.2)
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                st.pyplot(fig)

    # ==========================================
    # 1.5. MAPAS DE CALOR (TAXA BRUTA) - ADICIONADO
    # ==========================================
    elif visao_smp == "Mapas de Calor (Taxa Bruta)":
        # Recalcula os bins para a taxa bruta, pois os máximos costumam ser muito maiores
        colunas_brutas = [f'taxa_bruta_{a}' for a in anos]
        vmax_bruta = np.ceil(mapa_smp[colunas_brutas].max().max())
        bins_bruta = [0.5, 1.0, 2.0, 5.0, vmax_bruta]

        if modo_exibicao == "Ano Específico":
            coluna_bruta = f'taxa_bruta_{ano_selecionado}'
            st.subheader(f"Mapa de Calor - Taxa Bruta ({ano_selecionado})")
            st.write("*Visualização antes do tratamento espacial. Municípios com menos de 1000 acessos aparecem como 'Sem dados'.*")
            
            fig, ax = plt.subplots(figsize=(7, 5))
            mapa_smp.plot(
                column=coluna_bruta, 
                cmap='YlOrRd', 
                scheme='UserDefined',
                classification_kwds={'bins': bins_bruta}, 
                legend=True, 
                ax=ax,
                missing_kwds={'color': 'lightgrey', 'label': 'Sem dados'},
                legend_kwds={'loc': 'lower right', 'fontsize': 8}
            )
            ax.set_title(f"Taxa Bruta SMP: {ano_selecionado}", fontsize=12)
            ax.axis('off')
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig)

        else:
            st.subheader("Comparativo Histórico de Taxas Brutas (2020-2025)")

            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
            axes = axes.flatten()
            
            for i, ano in enumerate(anos):
                ax = axes[i]
                exibir_legenda = (i == 5)
                
                mapa_smp.plot(
                    column=f'taxa_bruta_{ano}', 
                    cmap='YlOrRd', 
                    scheme='UserDefined',
                    classification_kwds={'bins': bins_bruta}, 
                    legend=exibir_legenda, 
                    ax=ax,
                    missing_kwds={'color': 'lightgrey'},
                    legend_kwds={'loc': 'center left', 'bbox_to_anchor': (1.05, 0.5), 'fontsize': 9} if exibir_legenda else None
                )
                ax.set_title(f"Taxa Bruta: {ano}", fontsize=10)
                ax.axis('off')
            
            plt.subplots_adjust(right=0.85, wspace=0.1, hspace=0.2)
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                st.pyplot(fig)

    # ==========================================
    # 2. MAPAS DE CLUSTERS (LISA COROPLÉTICO)
    # ==========================================
    elif visao_smp == "Mapas de Clusters (LISA)":
        st.subheader("Mapas de Clusters Espaciais (LISA)")
        st.write("Identificação de aglomerados espaciais significativos (Alto-Alto, Baixo-Baixo, Alto-Baixo, Baixo-Alto)")

        if modo_exibicao == "Ano Específico":
            y = mapa_smp[f'taxa_suav_{ano_selecionado}'].values
            moran_loc = esda.moran.Moran_Local(y, w_hist)
            
            fig, ax = plt.subplots(figsize=(7, 5))
            lisa_cluster(moran_loc, mapa_smp, p=0.05, ax=ax, legend_kwds={'loc': 'lower right', 'fontsize': 8})
            ax.set_title(f"Clusters LISA - {ano_selecionado}", fontsize=12)
            ax.axis('off')
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig)
            
        else:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
            axes = axes.flatten()
            for i, ano in enumerate(anos):
                y = mapa_smp[f'taxa_suav_{ano}'].values
                moran_loc = esda.moran.Moran_Local(y, w_hist)
                ax = axes[i]
                
                # A função lisa_cluster trata legendas internamente, então exibimos em todos para garantir o padrão
                lisa_cluster(moran_loc, mapa_smp, p=0.05, ax=ax, legend_kwds={'loc': 'lower right', 'fontsize': 6})
                ax.set_title(f"Clusters - {ano}", fontsize=10)
                ax.axis('off')
            
            plt.tight_layout()
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                st.pyplot(fig)

    # ==========================================
    # 3. DIAGRAMAS DE DISPERSÃO (MORAN SCATTER)
    # ==========================================
    elif visao_smp == "Diagramas de Dispersão (Moran)":
        st.subheader("Diagramas de Dispersão de Moran Local")

        if modo_exibicao == "Ano Específico":
            y = mapa_smp[f'taxa_suav_{ano_selecionado}'].values
            moran_loc = esda.moran.Moran_Local(y, w_hist)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            moran_scatterplot(moran_loc, p=0.05, ax=ax)
            ax.set_title(f"Diagrama de Dispersão - {ano_selecionado}", fontsize=12)
            ax.set_xlabel("Taxa Suavizada do Município (Z-score)", fontsize=9)
            ax.set_ylabel("Média da Taxa dos Vizinhos (Spatial Lag)", fontsize=9)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig)
            
        else:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
            axes = axes.flatten()
            for i, ano in enumerate(anos):
                y = mapa_smp[f'taxa_suav_{ano}'].values
                moran_loc = esda.moran.Moran_Local(y, w_hist)
                ax = axes[i]
                moran_scatterplot(moran_loc, p=0.05, ax=ax)
                ax.set_title(f"Moran - {ano}", fontsize=10)
                ax.set_xlabel("Z-score", fontsize=7)
                ax.set_ylabel("Spatial Lag", fontsize=7)
            
            plt.tight_layout()
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                st.pyplot(fig)

    # ==========================================
    # 4. TABELA DE INDICADORES (TABELA 1)
    # ==========================================
    elif visao_smp == "Tabela de Indicadores":
        st.subheader(f"Indicadores de Municípios e Capitais - {ano_selecionado}")
        st.write("Comparativo dos 20 municípios com menos acessos e das 27 capitais brasileiras.")
        
        col_acessos = f'Mediana_Acessos_PF_{ano_selecionado}'
        col_reclamacoes = f'qtd_reclamacoes_{ano_selecionado}'
        col_bruta = f'taxa_bruta_{ano_selecionado}'
        col_suav = f'taxa_suav_{ano_selecionado}'

        df_base = mapa_smp[['codigo_ibge', 'NM_MUN', col_acessos, col_reclamacoes, col_bruta, col_suav]].copy()
        df_base = df_base.rename(columns={
            'NM_MUN': 'Município', 
            col_acessos: 'Acessos', 
            col_reclamacoes: 'Reclamações', 
            col_bruta: 'Taxa bruta', 
            col_suav: 'Taxa bayesiana empírica espacial'
        })
        
        df_menores_20 = df_base[df_base['Acessos'] > 0].nsmallest(20, 'Acessos')
        
        codigos_capitais = [
            1100205, 1200401, 1302603, 1400100, 1501402, 1600303, 1721000, 
            2111300, 2211001, 2304400, 2408102, 2507507, 2611606, 2704302, 
            2800308, 2927408, 3106200, 3205309, 3304557, 3550308, 4106902, 
            4205407, 4314902, 5002704, 5103403, 5208707, 5300108
        ]
        df_capitais = df_base[df_base['codigo_ibge'].isin(codigos_capitais)].sort_values(by='Taxa bruta', ascending=False)
        
        tabela_final = pd.concat([df_menores_20, df_capitais]).drop(columns=['codigo_ibge'])
        
        st.dataframe(
            tabela_final.style.format({
                'Acessos': '{:,.0f}',
                'Reclamações': '{:,.0f}',
                'Taxa bruta': '{:.2f}',
                'Taxa bayesiana empírica espacial': '{:.2f}'
            }), 
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()