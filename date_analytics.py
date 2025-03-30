import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pycountry
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import requests


# Configuração da página
st.set_page_config(page_title="Análise especulação imobiliária para se viver", page_icon="🌍", layout="wide")

# Título
st.title("Análise especulação imobiliária para se viver, usando IA como ferramenta de análise")
st.write("A análise considera o Crescimento do PIB (%) e Crescimento Populacional (%) para identificar o melhor país.")

# Carregando os dados
target_columns = ["Country", "Year", "GDP Growth (%)", "Population Growth (%)"]
try:
    imoveis = pd.read_csv("imoveis.csv", encoding="latin1")
    imoveis.columns = imoveis.columns.str.strip()
    
    # Verificando colunas essenciais
    if not all(col in imoveis.columns for col in target_columns):
        missing_cols = [col for col in target_columns if col not in imoveis.columns]
        st.error(f"As seguintes colunas estão ausentes no dataset: {missing_cols}. Por favor, corrija e tente novamente.")
        st.stop()

except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}. Certifique-se de que o arquivo `imoveis.csv` está no diretório correto e possui os dados esperados.")
    st.stop()

# Selecionando o ano mais recente
ano_mais_recente = imoveis["Year"].max()
dados_recentes = imoveis[imoveis["Year"] == ano_mais_recente]

# Checando valores nulos
if dados_recentes[["GDP Growth (%)", "Population Growth (%)"]].isnull().values.any():
    st.warning("Existem valores nulos nas colunas usadas para análise. Por favor, verifique o dataset.")
    st.stop()

# Criando um índice de qualidade
pesos = {"GDP Growth (%)": 0.7, "Population Growth (%)": 0.3}
dados_recentes["Score"] = (
    dados_recentes["GDP Growth (%)"] * pesos["GDP Growth (%)"] +
    dados_recentes["Population Growth (%)"] * pesos["Population Growth (%)"]
)

# Identificando o melhor país
pais_ideal = dados_recentes.loc[dados_recentes["Score"].idxmax(), "Country"]
pib_pais = dados_recentes.loc[dados_recentes["Country"] == pais_ideal, "GDP Growth (%)"].values[0]
pop_pais = dados_recentes.loc[dados_recentes["Country"] == pais_ideal, "Population Growth (%)"].values[0]

# Gerando análise textual
analise_textual = f"""
🏆 O melhor país para se viver atualmente é **{pais_ideal}**!

🔹 **Crescimento do PIB**: {pib_pais:.2f}% ao ano.
🔹 **Crescimento Populacional**: {pop_pais:.2f}% ao ano.

📊 O {pais_ideal} apresenta um crescimento econômico sustentável e um equilíbrio populacional adequado.
Isso indica um ambiente favorável para oportunidades de trabalho, qualidade de vida e desenvolvimento social.

🌟 Se você está buscando um local para viver, e a(o) {pais_ideal} pode ser uma excelente escolha!
"""

st.dataframe(imoveis)

# Exibir análise textual
st.subheader(" Análise textual com IA")
st.markdown(analise_textual)

# Validar nomes de países
map_data = dados_recentes[["Country", "GDP Growth (%)", "Score"]].copy()
map_data["Country"] = map_data["Country"].str.strip()  # Remover espaços extras
invalid_countries = [
    country for country in map_data["Country"] 
    if not pycountry.countries.get(name=country)
]

if invalid_countries:
    st.warning(f"Os seguintes países não foram reconhecidos e podem não aparecer no mapa: {invalid_countries}. Verifique se os nomes estão em inglês e correspondem aos padrões internacionais.")


# Comparativo de Score
fig_bar = px.bar(
    dados_recentes,
    x="Country",
    y="Score",
    color="Score",
    title="Comparativo de Score por País",
    labels={"Score": "Índice de Qualidade", "Country": "País"}
)
st.plotly_chart(fig_bar)

correlation = dados_recentes[["GDP Growth (%)", "Population Growth (%)"]].corr()
st.write("Correlação entre Crescimento do PIB e Crescimento Populacional:")
st.write(correlation)

fig, ax = plt.subplots()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
st.pyplot(fig)


fig_line = px.line(
    imoveis,
    x="Year",
    y=["GDP Growth (%)", "Population Growth (%)"],
    color="Country",
    title="Tendência de Crescimento ao Longo dos Anos",
    labels={"value": "Percentual (%)", "Year": "Ano"}
)
st.plotly_chart(fig_line)

# Padronizando os dados
scaler = StandardScaler()
dados_cluster = scaler.fit_transform(dados_recentes[["GDP Growth (%)", "Population Growth (%)"]])

# Aplicando K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
dados_recentes["Cluster"] = kmeans.fit_predict(dados_cluster)

# Visualizando os clusters
fig_cluster = px.scatter(
    dados_recentes,
    x="GDP Growth (%)",
    y="Population Growth (%)",
    color="Cluster",
    hover_name="Country",
    title="Clusterização de Países com Base em Crescimento Econômico e Populacional"
)
st.plotly_chart(fig_cluster)

pais_selecionado = st.selectbox("Selecione um país para análise detalhada:", dados_recentes["Country"].unique())
dados_pais = imoveis[imoveis["Country"] == pais_selecionado]

fig_pais = px.bar(
    dados_pais,
    x="Year",
    y="GDP Growth (%)",
    title=f"Crescimento do PIB de {pais_selecionado} ao Longo dos Anos"
)
st.plotly_chart(fig_pais)

# Dividindo os dados
X = imoveis[["Year"]]
y = imoveis["GDP Growth (%)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Treinando o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazendo previsões
future_years = pd.DataFrame({"Year": range(2025, 2031)})
predictions = model.predict(future_years)
future_years["Predicted GDP Growth (%)"] = predictions

# Visualizando as previsões
fig_forecast = px.line(future_years, x="Year", y="Predicted GDP Growth (%)", title="Previsão de Crescimento do PIB (%)")
st.plotly_chart(fig_forecast)

fig = px.choropleth(
    map_data,
    locations="Country",
    locationmode="country names",
    color="Score",
    hover_name="Country",
    title="Mapa Interativo: Índice de Qualidade por País",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig)

def generate_pdf(dataframe):
    # Cria um arquivo PDF em memória
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    # Título do PDF
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(100, 750, "Análise Descritiva dos Dados")

    # Adicionar análise descritiva ao PDF
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, 700, "Resumo Estatístico dos Dados:")
    summary = dataframe.describe()
    y_position = 680
    for column, values in summary.iteritems():
        pdf.drawString(50, y_position, f"{column}: {values}")
        y_position -= 20
        if y_position < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y_position = 750

    # Adicionar gráfico ao PDF
    pdf.drawString(50, y_position - 40, "Gráfico - Crescimento do PIB")
    plt.figure(figsize=(6, 4))
    dataframe["GDP Growth (%)"].plot(kind="bar", title="Crescimento do PIB (%) por País")
    plt.savefig(buffer, format="png")
    pdf.drawInlineImage(buffer, 100, y_position - 150, width=400, height=200)

    # Finalizar o PDF
    pdf.save()
    buffer.seek(0)  # Voltar ao início do arquivo
    return buffer.getvalue()

st.sidebar.image('logo.webp', width=400)
st.sidebar.markdown("## Análise Descritiva dos Dados imobiliários")
st.sidebar.divider()
st.sidebar.markdown("""
    ## Conecte-se comigo:
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Williams%20Rodrigues-blue?logo=linkedin)](https://www.linkedin.com/in/williamsrvs/)
""")
st.sidebar.markdown("""
     [📲 Me Chama no zap](https://wa.me/5582981090042)
""")
st.sidebar.markdown("""
## Fale Conosco:
📧 [Email: dateanalytics@outlook.com](mailto:dateanalytics@outlook.com)
                    
                    """)
st.sidebar.markdown("""
## Saiba Mais:                    
Williams Rodrigues - 
    Analista de Dados e Desenvolvedor Python
## Formação Acadêmica:
🎓 MBA em Ciência de bigdate Analytics
🎓 Bacharel em Administração de Empresas

## Outras áreas de atuação
                    
🏛 Monitor e Consultor de Microsoft Excel
                    
💻Especialista em Power Bi
                    
📊 Consultor de Business Inteligence                        
""")

##Visite meu portífólio

st.sidebar.link_button("Portfólio Williams Rodrigues", url="https://portifolio-wrvs.streamlit.app/", icon="🌎")


# Função para obter a taxa de câmbio
def get_exchange_rate(moeda_origem, moeda_destino):
    url = "https://api.exchangerate-api.com/v4/latest/USD"  # API gratuita
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "rates" not in data:
            return 1  # Retorna 1 caso a API falhe
        
        # Converte para USD primeiro, depois para a moeda de destino
        taxa_origem = data["rates"].get(moeda_origem, None)
        taxa_destino = data["rates"].get(moeda_destino, None)
        
        if taxa_origem and taxa_destino:
            return taxa_destino / taxa_origem
        else:
            return 1  # Caso a moeda não seja encontrada
    
    except Exception as e:
        st.sidebar.error(f"Erro ao obter taxa de câmbio: {e}")
        return 1  # Evita erro no cálculo

# Sidebar
st.sidebar.header("Conversor de Moedas")

# Lista de moedas suportadas
moedas = ["BRL", "USD", "EUR"]  # BRL = Real, USD = Dollar, EUR = Euro

moeda_origem = st.sidebar.selectbox("Moeda de Origem", moedas)
moeda_destino = st.sidebar.selectbox("Moeda de Destino", moedas)
valor = st.sidebar.number_input("Digite o valor a ser convertido", min_value=0.0)

if st.sidebar.button("Converter"):
    taxa = get_exchange_rate(moeda_origem, moeda_destino)
    valor_convertido = valor * taxa
    st.sidebar.write(f"O valor convertido de {moeda_origem} para {moeda_destino} é: {valor_convertido:.2f}")


#Agradecimentos ao desenvolvedor da lib
st.markdown("""
            
    ## Agradecimentos:
            
Foram mais de 6 horas de desenvolvimento deste projeto de estudo usando IA e Python. Agradeço a minha esposa que me ajudou durante esse processo, e foi compreensiva para me ajudar. 
            
Se estiverem com dificuldades ou precisarem de ajuda, sinta-se livre para entrar em contato comigo. Será uma satisfação enorme poder te ajudar.

Ass.: Williams Rodrigues    
 
    """)
