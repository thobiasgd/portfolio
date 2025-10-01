# **Reconhecimento Facial com OpenCV e ONNX**
## Introdução: O rosto da tecnologia

Vivemos em uma era em que a tecnologia está literalmente reconhecendo nossos rostos. Você desbloqueia o celular apenas olhando para ele. Entra no banco, no aeroporto ou até mesmo em salas de aula – e é identificado automaticamente. O reconhecimento facial, antes um conceito de ficção científica, agora é uma realidade presente em diversos setores:

- Segurança pública: câmeras que reconhecem foragidos automaticamente.
- Empresas: controle de ponto por face.
- Acessibilidade: identificação de pessoas para deficientes visuais.
- Entretenimento e mídia: filtros e efeitos baseados no seu rosto.

Mas como tudo isso acontece? O que há por trás da câmera?
Este projeto mostra, na prática, como construir um sistema simples e eficaz de reconhecimento facial utilizando OpenCV + ONNX.

## **Sobre o Projeto**
Este projeto foi construído com o objetivo de:
- Treinar um banco de dados facial baseado em imagens.
- Reconhecer pessoas em vídeos, identificando quem é quem.

Utilizar modelos leves e eficientes em formato ONNX, garantindo portabilidade e performance.

## **Requisitos**

-Python 3.8+
-OpenCV com suporte à FaceDetectorYN (opencv-contrib-python)
-ONNX Runtime
-Tqdm
-NumPy

Instale com:
```
python -m pip install -r requirements.txt
```

## **Estrutura do Projeto**

Abaixo, explicamos cada parte do projeto para que você entenda como tudo se conecta.

### **```app.py``` — O cérebro do sistema**

Este é o arquivo principal, o ponto de entrada da aplicação. Ele permite que você escolha entre dois modos de operação:

- ```train```: escaneia uma pasta com imagens organizadas por pessoa, detecta os rostos e gera um banco de dados de "embeddings" (vetores que representam a face).

- ```infer```: faz a leitura de um vídeo e realiza a identificação dos rostos frame por frame.

Esse script aceita argumentos via terminal e chama os módulos corretos conforme o modo selecionado.

---

### **```config.py``` — Onde vivem as configurações**

Contém **parâmetros globais**, como:

- Caminhos para modelos ONNX (```models/```)
- Thresholds de detecção e reconhecimento
- Cores para desenhar caixas no vídeo
- Caminhos dos vídeos de entrada/saída
- Nome dos arquivos de cache (```bank_cache.npz```) e banco de dados (```database.json```)

Isso facilita ajustes sem mexer no código-fonte principal.

---

### **```databaseEmbeddingGenerator.py``` — Criando o banco de rostos**

Aqui é onde o treinamento acontece.

1. Percorre a pasta do dataset (```dataset/pessoa/*.jpg```).
2. Usa o modelo ONNX de detecção facial para localizar rostos.
3. Faz o crop da face detectada.
4. Extrai os embeddings (representações numéricas da face) usando o modelo de reconhecimento.
5. Salva tudo em um arquivo ```database.json```.

Esse arquivo é depois usado para comparar rostos e identificar pessoas em vídeos.

---

### **```functions.py``` — Utilitários essenciais**

Este módulo contém funções auxiliares, como:

- ```preprocessForModel```: redimensiona e normaliza a imagem da face para o modelo.
- ```l2_normalize```: normaliza vetores para facilitar a comparação.
- Funções para carregar/salvar o banco de embeddings (```.json``` e ```.npz```).
- ```get_inference```: prepara os modelos ONNX e retorna as sessões de inferência.

---

### **```inference.py``` — Rosto a rosto, frame a frame**

Aqui é onde a mágica acontece: o vídeo é processado e os rostos são reconhecidos.

**Fluxo da inferência:**

1. Carrega o banco de embeddings (.npz ou .json).
2. Abre o vídeo de entrada.
3. Para cada frame:
    - Detecta os rostos com YuNet.
    - Extrai embeddings usando o modelo ONNX.
    - Compara com o banco de dados.
    - Identifica a pessoa (ou mostra como "Unknown").
    - Desenha caixas e rótulos no vídeo.
4. Salva o vídeo de saída com as informações sobre os rostos detectados.

O reconhecimento é feito usando similaridade de cosseno entre embeddings.

## **Como Funciona o Reconhecimento Facial?**

1. Detecção Facial: primeiro localizamos onde estão os rostos.
2. Alinhamento e Normalização: cortamos a face e preparamos a imagem.
3. Extração de Embeddings: passamos a imagem por um modelo neural que converte rostos em vetores numéricos.
4. Comparação: comparamos esses vetores com os do banco de dados.
5. Classificação: se a similaridade for alta o suficiente, classificamos como uma pessoa conhecida.

## **Dataset Esperado**

Formato do dataset para o modo ```train```:

```
dataset/
├── maria/
│   ├── 1.jpg
│   └── 2.jpg
├── joao/
│   └── 1.jpg
```

Cada subpasta representa uma pessoa diferente.

## **Exemplos de Execução**
```
# Etapa 1: Treinar o banco de dados
python app.py train dataset/

# Etapa 2: Rodar inferência no vídeo
python app.py infer input.mp4 output.mp4
```

💼 Casos de Uso

Controle de acesso por reconhecimento facial

Sistemas de presença em salas de aula

Análise automática de vídeos de segurança

Protótipos de aplicações com visão computacional
