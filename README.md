<p align="center">
  <img src="banner_tecnologia.jpg" alt="Banner de tecnologia">
</p>

# Thobias Gonçalves Dordete

<sub>Engenheiro Mecatrônico | Desenvolvedor de Software | Computer Vision | Machine Vision | Robótica</sub>

Sou **Engenheiro Mecatrônico e Desenvolvedor de Software**, com foco na criação de soluções que conectam software, visão computacional, automação e sistemas físicos.

Atuo com **Computer Vision e Machine Vision**, desenvolvendo aplicações para inspeção, detecção, reconhecimento e análise automatizada de imagens, além da integração dessas soluções com APIs, interfaces web, sensores e sistemas embarcados.

Tenho experiência com **Python, C++, PyTorch, OpenCV e ROS2**, além de tecnologias de desenvolvimento web e backend. Busco aplicar engenharia de software e percepção computacional na construção de sistemas robustos, eficientes e voltados a problemas reais.

### Principais competências

- Machine Vision e Computer Vision
- Python e C++
- PyTorch e OpenCV
- Robótica e ROS2
- Integração entre software, sensores e sistemas físicos
- APIs e aplicações web para soluções de visão computacional

### Links

- [LinkedIn](https://www.linkedin.com/in/thobias-gonçalves-33b19720a)
- [GitHub](https://github.com/thobiasgd)

---

## Projetos

### Machine Vision / Computer Vision

#### [Industrial Inspection — Anomaly Detection](https://github.com/thobiasgd/industrial-inspection)

Sistema completo de inspeção visual industrial para detecção de anomalias em imagens de produtos.

O projeto utiliza **PyTorch, OpenCV e ResNet18** para extração de características e inferência em GPU, além de uma arquitetura web com **FastAPI, NestJS, React e TypeScript**. A aplicação classifica produtos como `APPROVED` ou `REJECTED`, gera heatmaps, overlays e bounding boxes para destacar regiões suspeitas.

No conjunto de teste utilizado com a categoria Bottle do MVTec AD, o protótipo alcançou **100% de acurácia nas 83 imagens avaliadas**, com tempo médio de processamento de aproximadamente **17 ms por imagem** em uma RTX 3060.

**Tecnologias:** Python · PyTorch · OpenCV · FastAPI · NestJS · TypeScript · React · Zod

---

#### [Reconhecimento Facial com OpenCV e ONNX](https://github.com/thobiasgd/portfolio/tree/main/Recognizer)

Pipeline de reconhecimento facial baseado em modelos ONNX, com uma etapa de construção do banco de embeddings e outra de inferência sobre vídeos.

O sistema utiliza o **YuNet**, através do `FaceDetectorYN` do OpenCV, para localizar faces e um modelo de reconhecimento executado com **ONNX Runtime** para gerar embeddings normalizados. Durante a inferência, os vetores são comparados com o banco cadastrado usando similaridade de cosseno, permitindo identificar pessoas conhecidas ou classificá-las como `Unknown`. O projeto também utiliza cache em `.npz`, banco em JSON e processamento frame a frame com geração de vídeo anotado.

**Tecnologias:** Python · OpenCV · ONNX Runtime · NumPy · YuNet · tqdm

---

#### [Detecção e Reconhecimento Facial com Dlib](https://github.com/thobiasgd/portfolio/tree/main/dlib_recognizer)

Sistema de reconhecimento facial em vídeo utilizando detecção de faces, landmarks e embeddings para identificação de pessoas cadastradas em um banco local.

O projeto utiliza o detector frontal da **Dlib**, o modelo de **68 facial landmarks** e o modelo ResNet de reconhecimento facial da própria biblioteca para gerar descritores numéricos das faces. Na etapa de inferência, esses descritores são comparados por distância Euclidiana com as referências conhecidas; o OpenCV é responsável pela leitura e gravação do vídeo, bounding boxes, landmarks e identificação visual de rostos reconhecidos e desconhecidos.

**Tecnologias:** Python · Dlib · OpenCV · NumPy · Pillow · tqdm · ResNet facial embeddings

---

### Robótica / ROS2

#### [Turtlesim Catch Them All](https://github.com/thobiasgd/portfolio/tree/main/turtlesim_catch_them_all)

Aplicação distribuída em ROS 2 na qual uma tartaruga controlada autonomamente localiza, persegue e captura outras tartarugas geradas dinamicamente no ambiente Turtlesim.

A solução foi desenvolvida em **C++** e separada em múltiplos nodes: um spawner responsável por criar e gerenciar os alvos e um controller que seleciona a tartaruga mais próxima e utiliza um **controlador proporcional (P)** para comandar velocidade linear e angular. O projeto explora comunicação por topics e services, parâmetros configuráveis, mensagens e serviços customizados (`Turtle.msg`, `TurtleArray.msg` e `CatchTurtle.srv`) e inicialização da aplicação através de launch file.

**Tecnologias:** C++ · ROS 2 · rclcpp · Turtlesim · CMake · Topics · Services · Custom Messages · Launch XML

---

### Otimização

#### [Otimização Aeronáutica com Algoritmo Genético e OpenVSP](https://github.com/thobiasgd/portfolio/tree/main/MDO)

Sistema de otimização geométrica de asas que integra um algoritmo genético a simulações aerodinâmicas realizadas com **OpenVSP/VSPAERO**.

Cada indivíduo representa uma geometria parametrizada por variáveis como envergadura, cordas, sweep e incidência. O algoritmo realiza seleção, crossover, mutação e elitismo, enquanto o OpenVSP gera as geometrias e executa análises aerodinâmicas. A função de fitness combina eficiência aerodinâmica (`CL/CD`) e desempenho de decolagem, e o projeto inclui execução paralela das avaliações, geração de modelos `.vsp3`, logs e acompanhamento da evolução ao longo das gerações.

**Tecnologias:** Python · OpenVSP · VSPAERO · NumPy · Pandas · Matplotlib · Multiprocessing · Algoritmos Genéticos

---

### Data Science

#### [Análise dos Preços do Airbnb em Roma](https://colab.research.google.com/github/thobiasgd/portfolio/blob/main/analise_airbnb_roma.ipynb)

Análise exploratória dos dados públicos do **Inside Airbnb** para investigar o mercado de hospedagens da cidade de Roma e extrair padrões relacionados a preço, localização e características dos anúncios.

O notebook realiza preparação e exploração do dataset `listings.csv`, análise das distribuições, tratamento e investigação de outliers com histogramas e boxplots, comparação de preços, análise de correlação entre variáveis e exploração geográfica dos anúncios. A visualização espacial utiliza **Folium** para representar imóveis e preços diretamente no mapa de Roma.

**Tecnologias:** Python · Pandas · Matplotlib · Seaborn · Folium · Google Colab
