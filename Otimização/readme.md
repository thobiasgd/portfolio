# Otimização Aeronáutica com Algoritmo Genético e OpenVSP

Implementação modular de um sistema de otimização geométrica de asas utilizando **OpenVSP** como backend de análise e um **algoritmo genético** para busca evolutiva.  
O projeto foi reestruturado com foco em clareza, desempenho e reprodutibilidade, ideal para uso acadêmico e experimentação técnica.

---

## 🎯 Objetivo

Automatizar a criação, simulação e otimização de asas aeronáuticas, avaliando o desempenho aerodinâmico (CL/CD, distância de decolagem, eficiência geométrica) por meio de um processo evolutivo iterativo.

---

## 🧩 Estrutura do projeto

```
algoritmo_genetico/
├── main.py              # ponto de entrada e configuração da otimização
├── genetic_core.py      # núcleo do algoritmo genético (seleção, crossover, mutação, elitismo)
├── aircraft_model.py    # integração com OpenVSP (modelagem e simulação)
├── evaluator.py         # cálculo da função de aptidão (fitness)
└── utils.py             # logs, caminhos, gráficos e funções auxiliares
```

---

## ⚙️ Principais recursos

- **Arquitetura modular:** facilita manutenção e testes.  
- **Elitismo automático:** o melhor indivíduo é preservado a cada geração.  
- **Mutação adaptativa:** perturbações gaussianas proporcionais ao gene.  
- **Execução paralela:** simulações podem ser distribuídas em múltiplos processos.  
- **Resultados organizados:** arquivos `.vsp3`, gráficos e logs salvos em `/resultados`.  
- **Logging estruturado:** controle de execução via timestamps e níveis de log.

---

## 🧮 Dependências

- Python ≥ 3.9  
- OpenVSP (com suporte à API Python)  
- Pacotes:
  ```bash
  pip install numpy pandas matplotlib
  ```

---

## 🚀 Como executar

1. Ajuste o caminho do OpenVSP no início do projeto, se necessário:
   ```python
   import openvsp as vsp
   vsp.SetVSPAEROPath("C:/OpenVSP")
   ```

2. Configure os parâmetros em `main.py`:
   ```python
   optimizer = GeneticOptimizer(
       pop_size=10,
       mutation_rate=0.1,
       generations=10,
       logger=logger
   )
   ```

3. Execute:
   ```bash
   python main.py
   ```

O processo gera:
- Modelos `.vsp3` em `/resultados`
- Log detalhado da execução
- Gráfico de evolução (`evolucao.png`)

---

## 🧠 Estrutura conceitual

1. **Modelagem:** cada indivíduo gera uma geometria de asa com base nos genes (envergadura, corda, sweep, etc.).  
2. **Simulação:** o OpenVSP executa o `VSPAEROSweep`, gerando o polar aerodinâmico.  
3. **Avaliação:** a função de fitness considera CL/CD e desempenho de decolagem.  
4. **Evolução:** operadores genéticos geram novas combinações até convergir.

---

## 📊 Exemplo de saída (gráfico de convergência)

O arquivo `evolucao.png` mostra a melhoria progressiva do desempenho médio por geração.

---

## 🧱 Boas práticas

- Mantenha versões compatíveis do OpenVSP documentadas.  
- Use ambientes virtuais (`venv` ou `conda`).  
- Execute com CPU dedicada (simulações são intensivas).  
- Considere normalizar métricas para evitar dominância de variáveis.  
- Evite rodar o multiprocessing em modo debug (incompatível com OpenVSP em alguns sistemas).

---

## 🔬 Próximos passos sugeridos

- Implementar fitness multiobjetivo com pesos configuráveis.  
- Adicionar histórico CSV de cada geração.  
- Integrar frameworks como `DEAP` ou `pymoo` para comparação de estratégias genéticas.  
- Criar `Dockerfile` com OpenVSP pré-instalado para garantir portabilidade.  
- Implementar critério de parada baseado em estagnação (x gerações sem melhora).

---

## 📜 Licença

Uso livre para fins acadêmicos e de pesquisa, mediante citação do autor original.

---

> Projeto desenvolvido como evolução de um TCC em engenharia, reescrito para uso profissional e reprodutível em ambiente de otimização computacional.
