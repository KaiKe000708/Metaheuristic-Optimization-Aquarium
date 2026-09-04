# Metaheuristic-Optimization-Aquarium

Comparação prática de três abordagens de Inteligência Artificial/Otimização aplicadas ao mesmo problema: **dado um volume de água desejado, encontrar as dimensões de um aquário (Comprimento, Largura, Altura) que minimizem a quantidade de vidro gasto**.

## 📐 O problema

Aquário em formato de caixa retangular, sem tampa:

```
Volume:        V = L * W * H
Área de vidro: S = L*W (fundo) + 2*L*H (frente/trás) + 2*W*H (laterais)
```

O objetivo é minimizar `S`, respeitando exatamente o volume `V` pedido pelo usuário.

Existe uma **solução analítica exata** (via Multiplicadores de Lagrange), usada neste repositório como referência (ground truth) para avaliar o quão perto cada técnica chega do ótimo real:

```
L = W = (2V)^(1/3)        H = V / (L * W)
```

## 🧪 Três abordagens implementadas

| # | Abordagem | Como funciona |
|---|---|---|
| 01 | **Algoritmo Genético (AG)** | Busca evolutiva clássica: população de soluções (L, W) evolui por seleção por torneio, cruzamento aritmético e mutação gaussiana, ao longo de gerações, até convergir para o mínimo de vidro. |
| 02 | **AG + SAMP** (Self-Adaptive Mutation Parameters) | Evolução do algoritmo 01: cada indivíduo carrega seu próprio parâmetro de mutação (sigma), que evolui junto com a solução (técnica clássica de Estratégias Evolutivas). Isso permite exploração ampla no início e refinamento fino ao final, sem precisar programar um cronograma manual de mutação. |
| 03 | **Rede Neural (MLP)** | Em vez de buscar a solução a cada execução, a rede é **treinada uma vez** sobre milhares de exemplos (Volume → Dimensões ótimas, gerados pela fórmula analítica) e aprende o padrão geral. Depois de treinada, responde instantaneamente a qualquer volume novo, sem rodar otimização. Implementada do zero com `numpy` (forward/backward propagation manual, otimizador Adam). |

## 📊 Resultado comparativo (exemplo: 200 litros)

| Abordagem | L (cm) | W (cm) | H (cm) | Vidro gasto (cm²) | Erro vs. ótimo teórico |
|---|---|---|---|---|---|
| Algoritmo Genético | 73.68 | 73.68 | 36.84 | 16.286,51 | 0,00% |
| AG + SAMP | 73.68 | 73.68 | 36.84 | 16.286,51 | 0,00% |
| Rede Neural (MLP) | 72.56 | 72.56 | 36.28 | 15.795,74 | ~1,3%* |
| **Ótimo teórico (Lagrange)** | 73.68 | 73.68 | 36.84 | 16.286,51 | — |

\* *A rede neural aproxima o padrão aprendido em vez de recalcular a otimização exata a cada execução — por isso não converge para 0% de erro como os algoritmos genéticos, mas responde de forma praticamente instantânea, sem precisar rodar um processo evolutivo novamente.*

### 💡 Principal conclusão

Os dois Algoritmos Genéticos **convergem exatamente para o ótimo teórico** (0% de erro), pois recalculam a otimização do zero a cada execução. Já a Rede Neural troca essa precisão exata por **velocidade de resposta**: uma vez treinada, ela prevê a solução instantaneamente para qualquer volume, sem rodar nenhuma busca — um trade-off clássico entre exatidão e velocidade em Inteligência Artificial.

---

## 📁 Estrutura do repositório

```
Metaheuristic-Optimization-Aquarium/
├── 01_algoritmo_genetico/
│   └── aquario_ag.py
├── 02_ag_samp/
│   └── aquario_ag_samp.py
├── 03_rede_neural/
│   └── aquario_rede_neural.py
├── requirements.txt
├── LICENSE
└── README.md
```

## ⚙️ Instalação

```bash
git clone https://github.com/KaiKe000708/Metaheuristic-Optimization-Aquarium.git
cd Metaheuristic-Optimization-Aquarium
pip install -r requirements.txt
```

## ▶️ Como usar

Cada script pede o volume desejado (em litros) e pode ser executado de duas formas:

```bash
# Interativo (pede o volume durante a execução)
python 01_algoritmo_genetico/aquario_ag.py

# Direto, passando o volume como argumento
python 01_algoritmo_genetico/aquario_ag.py 200
```

O mesmo vale para `02_ag_samp/aquario_ag_samp.py` e `03_rede_neural/aquario_rede_neural.py`.

---

## 🖥️ Requisitos

- Python 3.9 ou superior
- `numpy` (apenas para o script 03 — os dois algoritmos genéticos usam só a biblioteca padrão do Python)

---

## 📄 Licença

Este projeto está sob a licença MIT — veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## ✏️ Notas

Projeto desenvolvido para explorar, na prática, diferentes estratégias de otimização e aprendizado de máquina sobre um mesmo problema de engenharia, comparando o trade-off entre exatidão (algoritmos genéticos) e velocidade de inferência (rede neural treinada).
