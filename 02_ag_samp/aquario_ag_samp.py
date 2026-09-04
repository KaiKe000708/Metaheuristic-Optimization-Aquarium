"""
====================================================================
 AQUÁRIO OTIMIZADO COM AG + SAMP
 (Self-Adaptive Mutation Parameters — Parâmetros de Mutação
  Autoadaptativos)
====================================================================
Igual ao primeiro programa (AG clássico), mas aqui cada indivíduo NÃO
usa uma taxa/intensidade de mutação fixa e global. Em vez disso, cada
indivíduo carrega seu PRÓPRIO parâmetro de mutação (sigma), que evolui
JUNTO com a solução — essa é a técnica de "mutação autoadaptativa",
clássica de Estratégias Evolutivas (Evolution Strategies) e incorporada
aqui a um AG.

Indivíduo = [L, W, sigma]
    - L, W  : dimensões do aquário (a altura H é sempre derivada do
              volume exato, H = V/(L*W), igual ao programa 1)
    - sigma : "tamanho do passo" de mutação daquele indivíduo (o quanto
              ele costuma perturbar L e W)

Por que isso ajuda a economizar mais vidro?
    - No início da busca, sigmas grandes ajudam a explorar bastante o
      espaço de soluções (evita ficar preso em ótimos locais ruins).
    - Conforme a população converge, indivíduos com sigma pequeno
      (mutações mais finas) tendem a gerar filhos melhores e sobrevivem
      mais — a própria seleção natural "afina" a busca sem precisarmos
      programar um cronograma de mutação manualmente.

Regra clássica de autoadaptação (Schwefel/Bäck), aplicada a cada gene
de mutação:
        sigma' = sigma * exp(tau * N(0,1))
        gene'  = gene  + sigma' * N(0,1)

Modelo físico do aquário (mesmo dos outros programas):
    Volume:        V = L * W * H
    Área de vidro: S(L, W) = L*W + 2V/L + 2V/W   (com H = V/(L*W))
====================================================================
"""

import random
import math
import sys

# --------------------------------------------------------------
# 1. PARÂMETROS DO ALGORITMO
# --------------------------------------------------------------
TAMANHO_POPULACAO = 100
NUM_GERACOES = 200
TAXA_CRUZAMENTO = 0.85
TAMANHO_TORNEIO = 4
ELITISMO = 2
LIMITE_DIM_CM = (2.0, 500.0)      # faixa plausível para L e W
SIGMA_INICIAL = (5.0, 15.0)       # faixa inicial aleatória do sigma de cada indivíduo
SIGMA_MINIMO = 0.01               # sigma não pode colapsar a zero (perderia diversidade)
TAU = 1.0 / math.sqrt(2 * 2)      # taxa de aprendizado da autoadaptação (2 = nº de genes)


# --------------------------------------------------------------
# 2. FUNÇÕES DO PROBLEMA (física do aquário)
# --------------------------------------------------------------
def altura_necessaria(L, W, volume_alvo):
    return volume_alvo / (L * W)


def area_vidro_LW(L, W, volume_alvo):
    return L * W + 2 * volume_alvo / L + 2 * volume_alvo / W


def solucao_analitica(volume_alvo):
    L = W = (2 * volume_alvo) ** (1 / 3)
    H = volume_alvo / (L * W)
    return L, W, H


# --------------------------------------------------------------
# 3. OPERADORES GENÉTICOS  (indivíduo = [L, W, sigma])
# --------------------------------------------------------------
def gerar_individuo():
    L = random.uniform(*LIMITE_DIM_CM)
    W = random.uniform(*LIMITE_DIM_CM)
    sigma = random.uniform(*SIGMA_INICIAL)
    return [L, W, sigma]


def gerar_populacao_inicial():
    return [gerar_individuo() for _ in range(TAMANHO_POPULACAO)]


def custo(individuo, volume_alvo):
    L, W, _ = individuo
    return area_vidro_LW(L, W, volume_alvo)


def selecao_torneio(populacao, volume_alvo):
    competidores = random.sample(populacao, TAMANHO_TORNEIO)
    competidores.sort(key=lambda ind: custo(ind, volume_alvo))
    return competidores[0]


def cruzamento(pai1, pai2):
    """Cruzamento aritmético também aplicado ao gene sigma."""
    if random.random() > TAXA_CRUZAMENTO:
        return pai1[:], pai2[:]
    alpha = random.random()
    filho1 = [alpha * g1 + (1 - alpha) * g2 for g1, g2 in zip(pai1, pai2)]
    filho2 = [alpha * g2 + (1 - alpha) * g1 for g1, g2 in zip(pai1, pai2)]
    return filho1, filho2


def mutacao_autoadaptativa(individuo):
    """
    Núcleo do SAMP:
    1) o próprio sigma sofre mutação log-normal (sigma nunca é fixo);
    2) L e W são então perturbados usando o NOVO sigma daquele indivíduo.
    """
    L, W, sigma = individuo

    # 1) sigma evolui primeiro
    novo_sigma = sigma * math.exp(TAU * random.gauss(0, 1))
    novo_sigma = max(novo_sigma, SIGMA_MINIMO)

    # 2) L e W mutam usando o novo sigma (passo de mutação "pessoal")
    novo_L = L + novo_sigma * random.gauss(0, 1)
    novo_W = W + novo_sigma * random.gauss(0, 1)

    novo_L = min(max(novo_L, LIMITE_DIM_CM[0]), LIMITE_DIM_CM[1])
    novo_W = min(max(novo_W, LIMITE_DIM_CM[0]), LIMITE_DIM_CM[1])

    return [novo_L, novo_W, novo_sigma]


# --------------------------------------------------------------
# 4. LOOP EVOLUTIVO PRINCIPAL
# --------------------------------------------------------------
def algoritmo_genetico_samp(volume_alvo, verbose=True):
    populacao = gerar_populacao_inicial()

    for geracao in range(NUM_GERACOES):
        populacao.sort(key=lambda ind: custo(ind, volume_alvo))
        nova_populacao = populacao[:ELITISMO]

        while len(nova_populacao) < TAMANHO_POPULACAO:
            pai1 = selecao_torneio(populacao, volume_alvo)
            pai2 = selecao_torneio(populacao, volume_alvo)
            filho1, filho2 = cruzamento(pai1, pai2)
            nova_populacao.append(mutacao_autoadaptativa(filho1))
            if len(nova_populacao) < TAMANHO_POPULACAO:
                nova_populacao.append(mutacao_autoadaptativa(filho2))

        populacao = nova_populacao

        if verbose and (geracao % 40 == 0 or geracao == NUM_GERACOES - 1):
            melhor = min(populacao, key=lambda ind: custo(ind, volume_alvo))
            L, W, sigma = melhor
            H = altura_necessaria(L, W, volume_alvo)
            sigma_medio = sum(ind[2] for ind in populacao) / len(populacao)
            print(f"Geração {geracao:3d} | vidro={area_vidro_LW(L, W, volume_alvo):9.2f} cm² "
                  f"| L={L:6.2f} W={W:6.2f} H={H:6.2f} "
                  f"| sigma(melhor)={sigma:5.3f} | sigma(médio pop.)={sigma_medio:5.3f}")

    melhor_final = min(populacao, key=lambda ind: custo(ind, volume_alvo))
    return melhor_final


# --------------------------------------------------------------
# 5. PROGRAMA PRINCIPAL
# --------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        volume_litros = float(sys.argv[1])
    else:
        volume_litros = float(input("Digite o volume de água desejado (em litros): "))

    volume_cm3 = volume_litros * 1000.0

    print(f"\n=== Otimizando aquário (AG + SAMP) para {volume_litros:.1f} L "
          f"({volume_cm3:.0f} cm³) ===\n")
    L, W, sigma = algoritmo_genetico_samp(volume_cm3)
    H = altura_necessaria(L, W, volume_cm3)
    S = area_vidro_LW(L, W, volume_cm3)
    V = L * W * H

    L_opt, W_opt, H_opt = solucao_analitica(volume_cm3)
    S_opt = area_vidro_LW(L_opt, W_opt, volume_cm3)

    print("\n---------------------------------------------------")
    print("RESULTADO DO AG COM MUTAÇÃO AUTOADAPTATIVA (SAMP)")
    print("---------------------------------------------------")
    print(f"Comprimento (L): {L:8.2f} cm")
    print(f"Largura     (W): {W:8.2f} cm")
    print(f"Altura      (H): {H:8.2f} cm")
    print(f"Sigma final do melhor indivíduo: {sigma:.4f}  (convergiu = passo bem pequeno)")
    print(f"Volume obtido  : {V:8.1f} cm³  (alvo: {volume_cm3:.1f} cm³)")
    print(f"Vidro gasto    : {S:8.2f} cm²")
    print("\n---------------------------------------------------")
    print("SOLUÇÃO ANALÍTICA (ótimo teórico, via Lagrange)")
    print("---------------------------------------------------")
    print(f"L=W={L_opt:.2f} cm | H={H_opt:.2f} cm | Vidro={S_opt:.2f} cm²")
    print(f"\nDiferença do AG+SAMP para o ótimo teórico: {abs(S - S_opt):.2f} cm² "
          f"({100*abs(S - S_opt)/S_opt:.2f}%)")


if __name__ == "__main__":
    main()
