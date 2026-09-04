"""
====================================================================
 AQUÁRIO OTIMIZADO COM ALGORITMO GENÉTICO (AG)
====================================================================
Objetivo: dado um VOLUME de água desejado (entrada do usuário), o AG
descobre o Comprimento (L), a Largura (W) e a Altura (H) do aquário
que MINIMIZAM a quantidade de vidro gasto, respeitando exatamente o
volume pedido.

Modelo físico (aquário sem tampa, formato de caixa retangular):
    Volume:        V = L * W * H
    Área de vidro: S = L*W (fundo) + 2*L*H (frente/trás) + 2*W*H (laterais)

TRUQUE DE MODELAGEM (o que torna o AG rápido e sempre viável):
Em vez de deixar o AG "adivinhar" L, W e H livremente e depois penalizar
quem erra o volume, cada indivíduo carrega apenas os genes (L, W).
A altura H é sempre CALCULADA para satisfazer o volume exatamente:

        H = V / (L * W)

Assim, toda solução gerada já respeita o volume pedido "de graça", e o
AG foca 100% da busca em economizar vidro. Substituindo H na fórmula
da área, o custo a minimizar vira:

        S(L, W) = L*W + 2V/L + 2V/W

Solução analítica de referência (via Lagrange, só para comparação):
    L = W = (2V)^(1/3)          H = V / (L*W)
====================================================================
"""

import random
import sys

# --------------------------------------------------------------
# 1. PARÂMETROS DO ALGORITMO GENÉTICO
# --------------------------------------------------------------
TAMANHO_POPULACAO = 100
NUM_GERACOES = 200
TAXA_CRUZAMENTO = 0.85
TAXA_MUTACAO = 0.20
INTENSIDADE_MUTACAO = 0.10     # % de perturbação relativa ao valor do gene
TAMANHO_TORNEIO = 4
ELITISMO = 2                   # nº de melhores indivíduos preservados por geração
LIMITE_DIM_CM = (2.0, 500.0)   # faixa plausível para L e W, em cm


# --------------------------------------------------------------
# 2. FUNÇÕES DO PROBLEMA (física do aquário)
# --------------------------------------------------------------
def altura_necessaria(L, W, volume_alvo):
    """H sempre calculado para que o volume seja exatamente o pedido."""
    return volume_alvo / (L * W)


def area_vidro_LW(L, W, volume_alvo):
    """
    Área de vidro em função só de L e W (H já embutido via volume exato):
        S(L, W) = L*W + 2V/L + 2V/W
    Essa é a função de CUSTO que o AG minimiza (quanto menor, melhor).
    """
    return L * W + 2 * volume_alvo / L + 2 * volume_alvo / W


def solucao_analitica(volume_alvo):
    """Ótimo teórico (Lagrange) só para comparação com o AG."""
    L = W = (2 * volume_alvo) ** (1 / 3)
    H = volume_alvo / (L * W)
    return L, W, H


# --------------------------------------------------------------
# 3. OPERADORES GENÉTICOS  (indivíduo = [L, W])
# --------------------------------------------------------------
def gerar_individuo():
    return [random.uniform(*LIMITE_DIM_CM) for _ in range(2)]


def gerar_populacao_inicial():
    return [gerar_individuo() for _ in range(TAMANHO_POPULACAO)]


def selecao_torneio(populacao, volume_alvo):
    competidores = random.sample(populacao, TAMANHO_TORNEIO)
    competidores.sort(key=lambda ind: area_vidro_LW(ind[0], ind[1], volume_alvo))
    return competidores[0]


def cruzamento(pai1, pai2):
    """Cruzamento aritmético (blend crossover)."""
    if random.random() > TAXA_CRUZAMENTO:
        return pai1[:], pai2[:]
    alpha = random.random()
    filho1 = [alpha * g1 + (1 - alpha) * g2 for g1, g2 in zip(pai1, pai2)]
    filho2 = [alpha * g2 + (1 - alpha) * g1 for g1, g2 in zip(pai1, pai2)]
    return filho1, filho2


def mutacao(individuo):
    """Mutação gaussiana relativa, com clamp nos limites plausíveis."""
    novo = individuo[:]
    for i in range(len(novo)):
        if random.random() < TAXA_MUTACAO:
            desvio = novo[i] * INTENSIDADE_MUTACAO
            novo[i] += random.gauss(0, desvio)
            novo[i] = min(max(novo[i], LIMITE_DIM_CM[0]), LIMITE_DIM_CM[1])
    return novo


# --------------------------------------------------------------
# 4. LOOP EVOLUTIVO PRINCIPAL
# --------------------------------------------------------------
def algoritmo_genetico(volume_alvo, verbose=True):
    populacao = gerar_populacao_inicial()

    for geracao in range(NUM_GERACOES):
        populacao.sort(key=lambda ind: area_vidro_LW(ind[0], ind[1], volume_alvo))
        nova_populacao = populacao[:ELITISMO]

        while len(nova_populacao) < TAMANHO_POPULACAO:
            pai1 = selecao_torneio(populacao, volume_alvo)
            pai2 = selecao_torneio(populacao, volume_alvo)
            filho1, filho2 = cruzamento(pai1, pai2)
            nova_populacao.append(mutacao(filho1))
            if len(nova_populacao) < TAMANHO_POPULACAO:
                nova_populacao.append(mutacao(filho2))

        populacao = nova_populacao

        if verbose and (geracao % 40 == 0 or geracao == NUM_GERACOES - 1):
            melhor = min(populacao, key=lambda ind: area_vidro_LW(ind[0], ind[1], volume_alvo))
            L, W = melhor
            H = altura_necessaria(L, W, volume_alvo)
            print(f"Geração {geracao:3d} | vidro={area_vidro_LW(L, W, volume_alvo):9.2f} cm² "
                  f"| L={L:6.2f} W={W:6.2f} H={H:6.2f}")

    melhor_final = min(populacao, key=lambda ind: area_vidro_LW(ind[0], ind[1], volume_alvo))
    return melhor_final


# --------------------------------------------------------------
# 5. PROGRAMA PRINCIPAL
# --------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        volume_litros = float(sys.argv[1])
    else:
        volume_litros = float(input("Digite o volume de água desejado (em litros): "))

    volume_cm3 = volume_litros * 1000.0  # 1 litro = 1000 cm^3

    print(f"\n=== Otimizando aquário para {volume_litros:.1f} L ({volume_cm3:.0f} cm³) ===\n")
    L, W = algoritmo_genetico(volume_cm3)
    H = altura_necessaria(L, W, volume_cm3)
    S = area_vidro_LW(L, W, volume_cm3)
    V = L * W * H

    L_opt, W_opt, H_opt = solucao_analitica(volume_cm3)
    S_opt = area_vidro_LW(L_opt, W_opt, volume_cm3)

    print("\n---------------------------------------------------")
    print("RESULTADO DO ALGORITMO GENÉTICO")
    print("---------------------------------------------------")
    print(f"Comprimento (L): {L:8.2f} cm")
    print(f"Largura     (W): {W:8.2f} cm")
    print(f"Altura      (H): {H:8.2f} cm")
    print(f"Volume obtido  : {V:8.1f} cm³  (alvo: {volume_cm3:.1f} cm³)")
    print(f"Vidro gasto    : {S:8.2f} cm²")
    print("\n---------------------------------------------------")
    print("SOLUÇÃO ANALÍTICA (ótimo teórico, via Lagrange)")
    print("---------------------------------------------------")
    print(f"L=W={L_opt:.2f} cm | H={H_opt:.2f} cm | Vidro={S_opt:.2f} cm²")
    print(f"\nDiferença do AG para o ótimo teórico: {abs(S - S_opt):.2f} cm² "
          f"({100*abs(S - S_opt)/S_opt:.2f}%)")


if __name__ == "__main__":
    main()
