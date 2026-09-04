"""
====================================================================
 AQUÁRIO OTIMIZADO COM REDE NEURAL (MLP / MULTI-LAYER PERCEPTRON)
====================================================================
Objetivo: treinar uma rede neural (um Perceptron Multicamadas, MLP)
para APRENDER a relação entre o Volume de água desejado e as dimensões
(Comprimento, Largura, Altura) do aquário que gastam o MÍNIMO de vidro.

Diferente dos programas 1 e 2 (que usam Algoritmo Genético para
BUSCAR a solução a cada execução), aqui a rede neural aprende, durante
o TREINAMENTO, o padrão geral "Volume -> Dimensões ótimas". Depois de
treinada, ela calcula a resposta instantaneamente (um simples "forward
pass"), sem precisar rodar nenhuma otimização de novo.

Modelo físico (aquário sem tampa, formato de caixa retangular):
    Volume:        V = L * W * H
    Área de vidro: S = L*W + 2*L*H + 2*W*H

DADOS DE TREINO: como sabemos, por cálculo (Lagrange), que a forma que
minimiza vidro para um volume V é sempre:
        L = W = (2V)^(1/3)         H = V / (L*W)
geramos milhares de exemplos (V -> L,W,H) usando essa fórmula e
treinamos a rede para reproduzir esse mapeamento. Ou seja, a rede
neural aprende, por exemplos, a "redescobrir" sozinha (via gradiente
descendente / backpropagation) a regra ótima de economia de vidro —
sem que ela receba a fórmula explicitamente, só os pares
(entrada = volume, saída = dimensões ótimas).

ARQUITETURA DA REDE (implementada do zero, apenas com numpy):
    Entrada (1 neurônio: Volume)
        -> Camada oculta 1 (16 perceptrons, ativação ReLU)
        -> Camada oculta 2 (16 perceptrons, ativação ReLU)
        -> Saída (3 neurônios lineares: L, W, H)
====================================================================
"""

import numpy as np
import sys

np.random.seed(42)

# --------------------------------------------------------------
# 1. GERAÇÃO DO DATASET DE TREINO
# --------------------------------------------------------------
def dimensoes_otimas(V):
    """Fórmula analítica (Lagrange) do aquário que minimiza vidro para volume V."""
    L = W = (2 * V) ** (1 / 3)
    H = V / (L * W)
    return L, W, H


def area_vidro(L, W, H):
    return L * W + 2 * L * H + 2 * W * H


def gerar_dataset(n_amostras=4000, v_min_litros=1, v_max_litros=2000):
    """Gera pares (Volume -> L, W, H ótimos) cobrindo uma ampla faixa de volumes."""
    volumes_litros = np.random.uniform(v_min_litros, v_max_litros, n_amostras)
    volumes_cm3 = volumes_litros * 1000.0

    X = volumes_cm3.reshape(-1, 1)
    Y = np.array([dimensoes_otimas(v) for v in volumes_cm3])
    return X, Y


# --------------------------------------------------------------
# 2. NORMALIZAÇÃO (essencial para a rede treinar bem)
# --------------------------------------------------------------
class Normalizador:
    """Padroniza dados para média 0 e desvio-padrão 1 (z-score)."""
    def __init__(self, dados):
        self.media = dados.mean(axis=0)
        self.desvio = dados.std(axis=0) + 1e-8

    def transformar(self, dados):
        return (dados - self.media) / self.desvio

    def inverter(self, dados_norm):
        return dados_norm * self.desvio + self.media


# --------------------------------------------------------------
# 3. REDE NEURAL (MLP) IMPLEMENTADA DO ZERO
# --------------------------------------------------------------
class RedeNeuralMLP:
    """
    Perceptron multicamadas simples:
        entrada(1) -> oculta1(16, ReLU) -> oculta2(16, ReLU) -> saida(3, linear)
    Treinada via backpropagation + gradiente descendente (Adam).
    """

    def __init__(self, tam_entrada=1, tam_oculta1=16, tam_oculta2=16, tam_saida=3,
                 taxa_aprendizado=0.01):
        # Inicialização "He" (boa para ReLU)
        self.W1 = np.random.randn(tam_entrada, tam_oculta1) * np.sqrt(2 / tam_entrada)
        self.b1 = np.zeros((1, tam_oculta1))
        self.W2 = np.random.randn(tam_oculta1, tam_oculta2) * np.sqrt(2 / tam_oculta1)
        self.b2 = np.zeros((1, tam_oculta2))
        self.W3 = np.random.randn(tam_oculta2, tam_saida) * np.sqrt(2 / tam_oculta2)
        self.b3 = np.zeros((1, tam_saida))

        self.lr = taxa_aprendizado

        # Estado do otimizador Adam (para cada parâmetro)
        self._adam_estado = {}
        for nome in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            param = getattr(self, nome)
            self._adam_estado[nome] = {"m": np.zeros_like(param), "v": np.zeros_like(param), "t": 0}

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivada(z):
        return (z > 0).astype(float)

    def forward(self, X):
        """Propagação direta: calcula a saída da rede para as entradas X."""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3   # saída linear (regressão)
        return self.z3

    def backward(self, X, Y, saida):
        """Backpropagation: calcula os gradientes de todos os pesos/vieses."""
        n = X.shape[0]

        # Erro na saída (derivada do MSE)
        dz3 = (saida - Y) * (2 / n)
        dW3 = self.a2.T @ dz3
        db3 = dz3.sum(axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_derivada(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivada(self.z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def _passo_adam(self, nome, grad, beta1=0.9, beta2=0.999, eps=1e-8):
        """Atualiza um parâmetro usando o otimizador Adam."""
        estado = self._adam_estado[nome]
        estado["t"] += 1
        estado["m"] = beta1 * estado["m"] + (1 - beta1) * grad
        estado["v"] = beta2 * estado["v"] + (1 - beta2) * (grad ** 2)

        m_corrigido = estado["m"] / (1 - beta1 ** estado["t"])
        v_corrigido = estado["v"] / (1 - beta2 ** estado["t"])

        param = getattr(self, nome)
        param -= self.lr * m_corrigido / (np.sqrt(v_corrigido) + eps)
        setattr(self, nome, param)

    def treinar(self, X, Y, epocas=2000, tam_lote=64, verbose=True):
        n = X.shape[0]
        for epoca in range(epocas):
            indices = np.random.permutation(n)
            X_emb = X[indices]
            Y_emb = Y[indices]

            perda_epoca = 0.0
            for inicio in range(0, n, tam_lote):
                fim = inicio + tam_lote
                X_lote = X_emb[inicio:fim]
                Y_lote = Y_emb[inicio:fim]

                saida = self.forward(X_lote)
                perda = np.mean((saida - Y_lote) ** 2)
                perda_epoca += perda * len(X_lote)

                gradientes = self.backward(X_lote, Y_lote, saida)
                for nome, grad in gradientes.items():
                    self._passo_adam(nome, grad)

            if verbose and (epoca % 200 == 0 or epoca == epocas - 1):
                print(f"Época {epoca:4d} | perda (MSE, dados normalizados) = {perda_epoca / n:.6f}")

    def prever(self, X):
        return self.forward(X)


# --------------------------------------------------------------
# 4. TREINAMENTO E AVALIAÇÃO
# --------------------------------------------------------------
def treinar_rede():
    X, Y = gerar_dataset()

    norm_X = Normalizador(X)
    norm_Y = Normalizador(Y)
    X_norm = norm_X.transformar(X)
    Y_norm = norm_Y.transformar(Y)

    rede = RedeNeuralMLP(tam_entrada=1, tam_oculta1=16, tam_oculta2=16, tam_saida=3,
                          taxa_aprendizado=0.01)

    print("=== Treinando a rede neural (MLP) ===")
    rede.treinar(X_norm, Y_norm, epocas=2000, tam_lote=64)

    return rede, norm_X, norm_Y


def prever_dimensoes(rede, norm_X, norm_Y, volume_litros):
    volume_cm3 = volume_litros * 1000.0
    X_entrada = np.array([[volume_cm3]])
    X_entrada_norm = norm_X.transformar(X_entrada)

    saida_norm = rede.prever(X_entrada_norm)
    L, W, H = norm_Y.inverter(saida_norm)[0]
    return L, W, H


# --------------------------------------------------------------
# 5. PROGRAMA PRINCIPAL
# --------------------------------------------------------------
def main():
    rede, norm_X, norm_Y = treinar_rede()

    if len(sys.argv) > 1:
        volume_litros = float(sys.argv[1])
    else:
        volume_litros = float(input("\nDigite o volume de água desejado (em litros): "))

    L, W, H = prever_dimensoes(rede, norm_X, norm_Y, volume_litros)
    V = L * W * H
    S = area_vidro(L, W, H)

    L_opt, W_opt, H_opt = dimensoes_otimas(volume_litros * 1000.0)
    S_opt = area_vidro(L_opt, W_opt, H_opt)

    print(f"\n=== Previsão da rede neural para {volume_litros:.1f} L ===\n")
    print("---------------------------------------------------")
    print("RESULTADO DA REDE NEURAL (MLP)")
    print("---------------------------------------------------")
    print(f"Comprimento (L): {L:8.2f} cm")
    print(f"Largura     (W): {W:8.2f} cm")
    print(f"Altura      (H): {H:8.2f} cm")
    print(f"Volume obtido  : {V:8.1f} cm³  (alvo: {volume_litros*1000:.1f} cm³)")
    print(f"Vidro gasto    : {S:8.2f} cm²")
    print("\n---------------------------------------------------")
    print("SOLUÇÃO ANALÍTICA (ótimo teórico, via Lagrange)")
    print("---------------------------------------------------")
    print(f"L=W={L_opt:.2f} cm | H={H_opt:.2f} cm | Vidro={S_opt:.2f} cm²")
    print(f"\nDiferença da rede para o ótimo teórico: {abs(S - S_opt):.2f} cm² "
          f"({100*abs(S - S_opt)/S_opt:.2f}%)")


if __name__ == "__main__":
    main()
