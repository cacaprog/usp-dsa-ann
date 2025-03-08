# Artificial Neural Network class


# Deep Learning I

Era Negra das Redes Artificiais

Livros: Introduction to Deep Learning - Springer
Deep Learning with Python

### XOR nas redes neurais
Linear separability and the XOR problem
As redes neurais clássicas não conseguiam separar bolinhas brancas das pretas com base na posição

Rede Neural Multicamada
Indentifica relações não lineares

### Estrutura da Rede
Camadas = grupo de neurônios em um estágio do processo

### Função perda
Predicted x Actual
Permite verificar o quão assertivo é determinada previsão

Depende do tipo de variável (Target) que estamos observando.

- Variáveis categóricas
- Variáveis contínuas
- Variáveis discretas

Pra variáveis quanti
MSE - mean square error
AME - Erro absoluto médio

Pra classificação (0, 1)
Binary cross-entropy
Categorical cross-entropy

### Viés vs. Variância
Grande problema da estatística
Importância da generalização
Teoria do mapa - eu busco um problema específico mas em algo que pode ser generalizado

Ausência de viés - na média vc acerta
Redução de Variância - tiro ao alvo

Overfitting - o modelo memorizou nos dados de treinamento
Underfitting - o modelo não pegou nenhuma especificidade dos dados, não tem capacidade de perceber isso
Fitting - a curva passa próximo aos dados - esse é o cara ideal

### Redução do overfitting
Quanto mais parâmetro, menor é a linearidade do modelo

Regularização
- Dropout: redução aleatória dos neurônios para uma camada durante a iteração, Introduz um ‘barulho’ na rede
- L1: ajuste dos pesos. Leva até zero
- L2: igual L1, porém leva proximo a zero

Early Stopping
Sem novas melhorias? Pode parar

### Batch
Treinamento em lotes
Reduz o custo computacional

### Hiperparâmetro
Qualquer número utilizado pela rede que não é aprendido
- Learning rate
- Batch size
- Epochs
- Função de ativação

### AutoML
- AutoKeras, H20
- Transfer Learning
- GridSearch

### Cross validation
- K fold

A idéia é utilizá-lo para o tuning.
Fica dentro da amostra de treinamento ainda, um substrato é separado para essa validação.
Mas ela fica ainda dentro do treinamento.
Não é recomendável pra achar a acurácia.


### Mínimo local e ponto de sela
O objetivo é chegar no valor mais baixo de perda.
O learning rate vai definir a velocidade do gradiente buscar o mínimo. E dependendo disso, não chegamos no valor mais baixo.

Conforme vamos testando, podemos, talvez, encontrar um valor menor ainda.

O menor local pode dizer que é o menor ponto, dentro de um recorte. Se olharmos o todo, talvez não seja verdade.

### Otimizadores
- Adagrad (gradiente descendo melhorado)
- Adam (esse é o mais utilizado, mais genérico)
- RMSPROP (séries temporais)


### Colab
Upload `Aula_2_2_1.ipynb`

Vai comparar o valor entre rede neural e decision tree.
Lembrando que o objetivo é didático.

Com o Gemini fica bem mais interessante.

Dataset: California Housing
`fetch_california_housing`


—
O MSE encontrado não significa nada. É preciso ter um benchmark para efeito de comparação válida.

Geralmente rodamos diversos modelos, então teremos como tomará-los entre eles. O MSE sozinho nada significa.

Rede Neural
`hidden_layer_size=(50, 30, 1)` - 3 camadas e quantos neurônios em cada uma delas

activation=“identity” - stivação linear
solver=“adam” - otimizador, o adam é um ótimo default, exceto para séries temporais, que o professor começa com outro

max_iter = 500 : são as épocas

O modelo rede neural se saiu melhor do que o decision tree.

—
Aula_2_2_2.ipynb
Breast cancer

Vamos comparar a rede neural contra  uma regresso linear

Vamos classificar em benigno ou maligno.

Pré processamento
1. Excluir os nan values
2. Transformar variável `class` em números (0 ou 1)


# Deep Learning II

### Redes neurais recorrentes
Livros 
- Neural Networks and Deep Learning - 
- Introduction to Deep Learning

Alguns problemas importantes:
- Dados de texto
- Séries temporais
- Assistir um filme

A Sequência importa
O gato perseguiu o rato
O rato perseguiu o gato

RNN - não se ‘importa’com a ordem.
Ela constrói uma distribuição de probabilidade

Rede Neural com memória
Guardar o que é importante e fazer previsões mais acuradas

### Simple recurrent neural network
Rede de Elman


### Backpropragation
- Tudo se afeta ao mesmo tempo
- O passado influencia, os pesos são ajustados, mas quanto mais longe, mais difícil de saber o tamanho dessa influência

Dissipação do gradiente (Vanishing gradient)
- Multiplicação por valores maiores que 1: sobe, distancia do zero
- Multiplicação por valores menores que 1: os números se aproximam de zero


Ocorre em qualquer rede - porém com mais frequência em RNN

### Como resolver?

**Backpropagation truncada**
- Solução arbitrária
- Para a avaliação de pesos até certo ponto
- Custo computacional

**Resolver o Vanishing Gradient**
- Inicialização de matriz de peso
- Função de ativação ReLU

**Clipping do Gradiente (mais famosa)**
- Solução possível para dissipação e explosão
- Define um valor limite definido nos gradientes

Nas arquiteturas anteriores (Elman), tentava guardar todo o passado - tipo as mulheres, que não esquecem nada.
As novas arquiteturas consideram somente o que é importante, um recorte do todo.
 

### LSTM
Long short-term memory
Será que devemos guardar uma informação?

Algumas funções importantes
**TANH** - tangente hiperbólica
Resultado entre -1 e 1 (negativo, neutro, positivo)

**Sigmoid**
Resultado entre 0 e 1 (sim ou não)

Estado da célula: memória longa
Forget Gate - portão do esquecimento - quanto lembrar?
Input gate - quanto manter dos inputs?
Output gate - o que do estado da célula e do hidden state será utilizado como resultado?


**GRU - Gate Recurrent Unit**
Resolve problema da dissipação do gradiente
Baseada em portões:
- Reset Gate
- Update Gate

Guarda dependência longa


### Transformers
- Maior parte de PLN era feito com RNN
- Attention is all you need (artigo importante)
- Uso de RNN => perde informação conforme se distancia do início de uma série
- Contexto é essencial em PLN (processamento de linguagem natural)

Codificador
Decodificador

O que é mais rápido para descobrir uma informação: ler o livro inteiro ou procurar no índice?
Os transformers criam estes índices
