# Sombras Dinâmicas em Tempo Real com Stenciled Shadow Volumes

Implementação, em Python e OpenGL 3.3, de sombras dinâmicas em tempo real com **Stenciled Shadow Volumes** na variante **z-fail** ("Carmack's Reverse"), com suporte a até 8 luzes pontuais em movimento, e um benchmark que mede o custo do método em função do número de luzes e da complexidade da malha.

Trabalho da disciplina de **Computação Gráfica** (Eixo 7 — sistemas legados e retrocomputação).

## Autores

| Nome | RA |
|---|---|
| João Victor Camargo | 113803 |
| Guilherme Belém | 114031 |

**Curso de Engenharia de Computação**
**Fundação Hermínio Ometto (FHO/UNIARARAS) — Araras, SP**

## O que o projeto faz

- Calcula, a cada quadro, a **silhueta** de cada objeto em relação a cada luz, **extruda** o volume de sombra até o infinito (coordenadas homogêneas com `w = 0`) e o fecha com **tampas**.
- Marca o **stencil buffer** com o algoritmo **z-fail**, que funciona mesmo com a câmera dentro de uma sombra.
- Soma a iluminação de cada luz (Phong, com atenuação) só fora das sombras, em **múltiplos passes**.
- Oferece uma demonstração interativa com painel de botões e quatro cenas, e um **benchmark** que gera os dados do artigo.

Sem motor gráfico pronto: toda a iluminação e as sombras são implementadas no código, para poder medir o tempo de cada etapa.

## Arquivos

| Arquivo | Para que serve |
|---|---|
| `main.py` | Demonstração interativa (janela, câmera, cenas, painel de controle) |
| `hud.py` | Painel de botões desenhado na tela (usa Pillow) |
| `shadow.py` | Silhueta, extrusão e montagem do volume de sombra (`ShadowCaster`) |
| `benchmark.py` | Benchmark: FPS e custo por etapa, com e sem sombras |
| `plot_benchmark.py` | Gera gráficos, tabela LaTeX e resumo a partir do CSV do benchmark |
| `main.tex` | Artigo no formato IEEE (LaTeX) |
| `resultados/` | Saída do benchmark (criada ao rodar): CSV, informações do ambiente, capturas e gráficos |

## Requisitos

- Python 3 (testado com 3.13.1)
- GPU e driver com **OpenGL 3.3 ou superior** e stencil buffer de 8 bits
- Bibliotecas Python: `moderngl`, `glfw`, `numpy`, `trimesh`, `PyOpenGL`, `pillow` e, só para os gráficos, `matplotlib`

## Instalação

No Windows (PowerShell), dentro da pasta do projeto:

```
python -m venv venv
venv\Scripts\activate
pip install moderngl glfw numpy trimesh PyOpenGL pillow matplotlib
```

No Linux ou macOS, ative o ambiente com `source venv/bin/activate`.

## Demonstração interativa

```
python main.py
```

Opções (todas opcionais):

| Opção | Efeito |
|---|---|
| `--scene padrao\|predios\|bonecos\|cidade` | Cena inicial |
| `--lights N` | Número inicial de luzes (1 a 8) |
| `--shot arquivo.png` | Salva uma captura de tela e fecha |

### Cenas

- **Padrão:** chão, um cubo e uma esfera.
- **Prédios:** sete blocos de alturas e cores diferentes.
- **Bonecos:** dois personagens feitos de blocos e uma esfera para a cabeça.
- **Cidade:** quatro prédios altos, duas lojas, dois bonecos, quatro postes e dois carros.

### Controles

| Controle | Ação |
|---|---|
| `W` `A` `S` `D` | Move a câmera |
| Mouse | Gira a câmera |
| `TAB` | Libera/prende o mouse (para clicar nos botões do painel) |
| `1` a `8` | Número de luzes |
| `C` | Troca de cena |
| `X` | Liga/desliga as sombras (útil para comparar custo) |
| `V` | Mostra o volume de sombra da luz 0 (magenta translúcido) |
| `E` | Mostra as arestas de silhueta da luz 0 (amarelo) |
| `P` | Pausa o movimento das luzes |
| `L` | Mostra uma bolinha em cada luz |
| `Esc` | Sai |

Os botões do painel fazem o mesmo que as teclas. Para clicar neles, aperte `TAB` antes.

## Benchmark

O benchmark usa uma cena com um chão (que só recebe sombra) e três esferas (que projetam sombra), variando:

- **Número de luzes:** 1, 2, 4 e 8
- **Complexidade:** esferas com 20 a 20.480 triângulos cada (60 a 61.440 na cena)
- **Modo:** com sombras (z-fail) e sem sombras (só a iluminação multi-pass)

Para cada configuração mede o FPS, o tempo de quadro, o tempo de CPU por etapa (silhueta, extrusão, envio ao buffer) e o tempo de GPU por passe (ambiente, stencil, luz), com *timer queries* do OpenGL. A câmera e as luzes seguem trajetórias determinísticas.

```
python benchmark.py --quick     # teste rápido (cerca de 2 minutos)
python benchmark.py             # completo (cerca de 10 minutos)
python plot_benchmark.py        # gráficos, tabela LaTeX e resumo
```

Opções do `benchmark.py`: `--seconds`, `--repeats`, `--warmup`, `--min-frames`, `--levels`, `--lights`, `--size` e `--out`.

Para resultados confiáveis: deixe o computador na tomada e no modo de melhor desempenho, feche outros programas, mantenha a janela visível e não mexa no computador durante a medição.

Arquivos gerados em `resultados/`:

- `benchmark.csv` — um resultado por linha
- `info.txt` — GPU, OpenGL, CPU, sistema e versões das bibliotecas
- `cena_com_sombras.png` e `cena_sem_sombras.png` — capturas da cena de teste
- `fig1_fps_x_luzes.png`, `fig2_quadro_x_triangulos.png`, `fig3_custo_por_etapa.png`, `fig4_sobrecusto_sombras.png` — gráficos
- `tabela_fps.tex` e `resumo.txt` — tabela pronta para LaTeX e resumo dos ajustes

## Como o algoritmo funciona (resumo)

1. **Ambiente:** desenha a cena só com luz ambiente e grava a profundidade.
2. **Para cada luz:**
   1. zera o stencil;
   2. desenha o volume de sombra sem cor nem profundidade: faces de trás que falham no teste de profundidade incrementam o contador e faces da frente que falham decrementam (z-fail);
   3. redesenha a cena com a luz, somando por *blending* aditivo, só onde o stencil vale zero.

A projeção usa plano distante no infinito. Os volumes são fechados por tampas formadas pelos triângulos de costas para a luz.

## Resultados principais

Medidos em uma GPU integrada Intel Iris Xe (detalhes no artigo):

- O tempo de quadro cresce de forma aproximadamente **linear com o número de luzes**, com inclinação proporcional ao tamanho da malha.
- O sobrecusto das sombras em relação à iluminação sem sombras vai de cerca de **1,5×** (até 960 triângulos) a **19,8×** (61.440 triângulos).
- Com o critério de 60 FPS, o método atende cenas de até 3.840 triângulos com 8 luzes e de 15.360 triângulos com até 2 luzes.
- Nesta implementação o gargalo está na **CPU** (87% a 93% do quadro na maior cena), principalmente na extrusão do volume em Python/NumPy. O passe de stencil na GPU custou menos de 9 ms mesmo com 8 luzes.

## Limitações

- Testes em um único computador, com GPU integrada.
- Malhas estáticas, fechadas e luzes pontuais.
- Algumas configurações com sombras tiveram variação alta entre repetições (detalhes no artigo).
- As cenas Prédios, Bonecos e Cidade são demonstrações qualitativas e não foram medidas.
- Não foram aplicadas otimizações como recorte de pares luz/objeto, teste de escopo por luz (*scissor*) ou z-pass quando a câmera está fora do volume.

## Referências
C. Everitt and M. J. Kilgard, "Practical and robust stenciled shadow volumes for hardware-accelerated rendering," NVIDIA Corporation, 2002.
F. C. Crow, "Shadow algorithms for computer graphics," ACM SIGGRAPH Computer Graphics, vol. 11, no. 2, pp. 242–248, 1977.
T. Heidmann, "Real shadows, real time," Iris Universe, no. 18, pp. 28–31, Silicon Graphics Inc., 1991.
1. C. Everitt and M. J. Kilgard, "Practical and robust stenciled shadow volumes for hardware-accelerated rendering," NVIDIA Corporation, 2002.
2. F. C. Crow, "Shadow algorithms for computer graphics," *ACM SIGGRAPH Computer Graphics*, vol. 11, no. 2, pp. 242–248, 1977.
3. T. Heidmann, "Real shadows, real time," *Iris Universe*, no. 18, pp. 28–31, Silicon Graphics Inc., 1991.
