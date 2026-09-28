import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st


# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="Física Visual: Energia e Dinâmica",
    page_icon="⚡",
    layout="wide",
)


# ============================================
# CSS PERSONALIZADO
# ============================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1a1a2e;
            text-align: center;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #555;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .concept-card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1.2rem;
            border-left: 4px solid;
            margin-bottom: 1rem;
        }

        .param-box {
            background: #fff;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cores padrão das energias
COR_EC = "#3498db"    # Cinética — azul
COR_EPG = "#9b59b6"   # Potencial gravitacional — roxo
COR_EPE = "#2ecc71"   # Potencial elástica — verde
COR_EM = "#34495e"    # Mecânica — cinza escuro

CORES_ENERGIA = {
    "Ec": COR_EC,
    "Epg": COR_EPG,
    "Epe": COR_EPE,
    "Em": COR_EM,
}


# ============================================
# CONFIGURAÇÃO GLOBAL DA ANIMAÇÃO
# ============================================
# Este valor é a duração de CADA QUADRO da animação.
# Ele NÃO é o passo do integrador físico: as distâncias
# entre as amostras já vêm da solução física do movimento.
VELOCIDADES_MS = [5, 10, 20, 30, 50, 80, 120, 160]

if "velocidade_ms" not in st.session_state:
    st.session_state.velocidade_ms = 30

if "gravidade_global" not in st.session_state:
    st.session_state.gravidade_global = 9.81

if "reproducao_continua" not in st.session_state:
    st.session_state.reproducao_continua = False


# ============================================
# FUNÇÕES AUXILIARES DE DESENHO
# ============================================
def criar_mola(x0, x1, y0, n_voltas=12, largura=0.35):
    """
    Desenha a mola entre x0 e x1, que devem ser as
    extremidades REAIS da mola.
    """
    if x0 >= x1:
        return [x0, x1], [y0, y0]

    x_vals = np.linspace(x0, x1, n_voltas * 2 + 1)
    y_vals = np.zeros_like(x_vals)

    for i in range(len(x_vals)):
        if i == 0 or i == len(x_vals) - 1:
            y_vals[i] = y0
        elif i % 2 == 0:
            y_vals[i] = y0 + largura
        else:
            y_vals[i] = y0 - largura

    return x_vals, y_vals


def criar_bloco(x_centro, y_base, largura=0.8, altura=0.8):
    """
    Bloco alinhado aos eixos, com a base sobre a pista.
    x_centro e y_base são as coordenadas físicas.
    """
    hx = largura / 2

    x = [
        x_centro - hx,
        x_centro + hx,
        x_centro + hx,
        x_centro - hx,
        x_centro - hx,
    ]

    y = [
        y_base,
        y_base,
        y_base + altura,
        y_base + altura,
        y_base,
    ]

    return x, y


def _poligono_orientado(p, u_hat, n_hat, largura, altura):
    """
    Retângulo centrado em p, com:
        u_hat -> direção de avanço (comprimento = largura)
        n_hat -> direção perpendicular (comprimento = altura)
    """
    p = np.asarray(p, dtype=float)
    u_hat = np.asarray(u_hat, dtype=float)
    n_hat = np.asarray(n_hat, dtype=float)

    a = 0.5 * largura
    b = 0.5 * altura

    cantos = [
        p - a * u_hat - b * n_hat,
        p + a * u_hat - b * n_hat,
        p + a * u_hat + b * n_hat,
        p - a * u_hat + b * n_hat,
        p - a * u_hat - b * n_hat,
    ]

    return np.array([c[0] for c in cantos]), np.array([c[1] for c in cantos])


def mostrar_controles_velocidade(prefixo):
    """
    Controles de velocidade compartilhados pelas quatro abas.
    """
    col_b1, col_b2 = st.columns(2)

    if col_b1.button("⏩ Mais Rápido", key=f"fast_{prefixo}"):
        indice = VELOCIDADES_MS.index(st.session_state.velocidade_ms)
        st.session_state.velocidade_ms = VELOCIDADES_MS[max(0, indice - 1)]

    if col_b2.button("⏪ Mais Lento", key=f"slow_{prefixo}"):
        indice = VELOCIDADES_MS.index(st.session_state.velocidade_ms)
        st.session_state.velocidade_ms = VELOCIDADES_MS[
            min(len(VELOCIDADES_MS) - 1, indice + 1)
        ]

    st.markdown(
        f"<b>Velocidade atual:</b> {st.session_state.velocidade_ms} ms/quadro",
        unsafe_allow_html=True,
    )


# ============================================
# MOTOR DE MOVIMENTO CONSERVATIVO (1 grau de liberdade)
# ============================================
def _construir_grade(segmentos, extras=(), pontos=6000):
    limites = np.asarray(segmentos, dtype=float)

    if limites.ndim != 2 or limites.shape[1] != 2:
        raise ValueError("segmentos deve ser uma sequência de pares (a, b).")

    partes = [
        np.linspace(float(a), float(b), pontos + 1)
        for a, b in limites
    ]

    if len(extras) > 0:
        partes.append(np.asarray(extras, dtype=float).ravel())

    grade = np.unique(np.concatenate(partes))

    return (
        grade,
        float(limites[:, 0].min()),
        float(limites[:, 1].max()),
    )


def tabela_tempo(
    potencial,
    segmentos,
    s_inicial,
    massa,
    energia_total,
    v_inicial=0.0,
    extras=(),
    pontos=6000,
):
    """
    Monta a tabela (posição, tempo, velocidade) de um sistema
    conservativo unidimensional com potencial U(x).

        Ec = Em - U
        v  = ±√[2(Em - U)/m]
        dt = 2·dx / (v_i + v_{i+1})

    A posição inicial pode ser:
      * um ponto de retorno (v_inicial = 0), inclusive nas
        bordas do domínio; ou
      * um ponto com velocidade inicial não nula, desde que
        o sentido seja compatível com o movimento.

    A região acessível (U ≤ Em) é determinada automaticamente,
    e o ponto de virada do primeiro trecho é obtido por
    interpolação onde U(x) = Em.
    """
    if massa <= 0:
        raise ValueError("A massa deve ser positiva.")

    if energia_total <= 0:
        raise ValueError("A energia mecânica total deve ser positiva.")

    grade, lo, hi = _construir_grade(segmentos, extras, pontos)

    tolerancia = 1e-9 * max(1.0, abs(energia_total))

    if s_inicial < lo - tolerancia or s_inicial > hi + tolerancia:
        raise ValueError(
            f"A posição inicial ({s_inicial:.4f}) está fora do domínio "
            f"[{lo:.4f}, {hi:.4f}]."
        )

    s_inicial = float(np.clip(s_inicial, lo, hi))

    potencial_grade = np.asarray(potencial(grade), dtype=float)

    if not np.all(np.isfinite(potencial_grade)):
        raise ValueError("O potencial não está definido em todo o domínio.")

    acessivel = potencial_grade <= energia_total + tolerancia

    i0 = int(np.argmin(np.abs(grade - s_inicial)))

    if not acessivel[i0]:
        raise ValueError(
            "O estado inicial possui energia acima da energia mecânica "
            "informada: o movimento é impossível."
        )

    # Limite acessível à esquerda
    j = i0
    while j > 0 and acessivel[j - 1]:
        j -= 1

    if j == 0:
        a_acessivel = float(grade[0])
    else:
        a_acessivel = float(
            grade[j]
            + (grade[j - 1] - grade[j])
            * (energia_total - potencial_grade[j])
            / (potencial_grade[j - 1] - potencial_grade[j])
        )

    # Limite acessível à direita
    k = i0
    while k < len(grade) - 1 and acessivel[k + 1]:
        k += 1

    if k == len(grade) - 1:
        b_acessivel = float(grade[-1])
    else:
        b_acessivel = float(
            grade[k]
            + (grade[k + 1] - grade[k])
            * (energia_total - potencial_grade[k])
            / (potencial_grade[k + 1] - potencial_grade[k])
        )

    # O corpo segue até o outro ponto de retorno do trecho
    if abs(s_inicial - a_acessivel) >= abs(b_acessivel - s_inicial):
        alvo = a_acessivel
        sentido = -1.0
    else:
        alvo = b_acessivel
        sentido = +1.0

    if abs(alvo - s_inicial) < 1e-12:
        raise ValueError(
            "Movimento degenerado: a posição inicial é o único ponto "
            "acessível com a energia informada."
        )

    if v_inicial != 0.0:
        if math.copysign(1.0, v_inicial) != math.copysign(1.0, sentido):
            raise ValueError(
                "O sentido da velocidade inicial é incompatível com "
                "o primeiro trecho do movimento."
            )

    caminho = np.linspace(s_inicial, alvo, pontos + 1)

    potencial_caminho = np.minimum(
        np.asarray(potencial(caminho), dtype=float),
        energia_total,
    )

    velocidades = np.sqrt(
        np.maximum(0.0, 2.0 * (energia_total - potencial_caminho) / massa)
    )

    dx = np.diff(caminho)

    velocidade_media = (velocidades[:-1] + velocidades[1:]) / 2.0
    velocidade_media = np.maximum(velocidade_media, 1e-12)

    dt = 2.0 * np.abs(dx) / velocidade_media

    t_ida = np.concatenate([[0.0], np.cumsum(dt)])

    posicoes = np.concatenate([caminho, caminho[-2::-1]])
    tempos = np.concatenate([t_ida, 2.0 * t_ida[-1] - t_ida[-2::-1]])
    vel = np.concatenate([
        sentido * velocidades,
        -sentido * velocidades[-2::-1],
    ])

    if not np.all(np.diff(tempos) > 0):
        raise ValueError("Falha ao montar a tabela de tempo.")

    return posicoes, tempos, vel


def simular_unidimensional(
    potencial,
    segmentos,
    s_inicial,
    massa,
    energia_total,
    v_inicial=0.0,
    extras=(),
    fps=50,
    minimo_amostras=150,
    pontos=6000,
):
    """
    Devolve (t, s, v) com instantes uniformemente espaçados,
    obtained da solução física.
    """
    tabela = tabela_tempo(
        potencial,
        segmentos,
        s_inicial,
        massa,
        energia_total,
        v_inicial=v_inicial,
        extras=extras,
        pontos=pontos,
    )

    _, tempos_tab, _ = tabela

    duracao = float(tempos_tab[-1])

    n = max(minimo_amostras, int(math.ceil(duracao * fps)), 2)

    tempos = np.linspace(0.0, duracao, n)

    posicoes = np.interp(tempos, tempos_tab, tabela[0])
    velocidades = np.interp(tempos, tempos_tab, tabela[2])

    return tempos, posicoes, velocidades, duracao


# ============================================
# ENERGIA
# ============================================
def medir_energia(massa, energia_total, posicoes, componentes):
    """
    componentes: dicionário {"Epg": fn, "Epe": fn, ...}

    A energia cinética vem da conservação:
        Ec = Em - (Epg + Epe)
    """
    potencial_total = np.zeros_like(np.asarray(posicoes, dtype=float))

    valores = {}

    for rotulo, funcao in componentes.items():
        valores[rotulo] = np.asarray(funcao(posicoes), dtype=float)
        potencial_total = potencial_total + valores[rotulo]

    valores["Ec"] = energia_total - potencial_total

    return valores


def energia_a_partir_da_velocidade(massa, gravidade, alturas, vx, vy):
    """
    Energia total em sistemas com velocidade vetorial conhecida.
    """
    alturas = np.asarray(alturas, dtype=float)

    velocidade_ao_quadrado = np.asarray(vx, dtype=float) ** 2 + np.asarray(
        vy, dtype=float
    ) ** 2

    return {
        "Ec": 0.5 * massa * velocidade_ao_quadrado,
        "Epg": massa * gravidade * alturas,
        "Epe": np.zeros_like(alturas),
    }


def criar_barras_energia(energias, rotulos):
    alturas = [float(energias[nome]) for nome in rotulos]
    cores = [CORES_ENERGIA[nome] for nome in rotulos]
    textos = [f"{valor:.2f} J" for valor in alturas]

    return go.Bar(
        x=rotulos,
        y=alturas,
        marker_color=cores,
        text=textos,
        textposition="auto",
        hovertemplate="%{x}: %{y:.4f} J<extra></extra>",
    )


def criar_frames(energias, tempos, rotulos, indice_traces, repeticoes):
    """
    Cada quadro recebe as energias calculadas para aquele
    instante físico, e não para um ângulo artificial.
    """
    total = {
        "Ec": 0.0,
        "Epg": 0.0,
        "Epe": 0.0,
        "Em": 0.0,
    }

    for nome in total:
        valores = np.asarray(energias[nome], dtype=float)
        acumulado = valores.copy()

        for outro in total:
            if outro != nome and outro in energias:
                acumulado = acumulado + np.asarray(
                    energias[outro], dtype=float
                )

        total[nome] = valores + acumulado - valores  # placeholder

    # Em = soma de todos os componentes presentes
    Em = np.zeros_like(np.asarray(energias["Ec"], dtype=float))

    for nome in ("Ec", "Epg", "Epe"):
        if nome in energias:
            Em = Em + np.asarray(energias[nome], dtype=float)

    total["Em"] = Em

    frames = []

    duracao_periodo = float(tempos[-1] - tempos[0])

    for ciclo in range(repeticoes):
        deslocamento = ciclo * duracao_periodo

        for i in range(len(tempos)):
            alturas = [float(total[nome][i]) for nome in rotulos]
            textos = [f"{valor:.2f} J" for valor in alturas]

            frames.append(
                go.Frame(
                    data=[
                        go.Bar(
                            y=alturas,
                            text=textos,
                            hovertemplate="%{y:.4f} J<extra></extra>",
                        )
                    ],
                    traces=indice_traces,
                    name=f"t = {tempos[i] + deslocamento:.4f} s",
                )
            )

    return frames


def configurar_animacao(figura, tempos, frames, duracao_ms, passo_txt):
    """
    Play, Pause, Reiniciar e navegação temporal.
    """
    tempos = np.asarray(tempos, dtype=float)

    # Quadros extras parado no estado inicial, para suavizar
    # a transição entre repetições.
    parada = 12
    frames = list(frames) + [frames[0].copy()] * parada
    tempos = np.concatenate([tempos, np.full(parada, tempos[0])])

    figura.frames = frames

    opcoes_play = {
        "frame": {"duration": int(duracao_ms), "redraw": True},
        "fromcurrent": True,
        "transition": {"duration": 0, "easing": "linear"},
        "mode": "immediate",
    }

    botoes = [
        {
            "label": "▶ Play",
            "method": "animate",
            "args": [None, opcoes_play],
        },
        {
            "label": "❚❚ Pause",
            "method": "animate",
            "args": [
                [None],
                {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
        },
        {
            "label": "↺ Reiniciar",
            "method": "animate",
            "args": [
                [0],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
        },
    ]

    numero_passos = min(120, max(2, len(tempos) - 1))
    indices = np.unique(
        np.linspace(0, len(tempos) - 1, numero_passos).astype(int)
    )

    passos = [
        {
            "label": f"{tempos[i]:.3f} s",
            "method": "animate",
            "args": [
                [int(i)],
                {
                    "frame": {
                        "duration": int(duracao_ms),
                        "redraw": True,
                    },
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
        }
        for i in indices
    ]

    figura.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=58, b=10),
        height=430,
        uirevision="simulacao-conservativa",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.15,
                "buttons": botoes,
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.10,
                "y": 1.135,
                "len": 0.88,
                "pad": {"t": 38, "b": 10},
                "currentvalue": {
                    "prefix": "Tempo: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "steps": passos,
            }
        ],
    )


def publicar_animacao(
    figura,
    tempos,
    energias,
    rotulos,
    indice_traces,
    duracao_ms,
    passo_txt,
    passo_integrador=None,
):
    """
    Publica a animação e mostra o diagnóstico energético.
    """
    repeticoes = 3 if st.session_state.reproducao_continua else 1

    frames = criar_frames(
        energias,
        tempos,
        rotulos,
        indice_traces,
        repeticoes,
    )

    dt_fisico = float(tempos[1] - tempos[0])
    tempo_real = dt_fisico / (duracao_ms / 1000.0)

    configurar_animacao(
        figura,
        tempos,
        frames,
        duracao_ms,
        passo_txt,
    )

    st.plotly_chart(
        figura,
        use_container_width=True,
        config={"displayModeBar": False, "scrollZoom": False},
    )

    componentes = ["Ec", "Epg", "Epe"]
    componentes = [c for c in componentes if c in energias]

    Em = np.zeros_like(np.asarray(energias["Ec"], dtype=float))
    for nome in componentes:
        Em = Em + np.asarray(energias[nome], dtype=float)

    erro = np.max(np.abs(Em - Em[0])) / max(abs(Em[0]), 1e-12)

    info = (
        f"Δt físico entre quadros: <b>{dt_fisico:.6f} s</b> | "
        f"velocidade de reprodução: <b>{tempo_real:.2f}×</b> o tempo real"
    )

    if passo_integrador is not None:
        info += f" | passo do integrador: <b>{passo_integrador:.8f} s</b>"

    st.markdown(
        f"""
        <div class="concept-card"
             style="border-left-color: #7f8c8d; padding: 0.8rem;">
            {info}<br>
            Energia mecânica inicial: <b>{Em[0]:.4f} J</b><br>
            Erro relativo máximo de conservação: <b>{erro:.3e}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return dt_fisico


# ============================================
# ABA 1 — RAMPA EM U
# ============================================
def gerar_figura_rampa_u(massa, altura_max, duracao_ms, gravidade=9.81):
    x_max = 5.0
    coeficiente = altura_max / (x_max**2)

    def altura(x):
        x = np.asarray(x, dtype=float)
        return coeficiente * x**2

    energia_inicial = massa * gravidade * altura_max

    tempos, posicoes, velocidades, periodo = simular_unidimensional(
        lambda x: massa * gravidade * altura(x),
        segmentos=[(-x_max, 0.0), (0.0, x_max)],
        s_inicial=x_max,          # ponto de retorno, na borda do domínio
        massa=massa,
        energia_total=energia_inicial,
    )

    energias = medir_energia(
        massa,
        energia_inicial,
        posicoes,
        {"Epg": lambda x: massa * gravidade * altura(x)},
    )

    rotulos = ["Ec", "Epg", "Em"]

    figura = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.08,
    )

    x_pista = np.linspace(-x_max, x_max, 100)

    figura.add_trace(
        go.Scatter(
            x=x_pista,
            y=altura(x_pista),
            mode="lines",
            line=dict(color="#7f8c8d", width=4),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    figura.add_trace(
        go.Scatter(
            x=[x_max],
            y=[altura_max + 0.3],
            mode="markers",
            marker=dict(
                color="#e74c3c",
                size=22,
                line=dict(color="#c0392b", width=2),
            ),
            hovertemplate="Esfera",
        ),
        row=1,
        col=1,
    )

    indice_barras = len(figura.data)

    figura.add_trace(
        criar_barras_energia(
            {nome: energias[nome][:1] for nome in ("Ec", "Epg")},
            rotulos[:2],
        ),
        row=1,
        col=2,
    )

    figura.update_xaxes(
        range=[-6.5, 6.5], showgrid=False, zeroline=False,
        visible=False, row=1, col=1,
    )
    figura.update_yaxes(
        range=[-1, altura_max + 2.5], showgrid=False, zeroline=False,
        visible=False, row=1, col=1,
    )

    Em_max = float(energias["Ec"].max() + energias["Epg"].max())
    figura.update_yaxes(
        range=[0, max(10.0, Em_max * 1.25)],
        title="Energia (Joules)",
        row=1,
        col=2,
    )

    publicar_animacao(
        figura,
        tempos,
        energias,
        rotulos,
        [indice_barras],
        duracao_ms,
        "Δt = dx/v",
    )

    st.caption(
        f"""
        Período do vaivém: <b>{periodo:.4f} s</b>.
        Ponto mais baixo do fundo: <b>x = 0</b>, onde
        <b>v_max = {velocidades[np.argmax(np.abs(velocidades))]:.3f} m/s</b>.
        """
    )


# ============================================
# ABA 2 — SISTEMA MASSA-MOLA
# ============================================
def gerar_figura_massa_mola(massa, k_mola, amplitude, duracao_ms):
    """
    A amplitude é a extensão máxima da mola em relação à
    posição de equilíbrio. A extremidade esquerda da mola
    permanece FIXA.
    """
    x_ancora = -amplitude
    x_inicial = x_ancora + amplitude

    largura_bloco = 0.8
    x_face_direita = x_inicial + largura_bloco / 2.0
    x_parede = x_face_direita + 0.7

    def extensao(x):
        return np.asarray(x, dtype=float) - x_ancora

    energia_inicial = 0.5 * k_mola * amplitude**2

    tempos, posicoes, velocidades, periodo = simular_unidimensional(
        lambda x: 0.5 * k_mola * extensao(x) ** 2,
        segmentos=[(x_ancora - amplitude - 0.6, x_ancora + amplitude + 0.6)],
        s_inicial=x_inicial,
        massa=massa,
        energia_total=energia_inicial,
    )

    energias = medir_energia(
        massa,
        energia_inicial,
        posicoes,
        {"Epe": lambda x: 0.5 * k_mola * extensao(x) ** 2},
    )

    rotulos = ["Ec", "Epe", "Em"]

    molas = [
        criar_mola(x_ancora, x - largura_bloco / 2.0, 0.4, n_voltas=11)
        for x in posicoes
    ]

    figura = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.08,
    )

    figura.add_trace(
        go.Scatter(
            x=[
                x_parede - 1.0,
                x_face_direita + 1.0,
                x_face_direita + 1.0,
                x_parede - 1.0,
                x_parede - 1.0,
            ],
            y=[-0.8, -0.8, 0.0, 0.0, -0.8],
            fill="toself",
            fillcolor="#bdc3c7",
            line=dict(width=0),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    figura.add_trace(
        go.Scatter(
            x=[x_parede, x_parede],
            y=[0, 2.2],
            mode="lines",
            line=dict(color="#95a5a6", width=6),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    figura.add_trace(
        go.Scatter(
            x=molas[0][0],
            y=molas[0][1],
            mode="lines",
            line=dict(color="#7f8c8d", width=3),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    indice_bloco = len(figura.data)

    bx, by = criar_bloco(x_inicial, 0.0, largura_bloco, 0.8)

    figura.add_trace(
        go.Scatter(
            x=bx,
            y=by,
            fill="toself",
            fillcolor="#3498db",
            line=dict(color="#2980b9", width=2),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    indice_barras = len(figura.data)

    figura.add_trace(
        criar_barras_energia(
            {nome: energias[nome][:1] for nome in ("Ec", "Epe")},
            rotulos[:2],
        ),
        row=1,
        col=2,
    )

    # Quadros da mola e do bloco
    frames_mola = []
    frames_bloco = []

    for (xm, ym), x in zip(molas, posicoes):
        bxf, byf = criar_bloco(x, 0.0, largura_bloco, 0.8)

        frames_mola.append(
            go.Frame(
                data=[go.Scatter(x=xm, y=ym)],
                traces=[2],
                name="mola",
            )
        )

        frames_bloco.append(
            go.Frame(
                data=[go.Scatter(x=bxf, y=byf)],
                traces=[indice_bloco],
                name="bloco",
            )
        )

    figura.update_xaxes(
        range=[x_parede - 0.5, x_face_direita + 1.6],
        showgrid=False, zeroline=False, visible=False, row=1, col=1,
    )
    figura.update_yaxes(
        range=[-1.2, 2.8], showgrid=False, zeroline=False,
        visible=False, row=1, col=1,
    )

    Em_max = float(energias["Ec"].max() + energias["Epe"].max())
    figura.update_yaxes(
        range=[0, max(10.0, Em_max * 1.25)],
        title="Energia (Joules)", row=1, col=2,
    )

    dt = publicar_animacao(
        figura,
        tempos,
        energias,
        rotulos,
        [indice_barras],
        duracao_ms,
        "Δt = dx/v",
    )

    # Injeta os quadros do movimento nas barras de energia
    # (publicar_animacao já criou figura.frames: acrescentamos os
    #  dados que faltam, referenciando as mesmas posições).
    for i, frame in enumerate(frames_mola):
        figura.frames[i].data = list(frame.data) + list(figura.frames[i].data)
        figura.frames[i].traces = [2, indice_bloco] + list(figura.frames[i].traces)

    st.caption(
        f"""
        Posição de equilíbrio da mola: <b>x = {x_ancora:.2f} m</b>.  
        Frequência angular: <b>ω = √(k/m) = {math.sqrt(k_mola / massa):.4f} rad/s</b>.  
        Período teórico: <b>T = 2π√(m/k) = {2 * math.pi * math.sqrt(massa / k_mola):.4f} s</b>  
        (o período medido na animação foi de <b>{periodo:.4f} s</b>).
        """
    )


# ============================================
# ABA 3 — RAMPA + MOLA
# ============================================
def gerar_figura_rampa_mola_ref(
    massa, altura_max, k_mola, duracao_ms, gravidade=9.81
):
    x_topo_rampa = -6.0
    x_base_rampa = -2.0
    x_mola_esq = 2.0          # extremidade FIXA da mola

    largura_bloco = 0.7
    metade = largura_bloco / 2.0

    x_contato = x_mola_esq - metade   # centro do bloco no 1º contato
    x_parede_folga = 0.6              # folga entre bloco e parede

    energia_inicial = massa * gravidade * altura_max
    compressao_maxima = math.sqrt(2.0 * energia_inicial / k_mola)

    x_parede = x_contato + metade + compressao_maxima + x_parede_folga
    x_virada = x_parede - metade - compressao_maxima
    x_fim_dominio = x_virada + 0.6

    def altura(x):
        x = np.asarray(x, dtype=float)
        return np.where(
            x < x_base_rampa,
            altura_max
            * ((x - x_base_rampa) / (x_topo_rampa - x_base_rampa)) ** 2,
            0.0,
        )

    def compressao(x):
        return x_parede - (np.asarray(x, dtype=float) + metade)

    def potencial(x):
        x = np.asarray(x, dtype=float)
        comp = compressao(x)
        return massa * gravidade * altura(x) + np.where(
            x > x_contato, 0.5 * k_mola * comp**2, 0.0
        )

    tempos, posicoes, velocidades, periodo = simular_unidimensional(
        potencial,
        segmentos=[
            (x_topo_rampa, x_base_rampa),
            (x_base_rampa, x_contato),
            (x_contato, x_fim_dominio),
        ],
        s_inicial=x_topo_rampa,
        massa=massa,
        energia_total=energia_inicial,
        extras=[x_base_rampa, x_contato],
    )

    energias = medir_energia(
        massa,
        energia_inicial,
        posicoes,
        {
            "Epg": lambda x: massa * gravidade * altura(x),
            "Epe": lambda x: np.where(
                x > x_contato, 0.5 * k_mola * compressao(x) ** 2, 0.0
            ),
        },
    )

    rotulos = ["Ec", "Epg", "Epe", "Em"]

    figura = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.08,
    )

    xr = np.linspace(x_topo_rampa, x_base_rampa, 40)
    xp = np.linspace(x_base_rampa, x_parede, 40)

    figura.add_trace(
        go.Scatter(
            x=np.concatenate([xr, xp]),
            y=np.concatenate([altura(xr), np.zeros_like(xp)]),
            mode="lines",
            line=dict(color="#7f8c8d", width=4),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    figura.add_trace(
        go.Scatter(
            x=[x_parede, x_parede],
            y=[0, 1.8],
            mode="lines",
            line=dict(color="#95a5a6", width=6),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    molas = []
    for x in posicoes:
        if x + metade > x_mola_esq:
            molas.append(criar_mola(x_mola_esq, x + metade, 0.35, n_voltas=11))
        else:
            molas.append(([np.nan, np.nan], [0.35, 0.35]))

    figura.add_trace(
        go.Scatter(
            x=molas[0][0], y=molas[0][1],
            mode="lines",
            line=dict(color="#2ecc71", width=3),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    indice_bloco = len(figura.data)

    bx, by = criar_bloco(
        x_topo_rampa, altura_max, largura_bloco, largura_bloco
    )

    figura.add_trace(
        go.Scatter(
            x=bx, y=by,
            fill="toself",
            fillcolor="#e74c3c",
            line=dict(color="#c0392b", width=2),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    indice_barras = len(figura.data)

    figura.add_trace(
        criar_barras_energia(
            {nome: energias[nome][:1] for nome in ("Ec", "Epg", "Epe")},
            rotulos[:3],
        ),
        row=1,
        col=2,
    )

    figura.update_xaxes(
        range=[x_topo_rampa - 1, x_parede + 1],
        showgrid=False, zeroline=False, visible=False, row=1, col=1,
    )
    figura.update_yaxes(
        range=[-0.5, altura_max + 1.5],
        showgrid=False, zeroline=False, visible=False, row=1, col=1,
    )

    Em_max = float(energias["Ec"].max() + energias["Epg"].max()
                   + energias["Epe"].max())
    figura.update_yaxes(
        range=[0, max(10.0, Em_max * 1.25)],
        title="Energia (Joules)", row=1, col=2,
    )

    publicar_animacao(
        figura, tempos, energias, rotulos,
        [indice_barras], duracao_ms, "Δt = dx/v",
    )

    for i, x in enumerate(posicoes):
        xm, ym = molas[i]
        bxf, byf = criar_bloco(x, float(altura(x)), largura_bloco, largura_bloco)

        figura.frames[i].data = [
            go.Scatter(x=xm, y=ym),
            go.Scatter(x=bxf, y=byf),
        ] + list(figura.frames[i].data)

        figura.frames[i].traces = [2, indice_bloco] + list(
            figura.frames[i].traces
        )

    st.caption(
        f"""
        Compressão máxima teórica: <b>x_max = {compressao_maxima:.3f} m</b>
        (obtida em <b>x = {x_virada:.3f} m</b>).  
        A parede fica em <b>x = {x_parede:.3f} m</b>: na compressão máxima
        o bloco <b>para</b> e retorna, sem atravessá-la.  
        A extremidade esquerda da mola permanece fixa em
        <b>x = {x_mola_esq:.2f} m</b>.
        """
    )


# ============================================
# LOOPING — GEOMETRIA, CRITÉRIOS E VOO LIVRE
# ============================================
def resumo_looping(raio, velocidade_inicial, gravidade):
    energia_inicial = 0.5 * velocidade_inicial**2

    altura_maxima = energia_inicial / gravidade
    altura_topo = 2.0 * raio

    razao = velocidade_inicial**2 / (gravidade * raio)

    return {
        "energia_inicial": energia_inicial,
        "altura_maxima": altura_maxima,
        "altura_topo": altura_topo,
        "velocidade_topo_minima": math.sqrt(gravidade * raio),
        "velocidade_livre_minima": math.sqrt(5.0 * gravidade * raio),
        "pode_chegar_ao_topo": altura_maxima >= altura_topo,
        "contato_continuo": razao >= 5.0,
        "razao": razao,
    }


def geometria_loop(raio, x_inicio):
    return {
        "x_inicio": x_inicio,
        "L_reta": raio - x_inicio,
        "L_loop": 2.0 * math.pi * raio,
    }


def ponto_no_caminho(s, raio, x_inicio):
    """
    Converte a coordenada de caminho s em posição (x, y) e
    ângulo do loop (NaN fora do loop).

    s = 0            → extremo esquerdo da reta
    s = L_reta       → base do loop
    s = L_total      → base do loop,Após uma volta completa
    s = 2·L_total    → volta ao extremo esquerdo, sentido inverso
    """
    L_reta = raio - x_inicio
    L_loop = 2.0 * math.pi * raio
    L_total = L_reta + L_loop

    s = np.asarray(s, dtype=float)

    x = np.full(s.shape, np.nan)
    y = np.full(s.shape, np.nan)
    theta = np.full(s.shape, np.nan)

    m1 = (s >= 0.0) & (s <= L_reta)
    x[m1] = x_inicio + s[m1]
    y[m1] = 0.0

    m2 = (s > L_reta) & (s <= L_total)
    th = (s[m2] - L_reta) / raio
    x[m2] = raio + raio * np.sin(th)
    y[m2] = raio * (1.0 - np.cos(th))
    theta[m2] = th

    m3 = (s > L_total) & (s <= L_total + L_loop)
    th = 2.0 * math.pi - (s[m3] - L_total) / raio
    x[m3] = raio + raio * np.sin(th)
    y[m3] = raio * (1.0 - np.cos(th))
    theta[m3] = th

    m4 = (s > L_total + L_loop) & (s <= 2.0 * L_total)
    x[m4] = raio - (s[m4] - (L_total + L_loop))
    y[m4] = 0.0

    return x, y, theta, L_reta, L_loop, L_total


def altura_do_caminho(s, raio, x_inicio):
    return ponto_no_caminho(s, raio, x_inicio)[1]


def descolamento(raio, velocidade_inicial, gravidade):
    """
    Ponto em que a reação normal se anula:

        N/m = v²/R + g·cos θ = 0   →   cos θ_d = (2gR − v₀²)/(3gR)

    Só existe (no trecho ascendente) quando 4gR ≤ v₀² < 5gR.
    """
    razao = velocidade_inicial**2 / (gravidade * raio)

    if razao < 4.0 or razao >= 5.0:
        return None

    cosseno = (2.0 - razao) / 3.0
    theta_d = math.acos(max(-1.0, min(1.0, cosseno)))
    velocidade_d = math.sqrt(max(0.0, -gravidade * raio * math.cos(theta_d)))

    return theta_d, velocidade_d


def rk4_balistico(p, v, gravidade, dt):
    g = np.array([0.0, -gravidade], dtype=float)

    def acc(q, w):
        return g

    k1_p, k1_v = v, acc(p, v)
    k2_p, k2_v = acc(p + 0.5 * dt * k1_p, v + 0.5 * dt * k1_v), None
    k2_p = v + 0.5 * dt * g
    k3_p = v + 0.5 * dt * g
    k4_p = v + dt * g

    p_novo = p + dt / 6.0 * (v + 2 * k2_p + 2 * k3_p + k4_p)
    v_novo = v + dt * g

    return p_novo, v_novo


def distancia_ao_trilho(p, raio, x_inicio):
    """
    Distância do ponto à união do segmento de reta e do loop.
    """
    px, py = float(p[0]), float(p[1])

    # distância ao segmento de reta y = 0, de x_inicio até raio
    if x_inicio <= px <= raio:
        d_reta = abs(py)
    else:
        d_reta = min(
            math.hypot(px - x_inicio, py),
            math.hypot(px - raio, py),
        )

    centro = np.array([raio, raio], dtype=float)
    d_loop = abs(float(np.linalg.norm(p - centro)) - raio)

    return min(d_reta, d_loop)


def integracao_livre(raio, velocidade_inicial, gravidade, x_inicio):
    """
    Simulação 2D: o corpo segue o trilho até o descolamento
    e depois descreve uma parábola (RK4) até tocar o nível
    do trilho ou a própria pista.
    """
    theta_d, velocidade_d = descolamento(
        raio, velocidade_inicial, gravidade
    )

    L_reta, L_loop, L_total = (
        geometria_loop(raio, x_inicio)["L_reta"],
        geometria_loop(raio, x_inicio)["L_loop"],
        geometria_loop(raio, x_inicio)["L_reta"]
        + geometria_loop(raio, x_inicio)["L_loop"],
    )

    s_d = L_reta + raio * theta_d

    energia_inicial = 0.5 * velocidade_inicial**2

    # Trecho 1: preso ao trilho
    tempos_1, s_1, v_1, duracao_total = simular_unidimensional(
        lambda s: gravidade * altura_do_caminho(s, raio, x_inicio),
        segmentos=[(0.0, L_reta), (L_reta, L_total), (L_total, 2 * L_total)],
        s_inicial=0.0,
        massa=1.0,
        energia_total=energia_inicial,
        v_inicial=velocidade_inicial,
        extras=[L_reta, L_total, L_total + L_loop],
    )

    _, tempos_tab, _ = tabela_tempo(
        lambda s: gravidade * altura_do_caminho(s, raio, x_inicio),
        segmentos=[(0.0, L_reta), (L_reta, L_total), (L_total, 2 * L_total)],
        s_inicial=0.0,
        massa=1.0,
        energia_total=energia_inicial,
        v_inicial=velocidade_inicial,
        extras=[L_reta, L_total, L_total + L_loop],
    )

    t_d = float(np.interp(s_d, tabela_tempo(
        lambda s: gravidade * altura_do_caminho(s, raio, x_inicio),
        segmentos=[(0.0, L_reta), (L_reta, L_total), (L_total, 2 * L_total)],
        s_inicial=0.0, massa=1.0, energia_total=energia_inicial,
        v_inicial=velocidade_inicial,
        extras=[L_reta, L_total, L_total + L_loop],
    )[0], tempos_tab))

    # Estado exato no descolamento
    p0 = np.array(
        [raio + raio * math.sin(theta_d), raio * (1 - math.cos(theta_d))],
        dtype=float,
    )

    t0 = np.array([math.cos(theta_d), math.sin(theta_d)], dtype=float)
    v0 = velocidade_d * t0

    # Trecho 2: voo balístico
    dt = 1.0 / 600.0
    p = p0.copy()
    v = v0.copy()

    t = 0.0
    registros_t = [0.0]
    registros_p = [p.copy()]
    registros_v = [v.copy()]

    altura_maxima_energia = energia_inicial / gravidade
    t_limite = 3.0 * math.sqrt(
        2.0 * max(altura_maxima_energia, 0.1) / gravidade
    )

    while t < t_limite:
        p, v = rk4_balistico(p, v, gravidade, dt)
        t += dt

        registros_t.append(t)
        registros_p.append(p.copy())
        registros_v.append(v.copy())

        tocou_nivel = p[1] <= 0.02
        perto_do_trilho = (
            t > 0.25 and distancia_ao_trilho(p, raio, x_inicio) < 0.08
        )

        if tocou_nivel or perto_do_trilho:
            break

    registros_t = np.asarray(registros_t)
    registros_p = np.asarray(registros_p)
    registros_v = np.asarray(registros_v)

    duracao_voo = float(registros_t[-1])

    # Amostragem uniforme da trajectory completa
    tempo_total = t_d + duracao_voo
    n = max(240, int(math.ceil(tempo_total * 50)))

    tempos = np.linspace(0.0, tempo_total, n)

    px = np.empty(n)
    py = np.empty(n)
    vx = np.empty(n)
    vy = np.empty(n)

    # Trecho preso ao trilho
    mask_1 = tempos <= t_d
    tempos_1u = tempos[mask_1]
    s_1u = np.interp(tempos_1u, tempos_tab, tabela_tempo(
        lambda s: gravidade * altura_do_caminho(s, raio, x_inicio),
        segmentos=[(0.0, L_reta), (L_reta, L_total), (L_total, 2 * L_total)],
        s_inicial=0.0, massa=1.0, energia_total=energia_inicial,
        v_inicial=velocidade_inicial,
        extras=[L_reta, L_total, L_total + L_loop],
    )[0])
    v_1u = np.interp(tempos_1u, tempos_tab, tabela_tempo(
        lambda s: gravidade * altura_do_caminho(s, raio, x_inicio),
        segmentos=[(0.0, L_reta), (L_reta, L_total), (L_total, 2 * L_total)],
        s_inicial=0.0, massa=1.0, energia_total=energia_inicial,
        v_inicial=velocidade_inicial,
        extras=[L_reta, L_total, L_total + L_loop],
    )[2])

    x1, y1, th1, _, _, _ = ponto_no_caminho(s_1u, raio, x_inicio)
    dir_t = np.where(v_1u >= 0, 1.0, -1.0)

    px[mask_1] = x1
    py[mask_1] = y1
    vx[mask_1] = dir_t * v_1u * np.cos(np.nan_to_num(th1, nan=0.0))
    vy[mask_1] = dir_t * v_1u * np.sin(np.nan_to_num(th1, nan=0.0))

    # Trecho balístico
    mask_2 = tempos > t_d
    if np.any(mask_2):
        tl = tempos[mask_2] - t_d
        idx = np.clip(
            np.searchsorted(registros_t, tl, side="right") - 1,
            0,
            len(registros_t) - 2,
        )
        frac = (tl - registros_t[idx]) / (
            registros_t[idx + 1] - registros_t[idx]
        )

        px[mask_2] = registros_p[idx, 0] + frac * (
            registros_p[idx + 1, 0] - registros_p[idx, 0]
        )
        py[mask_2] = registros_p[idx, 1] + frac * (
            registros_p[idx + 1, 1] - registros_p[idx, 1]
        )
        vx[mask_2] = registros_v[idx, 0] + frac * (
            registros_v[idx + 1, 0] - registros_v[idx, 0]
        )
        vy[mask_2] = registros_v[idx, 1] + frac * (
            registros_v[idx + 1, 1] - registros_v[idx, 1]
        )

    return tempos, px, py, vx, vy, dt, t_d, duracao_voo


# ============================================
# ABA 4 — BRINQUEDO LOOPING
# ============================================
def gerar_figura_looping_corrigido(
    massa, raio, velocidade_inicial, duracao_ms,
    gravidade=9.81, modo="trilho",
):
    resumo = resumo_looping(raio, velocidade_inicial, gravidade)

    x_inicio = -5.0
    geo = geometria_loop(raio, x_inicio)

    L_reta = geo["L_reta"]
    L_loop = geo["L_loop"]
    L_total = L_reta + L_loop

    energia_inicial = resumo["energia_inicial"]

    usar_livre = (
        modo == "livre"
        and 4.0 <= resumo["razao"] < 5.0
    )

    info_voo = ""

    if usar_livre:
        (
            tempos, px, py, vx, vy,
            passo_integrador, t_descolamento, duracao_voo,
        ) = integracao_livre(
            raio, velocidade_inicial, gravidade, x_inicio
        )

        theta_d = descolamento(raio, velocidade_inicial, gravidade)[0]
        x_d = raio + raio * math.sin(theta_d)

        info_voo = (
            f" Descolamento em θ = {math.degrees(theta_d):.1f}° "
            f"(x = {x_d:.2f} m, t = {t_descolamento:.3f} s); "
            f"voo balístico de {duracao_voo:.3f} s."
        )
    else:
        passo_integrador = None

        tempos, s, v_s, duracao = simular_unidimensional(
            lambda ss: gravidade * altura_do_caminho(ss, raio, x_inicio),
            segmentos=[
                (0.0, L_reta),
                (L_reta, L_total),
                (L_total, 2.0 * L_total),
            ],
            s_inicial=0.0,
            massa=1.0,
            energia_total=energia_inicial,
            v_inicial=velocidade_inicial,
            extras=[L_reta, L_total, L_total + L_loop],
        )

        px, py, th, _, _, _ = ponto_no_caminho(s, raio, x_inicio)

        sinal = np.where(v_s >= 0, 1.0, -1.0)
        th = np.nan_to_num(th, nan=0.0)

        vx = sinal * v_s * np.cos(th)
        vy = sinal * v_s * np.sin(th)

    energias = energia_a_partir_da_velocidade(massa, gravidade, py, vx, vy)

    rotulos = ["Ec", "Epg", "Em"]

    figura = make_subplots(
        rows=1, cols=2,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.08,
    )

    # Trilho
    x_linha = np.linspace(x_inicio, raio, 50)
    theta_loop = np.linspace(0.0, 2.0 * math.pi, 220)

    x_loop = raio + raio * np.sin(theta_loop)
    y_loop = raio * (1.0 - np.cos(theta_loop))

    figura.add_trace(
        go.Scatter(
            x=np.concatenate([x_linha, x_loop]),
            y=np.concatenate([np.zeros_like(x_linha), y_loop]),
            mode="lines",
            line=dict(color="#7f8c8d", width=4),
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    indice_carrinho = len(figura.data)

    poligonos = []
    for i in range(len(px)):
        # tangente e normal internas do trilho
        th_i = math.atan2(2.0 * raio - py[i], max(px[i] - raio, 1e-9)) * 0.0
        dentro = (
            (px[i] - raio) ** 2 + (py[i] - raio) ** 2 <= (raio + 0.05) ** 2
        )

        if dentro:
            dx_ = px[i] - raio
            dy_ = py[i] - raio
            norma = max(math.hypot(dx_, dy_), 1e-9)
            n_hat = np.array([-dx_ / norma, -dy_ / norma])  # inward
            u_hat = np.array([-n_hat[1], n_hat[0]])
        else:
            vx_i, vy_i = vx[i], vy[i]
            vel = max(math.hypot(vx_i, vy_i), 1e-9)
            u_hat = np.array([vx_i / vel, vy_i / vel])
            n_hat = np.array([-u_hat[1], u_hat[0]])

        poligonos.append(
            _poligono_orientado(
                np.array([px[i], py[i]]), u_hat, n_hat, 0.6, 0.6
            )
        )

    figura.add_trace(
        go.Scatter(
            x=poligonos[0][0],
            y=poligonos[0][1],
            fill="toself",
            fillcolor="#e74c3c",
            line=dict(color="#c0392b", width=2),
            hovertemplate=(
                "x = %{x:.3f} m<br>y = %{y:.3f} m<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    indice_barras = len(figura.data)

    figura.add_trace(
        criar_barras_energia(
            {nome: energias[nome][:1] for nome in ("Ec", "Epg")},
            rotulos[:2],
        ),
        row=1, col=2,
    )

    x_min = min(x_inicio, float(np.min(px)) - 1.0)
    x_max = max(3.0 * raio, float(np.max(px)) + 1.0)
    y_max = max(2.0 * raio + 2.0, float(np.max(py)) + 2.0)

    figura.update_xaxes(
        range=[x_min, x_max], showgrid=False, zeroline=False,
        visible=False, row=1, col=1,
    )
    figura.update_yaxes(
        range=[-1.0, y_max], showgrid=False, zeroline=False,
        visible=False, row=1, col=1,
    )
    figura.update_yaxes(
        range=[0, max(10.0, float((energias["Ec"] + energias["Epg"]).max()) * 1.25)],
        title="Energia (Joules)", row=1, col=2,
    )

    publicar_animacao(
        figura, tempos, energias, rotulos,
        [indice_barras], duracao_ms,
        "Δt = dx/v" if passo_integrador is None else "RK4 dt = 1/600 s",
        passo_integrador=passo_integrador,
    )

    for i in range(len(tempos)):
        figura.frames[i].data = [
            go.Scatter(x=poligonos[i][0], y=poligonos[i][1])
        ] + list(figura.frames[i].data)

        figura.frames[i].traces = [indice_carrinho] + list(
            figura.frames[i].traces
        )

    if usar_livre:
        st.caption(
            "🛫 " + info_voo.strip() +
            " A colisão com o trilho **não** é modelada, pois seria "
            "inelástica e violaria a conservação da energia."
        )
    elif modo == "livre":
        st.caption(
            "Sem perda de contato nesta configuração: a solução presa ao "
            "trilho já é fisicamente exata."
        )

    return resumo


# ============================================
# TÍTULO E NAVEGAÇÃO
# ============================================
st.markdown(
    '<div class="main-title">⚡ Sistemas Conservativos e Dinâmica</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Simulações físicas completas com equações,
        conservação de energia e controle temporal
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================
# CONFIGURAÇÕES GLOBAIS
# ============================================
with st.expander("⚙️ Configurações globais da simulação", expanded=False):
    st.slider(
        "Gravidade (m/s²)",
        min_value=1.0,
        max_value=15.0,
        value=9.81,
        step=0.01,
        key="gravidade_global",
    )

    st.checkbox(
        "Reprodução contínua (repetir o movimento 3×)",
        value=False,
        key="reproducao_continua",
        help=(
            "A repetição é apenas visual. O estado inicial e toda a "
            "trajetória são recalculados a cada mudança de parâmetro."
        ),
    )

    st.caption(
        """
        A velocidade em **ms/quadro** controla apenas a reprodução.
        O movimento é calculado a partir de
        **v = √[2(Em − U)/m]** e **dt = dx/v**, com o tempo físico
        reportado abaixo de cada animação.
        """
    )


tab1, tab2, tab3, tab4 = st.tabs([
    "1. Rampa em 'U' (Gravitacional)",
    "2. Sistema Massa-Mola (Elástica)",
    "3. Rampa Inclinada + Mola",
    "4. Brinquedo Looping",
])


# ============================================
# ABA 1
# ============================================
with tab1:
    st.markdown(
        """
        <div class="concept-card" style="border-left-color: #9b59b6;">
            <b>Princípio e Equações:</b> Esfera oscilando livremente em
            pista sem atrito. A velocidade é obtida de
            <b>v = √[2(Em − Epg)/m]</b> e o tempo de cada trecho de
            <b>dt = dx/v</b>. O objeto <b>pára</b> nas extremidades e
            <b>acelera</b> no fundo da pista.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"$$ E_m = E_c + E_{pg} = \frac{1}{2}mv^2 + mgh = \text{constante} $$"
    )

    col_c1, col_c2 = st.columns([1, 2.5])

    with col_c1:
        st.markdown("<div class='param-box'>", unsafe_allow_html=True)

        massa_u = st.slider(
            "Massa da esfera (kg)", 1.0, 10.0, 2.0, step=0.5, key="mu"
        )
        h_max_u = st.slider(
            "Altura inicial (m)", 2.0, 10.0, 5.0, step=0.5, key="hu"
        )

        mostrar_controles_velocidade("u")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_c2:
        gerar_figura_rampa_u(
            massa=massa_u,
            altura_max=h_max_u,
            duracao_ms=st.session_state.velocidade_ms,
            gravidade=st.session_state.gravidade_global,
        )


# ============================================
# ABA 2
# ============================================
with tab2:
    st.markdown(
        """
        <div class="concept-card" style="border-left-color: #2ecc71;">
            <b>Princípio e Equações:</b> Bloco preso a uma mola cuja
            extremidade esquerda permanece <b>fixa</b>. A mola apenas
            <b>comprime e relaxa</b>, e o período depende de √(m/k).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"$$ E_m = E_c + E_{pe} = \frac{1}{2}mv^2 + \frac{1}{2}kx^2 = \text{constante} $$"
    )

    col_m1, col_m2 = st.columns([1, 2.5])

    with col_m1:
        st.markdown("<div class='param-box'>", unsafe_allow_html=True)

        massa_m = st.slider(
            "Massa do bloco (kg)", 1.0, 10.0, 2.0, step=0.5, key="mm"
        )
        k_m = st.slider(
            "Constante elástica (N/m)", 10, 100, 50, step=10, key="km"
        )
        amp_m = st.slider(
            "Amplitude (m)", 1.0, 5.0, 3.0, step=0.5, key="ampm"
        )

        mostrar_controles_velocidade("m")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_m2:
        gerar_figura_massa_mola(
            massa=massa_m,
            k_mola=k_m,
            amplitude=amp_m,
            duracao_ms=st.session_state.velocidade_ms,
        )


# ============================================
# ABA 3
# ============================================
with tab3:
    st.markdown(
        """
        <div class="concept-card" style="border-left-color: #3498db;">
            <b>Princípio e Equações:</b> Bloco solto do alto da rampa.
            A energia potencial gravitacional vira energia cinética e,
            depois, energia elástica. Na compressão máxima o bloco
            <b>para instantaneamente</b> e retorna: ele não atravessa
            a parede.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"$$ mgh = \frac{1}{2}kx_{\max}^2 \;\implies\; x_{\max} = \sqrt{\frac{2mgh}{k}} $$"
    )

    col_r1, col_r2 = st.columns([1, 2.5])

    with col_r1:
        st.markdown("<div class='param-box'>", unsafe_allow_html=True)

        massa_rm = st.slider(
            "Massa do bloco (kg)", 1.0, 10.0, 2.0, step=0.5, key="m_rm"
        )
        h_rm = st.slider(
            "Altura inicial (h)", 1.0, 8.0, 4.0, step=0.5, key="h_rm"
        )
        k_rm = st.slider(
            "Constante da mola (k — N/m)", 20, 200, 100, step=10, key="k_rm"
        )

        mostrar_controles_velocidade("rm")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_r2:
        gerar_figura_rampa_mola_ref(
            massa=massa_rm,
            altura_max=h_rm,
            k_mola=k_rm,
            duracao_ms=st.session_state.velocidade_ms,
            gravidade=st.session_state.gravidade_global,
        )


# ============================================
# ABA 4
# ============================================
with tab4:
    st.markdown(
        """
        <div class="concept-card" style="border-left-color: #e74c3c;">
            <b>Princípio e Equações:</b> O carrinho parte da reta
