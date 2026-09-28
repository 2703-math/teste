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
# Este valor controla a duração entre imagens da animação.
# NÃO é o passo do integrador físico.
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
    Desenha uma mola entre x0 e x1.

    x0 e x1 devem ser as extremidades REAIS da mola.
    Se x0 >= x1, retorna uma linha degenerada.
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
    Bloco visual com base em y_base.

    O centro do bloco é a coordenada física.
    O formato é apenas um marcador visual.
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


def mostrar_controles_velocidade(prefixo):
    """
    Mantém os controles de velocidade das quatro abas,
    usando o mesmo estado global.
    """
    col_b1, col_b2 = st.columns(2)

    if col_b1.button("⏩ Mais Rápido", key=f"fast_{prefixo}"):
        indice = VELOCIDADES_MS.index(st.session_state.velocidade_ms)
        st.session_state.velocidade_ms = VELOCIDADES_MS[
            max(0, indice - 1)
        ]

    if col_b2.button("⏪ Mais Lento", key=f"slow_{prefixo}"):
        indice = VELOCIDADES_MS.index(st.session_state.velocidade_ms)
        st.session_state.velocidade_ms = VELOCIDADES_MS[
            min(len(VELOCIDADES_MS) - 1, indice + 1)
        ]

    st.markdown(
        f"""
        <b>Velocidade atual:</b>
        {st.session_state.velocidade_ms} ms/quadro
        """,
        unsafe_allow_html=True,
    )


# ============================================
# SOLUÇÃO DE MOVIMENTO CONSERVATIVO EM 1D
# ============================================
def mapa_fase_temporal(
    potencial,
    segmentos,
    x_inicial,
    massa,
    energia_total,
    pontos_por_segmento=6000,
):
    """
    Constrói uma tabela posição -> tempo para:

        Ec = Em - U
        v = sqrt(2(Em - U)/m)
        dt = dx/v

    A posição inicial está sempre em repouso.

    O primeiro movimento percorre de x_inicial até a primeira
    posição de retorno. Depois, a solução é espelhada no tempo,
    pois o sistema é reversível e conserva energia.

    A tabela permite interpolar a posição em instantes
    aproximadamente uniformes.
    """
    if massa <= 0 or energia_total <= 0:
        raise ValueError("A massa e a energia total devem ser positivas.")

    limites = np.asarray(segmentos, dtype=float)

    if x_inicial <= limites.min() or x_inicial >= limites.max():
        raise ValueError("A posição inicial deve estar no interior do domínio.")

    partes_x = []

    for xa, xb in limites:
        partes_x.append(
            np.linspace(xa, xb, pontos_por_segmento + 1)
        )

    x_tabela = np.unique(np.concatenate(partes_x))
    u_tabela = np.asarray(potencial(x_tabela), dtype=float)

    # Remove somente os extremos que correspondem aos pontos
    # de retorno com U(x) = Em.
    x_tabela = x_tabela[
        (x_tabela > limites.min() + 1e-12)
        & (x_tabela < limites.max() - 1e-12)
    ]
    u_tabela = np.asarray(potencial(x_tabela), dtype=float)

    v_tabela = np.sqrt(
        np.maximum(0.0, 2.0 * (energia_total - u_tabela) / massa)
    )

    if np.any(~np.isfinite(v_tabela)):
        raise ValueError("A tabela de velocidades contém valores inválidos.")

    dx = np.diff(x_tabela)
    velocidade_media = (v_tabela[:-1] + v_tabela[1:]) / 2.0

    # Em um ponto de retorno, a velocidade média do intervalo
    # não é zero porque o outro extremo do intervalo não é.
    velocidade_media = np.maximum(velocidade_media, 1e-12)

    tempo_ate_retorno = np.zeros_like(x_tabela)
    tempo_ate_retorno[1:] = np.cumsum(
        2.0 * dx / velocidade_media
    )

    # Segundo trecho: do retorno até a posição inicial.
    tempo_de_volta = tempo_ate_retorno[-1] + np.cumsum(
        2.0 * dx / velocidade_media
    )

    x_fase = np.concatenate([x_tabela, x_tabela[::-1]])
    t_fase = np.concatenate([tempo_ate_retorno, tempo_de_volta])

    ordem = np.argsort(t_fase, kind="stable")

    x_fase = x_fase[ordem]
    t_fase = t_fase[ordem]

    # Remove eventuais duplicações numéricas de instantes.
    t_fase, indices_unicos = np.unique(
        t_fase, return_index=True
    )
    x_fase = x_fase[indices_unicos]

    periodo = float(t_fase[-1])

    if not np.isfinite(periodo) or periodo <= 0:
        raise ValueError("Não foi possível determinar o período físico.")

    return x_fase, t_fase


def amostrar_periodo(
    potencial,
    segmentos,
    x_inicial,
    massa,
    energia_total,
    fps=60,
    minimo_amostras=180,
):
    """
    Retorna instantes uniformes e as posições físicas
    correspondentes dentro de um período.
    """
    x_fase, t_fase = mapa_fase_temporal(
        potencial,
        segmentos,
        x_inicial,
        massa,
        energia_total,
    )

    numero_amostras = max(
        minimo_amostras,
        int(math.ceil(t_fase[-1] * fps)),
    )

    tempos = np.linspace(0.0, t_fase[-1], numero_amostras)

    posicoes = np.interp(
        tempos,
        t_fase,
        x_fase,
    )

    return tempos, posicoes


# ============================================
# ENERGIA E BARRAS
# ============================================
def medir_energia_conservativa(
    potencial,
    massa,
    energia_total,
    posicoes,
    gravidade,
    tipo,
):
    """
    Calcula as energias a partir da posição física.

    A velocidade é obtida da conservação da energia,
    e não da diferença entre dois frames de animação.
    """
    x = np.asarray(posicoes, dtype=float)

    energia_potencial = np.asarray(potencial(x), dtype=float)

    velocidade_ao_quadrado = np.maximum(
        0.0,
        2.0 * (energia_total - energia_potencial) / massa,
    )

    ec = 0.5 * massa * velocidade_ao_quadrado

    epg = np.zeros_like(ec)
    epe = np.zeros_like(ec)

    if tipo == "gravitacional":
        epg = energia_potencial

    elif tipo == " elastica":
        epe = energia_potencial

    elif tipo == "mista":
        altura = np.where(
            x < -2.0,
            altura_max * ((x + 2.0) / 4.0) ** 2,
            0.0,
        )

        epg = massa * gravidade * altura
        epe = energia_potencial - epg

    return ec, epg, epe


def criar_barras_energia(ec, epg, epe, rotulos):
    """
    Cria o gráfico de barras de energia.
    """
    valores = {
        "Ec": np.asarray(ec, dtype=float),
        "Epg": np.asarray(epg, dtype=float),
        "Epe": np.asarray(epe, dtype=float),
        "Em": (
            np.asarray(ec, dtype=float)
            + np.asarray(epg, dtype=float)
            + np.asarray(epe, dtype=float)
        ),
    }

    alturas = [valores[nome] for nome in rotulos]
    cores = [CORES_ENERGIA[nome] for nome in rotulos]

    textos = [
        f"{valor:.2f} J"
        for valor in alturas
    ]

    return go.Bar(
        x=rotulos,
        y=alturas,
        marker_color=cores,
        text=textos,
        textposition="auto",
        hovertemplate="%{x}: %{y:.4f} J<extra></extra>",
    )


def criar_frames_energia(tempos, ec, epg, epe, rotulos):
    """
    Cada frame recebe valores de energia calculados na posição
    física daquele instante, e não por um ângulo artificial.
    """
    frames = []

    for i, tempo in enumerate(tempos):
        valores = {
            "Ec": ec[i],
            "Epg": epg[i],
            "Epe": epe[i],
            "Em": ec[i] + epg[i] + epe[i],
        }

        alturas = [valores[nome] for nome in rotulos]

        textos = [
            f"{valor:.2f} J"
            for valor in alturas
        ]

        frames.append(
            go.Frame(
                data=[
                    go.Bar(
                        y=alturas,
                        text=textos,
                        hovertemplate="%{y:.4f} J<extra></extra>",
                    )
                ],
                traces=[-1],
                name=f"t = {tempo:.5f} s",
            )
        )

    return frames


def configurar_animacao(
    figura,
    tempos,
    frames,
    duracao_ms,
    reproducao_continua,
):
    """
    Configura Play, Pause, reinício e navegação temporal.

    A duração dos frames é constante, mas isso não altera
    a física: os estados exibidos já vêm das soluções físicas.
    """
    tempos = np.asarray(tempos, dtype=float)
    dt_fisico = tempos[1] - tempos[0]

    # Uns poucos quadros adicionais exibem o estado inicial
    # novamente para suavizar uma reprodução contínua.
    quantidade_parada = 12

    tempos_com_parada = np.concatenate(
        [
            tempos,
            np.full(quantidade_parada, tempos[0]),
        ]
    )

    frame_final = frames[0].copy()
    frames = list(frames) + [frame_final] * quantidade_parada

    figura.frames = frames

    reproducao = (
        None
        if reproducao_continua
        else [0]
    )

    opcoes_play = {
        "frame": {
            "duration": int(duracao_ms),
            "redraw": True,
        },
        "fromcurrent": True,
        "transition": {
            "duration": 0,
            "easing": "linear",
        },
        "mode": "immediate",
    }

    botoes = [
        {
            "label": "▶ Play",
            "method": "animate",
            "args": [reproducao, opcoes_play],
        },
        {
            "label": "❚❚ Pause",
            "method": "animate",
            "args": [
                [None],
                {
                    "frame": {
                        "duration": 0,
                        "redraw": False,
                    },
                    "mode": "immediate",
                    "transition": {
                        "duration": 0,
                    },
                },
            ],
        },
        {
            "label": "↺ Reiniciar",
            "method": "animate",
            "args": [
                [0],
                {
                    "frame": {
                        "duration": 0,
                        "redraw": True,
                    },
                    "mode": "immediate",
                    "transition": {
                        "duration": 0,
                    },
                },
            ],
        },
    ]

    # Navegação por uma seleção de instantes do movimento.
    numero_passos = min(100, max(2, len(tempos) - 1))
    indices_slider = np.unique(
        np.linspace(
            0,
            len(tempos) - 1,
            numero_passos,
        ).astype(int)
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
                    "transition": {
                        "duration": 0,
                    },
                },
            ],
        }
        for i in indices_slider
    ]

    figura.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=55, b=10),
        height=430,
        uirevision="simulacao-conservativa",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.14,
                "buttons": botoes,
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.12,
                "y": 1.13,
                "len": 0.86,
                "pad": {"t": 35, "b": 10},
                "currentvalue": {
                    "prefix": "Tempo: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "steps": passos,
            }
        ],
    )

    return dt_fisico


def publicar_animacao(
    figura,
    tempos,
    ec,
    epg,
    epe,
    duracao_ms,
    reproducao_continua,
    passo_integrador=None,
):
    """
    Publica a animação e mostra o diagnóstico energético.
    """
    rotulos = [str(trace.name) for trace in figura.data if isinstance(trace, go.Bar)]

    if not rotulos:
        raise ValueError("A figura precisa conter um gráfico de barras.")

    frames = criar_frames_energia(
        tempos,
        ec,
        epg,
        epe,
        rotulos,
    )

    dt_fisico = configurar_animacao(
        figura,
        tempos,
        frames,
        duracao_ms,
        reproducao_continua,
    )

    st.plotly_chart(
        figura,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
        },
    )

    energia_inicial = ec[0] + epg[0] + epe[0]
    energia_total = ec + epg + epe

    erro_relativo = np.max(
        np.abs(energia_total - energia_inicial)
    ) / max(abs(energia_inicial), 1e-12)

    informacao_tempo = (
        f"Δt entre amostras: <b>{dt_fisico:.6f} s</b>"
    )

    if passo_integrador is not None:
        informacao_tempo += (
            f" | passo interno do integrador:"
            f" <b>{passo_integrador:.8f} s</b>"
        )

    st.markdown(
        f"""
        <div class="concept-card"
             style="border-left-color: #7f8c8d; padding: 0.8rem;">
            {informacao_tempo}<br>
            Energia mecânica inicial:
            <b>{energia_inicial:.4f} J</b><br>
            Erro relativo máximo de conservação:
            <b>{erro_relativo:.3e}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return dt_fisico


# ============================================
# GERAÇÃO DE FIGURAS: ABA 1 — RAMPA EM U
# ============================================
def gerar_figura_rampa_u(
    massa,
    altura_max,
    duracao_ms,
    gravidade=9.81,
):
    x_max = 5.0

    # Potencial medido em relação ao fundo da pista.
    coeficiente = altura_max / (x_max**2)

    def potencial(x):
        x = np.asarray(x, dtype=float)
        return massa * gravidade * coeficiente * x**2

    energia_inicial = massa * gravidade * altura_max

    tempos, posicoes = amostrar_periodo(
        potencial,
        segmentos=[
            (x_inicial := -x_max, 0.0),
            (0.0, x_max),
        ],
        x_inicial=x_inicial,
        massa=massa,
        energia_total=energia_inicial,
    )

    ec, epg, epe = medir_energia_conservativa(
        potencial,
        massa,
        energia_inicial,
        posicoes,
        gravidade,
        tipo="gravitacional",
    )

    figura = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.08,
    )

    # Pista.
    x_pista = np.linspace(-x_max, x_max, 100)
    y_pista = coeficiente * x_pista**2

    figura.add_trace(
        go.Scatter(
            x=x_pista,
            y=y_pista,
            mode="lines",
            line=dict(color="#7f8c8d", width=4),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Esfera.
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

    figura.add_trace(
        criar_barras_energia(
            ec[:1],
            epg[:1],
            epe[:1],
            ["Ec", "Epg", "Em"],
        ),
        row=1,
        col=2,
    )

    figura.update_xaxes(
        range=[-6.5, 6.5],
        showgrid=False,
        zeroline=False,
        visible=False,
        row=1,
        col=1,
    )

    figura.update_yaxes(
        range=[-1, altura_max + 2.5],
        showgrid=False,
        zeroline=False,
        visible=False,
        row=1,
        col=1,
    )

    figura.update_yaxes(
        range=[0, max(10.0, float(ec.max() + epg.max()) * 1.25)],
        title="Energia (Joules)",
        row=1,
        col=2,
    )

    publicar_animacao(
        figura,
        tempos,
        ec,
        epg,
        epe,
        duracao_ms,
        st.session_state.reproducao_continua,
    )


# ============================================
# GERAÇÃO DE FIGURAS: ABA 2 — MASSA–MOLA
# ============================================
def gerar_figura_massa_mola(
    massa,
    k_mola,
    amplitude,
    duracao_ms,
):
    """
    A amplitude é a distância máxima entre o bloco e a parede
    no estado de equilíbrio da mola.
    """
    x_ancora = -amplitude
    x_inicial = x_ancora + amplitude

    largura_bloco = 0.8
    x_direita_bloco = x_inicial + largura_bloco / 2.0
    x_parede = x_direita_bloco + 0.7

    def potencial(x):
        extensao = np.asarray(x, dtype=float) - x_ancora
        return 0.5 * k_mola * extensao**2

    energia_inicial = 0.5 * k_mola * amplitude**2

    tempos, posicoes = amostrar_periodo(
        potencial,
        segmentos=[
            (x_inicial, 0.0),
            (0.0, x_ancora),
        ],
        x_inicial=x_inicial,
        massa=massa,
        energia_total=energia_inicial,
    )

    ec, epg, epe = medir_energia_conservativa(
        potencial,
        massa,
        energia_inicial,
        posicoes,
        gravidade=0.0,
        tipo="elastica",
    )

    figura = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.08,
    )

    # Pista.
    figura.add_trace(
        go.Scatter(
            x=[
                x_parede - 1.0,
                x_direita_bloco + 1.0,
                x_direita_bloco + 1.0,
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

    # Parede fixa.
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

    molas = []

    for x in posicoes:
        xm, ym = criar_mola(
            x_ancora,
            x - largura_bloco / 2.0,
            0.4,
            n_voltas=12,
        )

        molas.append((xm, ym))

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

    bx, by = criar_bloco(x_inicial, 0.0)

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

    figura.add_trace(
        criar_barras_energia(
            ec[:1],
            epg[:1],
            epe[:1],
            ["Ec", "Epe", "Em"],
        ),
        row=1,
        col=2,
    )

    figura.update_xaxes(
        range=[x_parede - 0.5, x_direita_bloco + 1.5],
        showgrid=False,
        zeroline=False,
        visible=False,
        row=1,
        col=1,
    )

    figura.update_yaxes(
        range=[-1.2, 2.8],
        showgrid=False,
        zeroline=False,
        visible=False,
        row=1,
        col=1,
    )

    figura.update_yaxes(
        range=[0, max(10.0, float((ec + epe).max()) * 1.25)],
        title="Energia (Joules)",
        row=1,
        col=2,
    )

    publicar_animacao(
        figura,
        tempos,
        ec,
        epg,
        epe,
        duracao_ms,
        st.session_state.reproducao_continua,
    )

    st.caption(
        f"""
        Posição de equilíbrio: **x = {x_ancora:.2f} m**.  
        Amplitude: **{amplitude:.2f} m**.  
        Período teórico:
        **T = 2π√(m/k) = {2.0 * math.pi * math.sqrt(massa / k_mola):.4f} s**.
        """
    )


# ============================================
# GERAÇÃO DE FIGURAS: ABA 3 — RAMPA + MOLA
# ============================================
def gerar_figura_rampa_mola_ref(
    massa,
    altura_max,
    k_mola,
    duracao_ms,
    gravidade=9.81,
):
    x_topo_rampa = -6.0
    x_base_rampa = -2.0
    x_inicio_mola = 2.0

    largura_bloco = 0.7
    metade_bloco = largura_bloco / 2.0

    energia_inicial = massa * gravidade * altura_max

    # Posição do bloco no ponto de compressão máxima.
    compressao_maxima = math.sqrt(
        2.0 * energia_inicial / k_mola
    )

    # A parede é afastada para que a compressão máxima
    # não atravesse a parede.
    x_parede = (
        x_inicio_mola
        + 1.2 * compressao_maxima
        + 1.0
    )

    x_final_sem_parede = x_parede - metade_bloco - 0.6

    def altura_rampa(x):
        x = np.asarray(x, dtype=float)

        return np.where(
            x < x_base_rampa,
            altura_max
            * (
                (x - x_base_rampa)
                / (x_topo_rampa - x_base_rampa)
            ) ** 2,
            0.0,
        )

    def potencial(x):
        x = np.asarray(x, dtype=float)

        altura = altura_rampa(x)

        energia_gravitacional = massa * gravidade * altura

        # A mola está comprimida à medida que o bloco avança.
        # A extremidade esquerda continua FIXA em x_inicio_mola.
        extensao = x_parede - (x + metade_bloco)

        energia_elastica = np.where(
            x > x_inicio_mola,
            0.5 * k_mola * extensao**2,
            0.0,
        )

        return energia_gravitacional + energia_elastica

    tempos, posicoes = amostrar_periodo(
        potencial,
        segmentos=[
            (x_topo_rampa, x_base_rampa),
            (x_base_rampa, x_inicio_mola),
            (x_inicio_mola, x_final_sem_parede),
        ],
        x_inicial=x_topo_rampa,
        massa=massa,
        energia_total=energia_inicial,
    )

    ec, epg, epe = medir_energia_conservativa(
        potencial,
        massa,
        energia_inicial,
        posicoes,
        gravidade,
        tipo="mista",
    )

    figura = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.08,
    )

    xr = np.linspace(
        x_topo_rampa,
        x_base_rampa,
        40,
    )

    yr = altura_rampa(xr)

    xp_plano = np.linspace(
        x_base_rampa,
        x_parede,
        40,
    )

    yp_plano = np.zeros_like(xp_plano)

    figura.add_trace(
        go.Scatter(
            x=np.concatenate([xr, xp_plano]),
            y=np.concatenate([yr, yp_plano]),
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
        x_direita_bloco = x + metade_bloco

        if x_direita_bloco > x_inicio_mola:
            xm, ym = criar_mola(
                x_inicio_mola,
                x_direita_bloco,
                0.35,
                n_voltas=12,
            )
        else:
            # A mola continua sendo uma entitlement do sistema,
            # mas não é desenhada esticada quando não está em uso.
            xm = [np.nan, np.nan]
            ym = [0.35, 0.35]

        molas.append((xm, ym))

    figura.add_trace(
        go.Scatter(
            x=molas[0][0],
            y=molas[0][1],
            mode="lines",
            line=dict(color="#2ecc71", width=3),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    bx, by = criar_bloco(
        x_topo_rampa,
        altura_max,
        largura_bloco,
        largura_bloco,
    )

    figura.add_trace(
        go.Scatter(
            x=bx,
            y=by,
            fill="toself",
            fillcolor="#e74c3c",
            line=dict(color="#c0392b", width=2),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    figura.add_trace(
        criar_barras_energia(
            ec[:1],
            epg[:1],
            epe[:1],
            ["Ec", "Epg", "Epe", "Em"],
        ),
        row=1,
        col=2,
    )

    figura.update_xaxes(
        range=[x_topo_rampa - 1, x_parede + 1],
        showgrid=False,
        zeroline=False,
        visible=False,
        row=1,
        col=1,
    )

    figura.update_yaxes(
        range=[-0.5, altura_max + 1.5],
        showgrid=False,
        zeroline=False,
        visible=False,
        row=1,
        col=1,
    )

    figura.update_yaxes(
        range=[
            0,
            max(10.0, float((ec + epg + epe).max()) * 1.25),
        ],
        title="Energia (Joules)",
        row=1,
        col=2,
    )

    publicar_animacao(
        figura,
        tempos,
        ec,
        epg,
        epe,
        duracao_ms,
        st.session_state.reproducao_continua,
    )

    st.caption(
        f"""
        Compressão máxima teórica da mola:
        **x_max = {compressao_maxima:.3f} m**.  
        A parede fica em **x = {x_parede:.3f} m** para evitar
        interferência com o bloco na compressão máxima.
        """
    )


# ============================================
# LOOPING: POTENCIAL, CRITÉRIOS E INTEGRAÇÃO LIVRE
# ============================================
def resumo_looping(raio, velocidade_inicial, gravidade):
    energia_inicial = 0.5 * velocidade_inicial**2

    altura_maxima = energia_inicial / gravidade
    altura_topo = 2.0 * raio

    razao = velocidade_inicial**2 / (gravidade * raio)

    pode_chegar_ao_topo = (
        altura_maxima >= altura_topo
    )

    # N/mg = 1 + v²/(gR) + cos(theta).
    #
    # Para manter contato por dentro em todo o trecho superior:
    # v_top² >= 5gR.
    contato_continuo = razao >= 5.0

    return {
        "energia_inicial": energia_inicial,
        "altura_maxima": altura_maxima,
        "altura_topo": altura_topo,
        "velocidade_topo_minima": math.sqrt(gravidade * raio),
        "velocidade_livre_minima": math.sqrt(5.0 * gravidade * raio),
        "pode_chegar_ao_topo": pode_chegar_ao_topo,
        "contato_continuo": contato_continuo,
        "razao": razao,
    }


def trajetoria_loop_preso(
    raio,
    velocidade_inicial,
    gravidade,
    duracao=None,
):
    """
    Movimento exato do centro material no trilho circular,
    descrito como uma coordenada unidimensional.
    """
    x_entrada = raio
    y_centro = raio

    def potencial(x):
        x = np.asarray(x, dtype=float)

        altura = (
            y_centro
            + np.sqrt(
                np.maximum(0.0, raio**2 - (x - x_entrada) ** 2)
            )
        )

        return massa * gravidade * altura

    # A massa se cancela no período de um potencial
    # independente da velocidade.
    massa = 1.0
    energia_normalizada = 0.5 * velocidade_inicial**2

    if velocidade_inicial**2 >= 2.0 * gravidade * raio:
        # Alcança o topo.
        x_topo = x_entrada + math.pi * raio

        _, tempos_ate_topo = mapa_fase_temporal(
            potencial,
            segmentos=[(x_entrada, x_topo)],
            x_inicial=x_entrada,
            massa=massa,
            energia_total=energia_normalizada,
        )

        periodo_meia_volta = float(tempos_ate_topo[-1])

    else:
        # Não alcança o topo. Existe um ponto de retorno.
        cosseno_limite = 1.0 - (
            velocidade_inicial**2
            / (2.0 * gravidade * raio)
        )

        theta_limite = math.acos(
            np.clip(cosseno_limite, -1.0, 1.0)
        )

        x_retorno = x_entrada + raio * math.sin(theta_limite)

        _, tempos_ate_retorno = mapa_fase_temporal(
            potencial,
            segmentos=[(x_entrada, x_retorno)],
            x_inicial=x_entrada,
            massa=massa,
            energia_total=energia_normalizada,
        )

        periodo_meia_volta = float(tempos_ate_retorno[-1])

    if duracao is None:
        if (
            velocidade_inicial**2 >= 2.0 * gravidade * raio
        ):
            duracao = 2.0 * periodo_meia_volta
        else:
            duracao = 2.0 * periodo_meia_volta

    numero_amostras = max(
        180,
        int(math.ceil(duracao * 60)),
    )

    tempos = np.linspace(0.0, duracao, numero_amostras)

    x_fase, t_fase = mapa_fase_temporal(
        potencial,
        segmentos=[
            (x_entrada, x_entrada + math.pi * raio)
        ],
        x_inicial=x_entrada,
        massa=massa,
        energia_total=energia_normalizada,
    )

    # A tabela do mapa possui um período.
    x_periodo = np.interp(
        tempos,
        t_fase,
        x_fase,
    )

    px = x_periodo
    py = y_centro + np.sqrt(
        np.maximum(0.0, raio**2 - (px - x_entrada) ** 2)
    )

    velocidade_ao_quadrado = np.maximum(
        0.0,
        2.0
        * (
            0.5 * velocidade_inicial**2
            - gravidade * py
        ),
    )

    velocidade_absoluta = np.sqrt(velocidade_absoluta := velocidade_ao_quadrado)

    sentido = np.where(
        np.arange(len(px)) < len(px) / 2.0,
        1.0,
        -1.0,
    )

    vx = sentido * velocidade_absoluta
    vy = -((px - x_entrada) / raio) * vx

    return tempos, px, py, vx, vy


def primeiro_retorno_ao_circulo(
    posicao,
    velocidade,
    raio,
    gravidade,
):
    """
    Procura o primeiro instante futuro em que a trajetória
    balística cruza novamente a circunferência.
    """
    gravidade_vetor = np.array([0.0, -gravidade], dtype=float)

    def distancia_circunferencia(t):
        p = (
            posicao
            + velocidade * t
            + 0.5 * gravidade_vetor * t**2
        )
        return float(np.dot(p, p) - raio**2)

    passo = 0.002

    # Tempo limite apenas para a busca numérica.
    limite = 4.0 * math.pi * math.sqrt(raio / gravidade)

    instantes = np.arange(
        passo,
        limite + passo,
        passo,
    )

    valores = np.array([
        distancia_circunferencia(t)
        for t in instantes
    ])

    candidatos = np.where(valores <= 0.0)[0]

    if len(candidatos) == 0:
        return None

    indice = int(candidatos[0])

    if indice == 0:
        return float(instantes[0])

    t_alto = float(instantes[indice])
    t_baixo = float(instantes[indice - 1])

    for _ in range(60):
        t_meio = 0.5 * (t_alto + t_baixo)

        if distancia_circunferencia(t_meio) > 0.0:
            t_baixo = t_meio
        else:
            t_alto = t_meio

    return 0.5 * (t_alto + t_baixo)


def passo_livre_rk4(
    posicao,
    velocidade,
    raio,
    gravidade,
    dt,
    em_contato,
):
    """
    Integração RK4 para o corpo livre.

    Quando em contato:
        a_t = g_t/R - v²/R²

    A força normal pode ser negativa. Nesse caso, o corpo
    deixa de ser guiado pela circunferência.

    Quando fora da pista:
        a = (0, -g)
    """
    g = np.array([0.0, -gravidade], dtype=float)

    if em_contato:
        tangente = np.array(
            [-posicao[1], posicao[0]],
            dtype=float,
        ) / raio

        velocidade_tangencial = float(
            np.dot(velocidade, tangente)
        )

        aceleracao_tangencial = (
            np.dot(g, tangente) / raio
            - velocidade_tangencial**2 / raio**2
        )

        def derivadas(p, v):
            a = aceleracao_tangencial * tangente
            return v, a

    else:
        def derivadas(p, v):
            return v, g

    k1_p, k1_v = derivadas(posicao, velocidade)

    k2_p, k2_v = derivadas(
        posicao + 0.5 * dt * k1_p,
        velocidade + 0.5 * dt * k1_v,
    )

    k3_p, k3_v = derivadas(
        posicao + 0.5 * dt * k2_p,
        velocidade + 0.5 * dt * k2_v,
    )

    k4_p, k4_v = derivadas(
        posicao + dt * k3_p,
        velocidade + dt * k3_v,
    )

    nova_posicao = posicao + dt / 6.0 * (
        k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p
    )

    nova_velocidade = velocidade + dt / 6.0 * (
        k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v
    )

    if em_contato:
        distancia = float(np.linalg.norm(nova_posicao))

        if distancia < 1e-12:
            direcao = np.array([1.0, 0.0])
        else:
            direcao = nova_posicao / distancia

        # Projeção geométrica na circunferência.
        nova_posicao = raio * direcao

        tangente_ccw = np.array(
            [-nova_posicao[1], nova_posicao[0]],
            dtype=float,
        ) / raio

        sentido = (
            1.0
            if np.dot(nova_velocidade, tangente_ccw) >= 0.0
            else -1.0
        )

        velocidade_tangencial = sentido * np.dot(
            nova_velocidade,
            tangente_ccw,
        )

        # Projeta a velocidade na tangente.
        # A componente normal é removida porque o trilho
        # pode impulses uma reação normal.
        nova_velocidade = (
            velocidade_tangencial * tangente_ccw
        )

        normal_por_unidade_massa = (
            velocidade_tangencial**2 / raio
            + gravidade * nova_posicao[1] / raio
        )

        if normal_por_unidade_massa < 0.0:
            # O trilho não pode puxar o corpo para fora.
            em_contato = False

    return (
        nova_posicao,
        nova_velocidade,
        em_contato,
    )


def trajetoria_loop_livre(
    raio,
    velocidade_inicial,
    gravidade,
    numero_amostras=600,
):
    """
    Simulação bidimensional com possibilidade de descolamento.

    A separação ocorre quando a reação normal calculada
    para manter contato se torna negativa.
    """
    x_entrada = raio
    y_centro = raio

    # Condição geométrica de descolamento no ramo direito:
    #
    # cos(theta_d) = v0²/(gR) - 2
    #
    # A separação só é possível antes do topo quando
    # 2gR < v0² < 5gR.
    theta_descolamento = math.acos(
        np.clip(
            velocidade_inicial**2 / (gravidade * raio) - 2.0,
            -1.0,
            1.0,
        )
    )

    posicao = np.array(
        [
            x_entrada + raio * math.sin(theta_descolamento),
            y_centro + raio * math.cos(theta_descolamento),
        ],
        dtype=float,
    )

    tangente = np.array(
        [
            math.cos(theta_descolamento),
            -math.sin(theta_descolamento),
        ],
        dtype=float,
    )

    velocidade_topo = math.sqrt(
        max(
            0.0,
            velocidade_inicial**2
            - 2.0 * gravidade * (2.0 * raio),
        )
    )

    velocidade_descolamento = math.sqrt(
        max(0.0, -gravidade * raio * math.cos(theta_descolamento))
    )

    velocidade = velocidade_descolamento * tangente

    _, tempos_ate_topo = mapa_fase_temporal(
        lambda x: gravidade * (
            raio
            + np.sqrt(np.maximum(0.0, raio**2 - (x - raio) ** 2))
        ),
        segmentos=[(raio, raio + math.pi * raio)],
        x_inicial=raio,
        massa=1.0,
        energia_total=0.5 * velocidade_inicial**2,
    )

    duracao = 2.0 * float(tempos_ate_topo[-1])

    tempo_primeiro_cruzamento = primeiro_retorno_ao_circulo(
        posicao,
        velocidade,
        raio,
        gravidade,
    )

    if tempo_primeiro_cruzamento is None:
        tempo_primeiro_cruzamento = duracao

    # Passo interno fixo: não depende da velocidade do player.
    passo_integrador = 1.0 / 480.0

    tempos = np.linspace(
        0.0,
        duracao,
        numero_amostras,
    )

    lista_x = []
    lista_y = []

    tempo = 0.0
    em_contato = False

    for tempo_alvo in tempos:
        while tempo < tempo_alvo - 1e-12:
            # Depois do primeiro cruzamento, o corpo pode
            # estar sobre a parte inferior do trilho.
            em_contato = (
                tempo >= tempo_primeiro_cruzamento - 1e-12
            )

            dt = min(
                passo_integrador,
                tempo_alvo - tempo,
            )

            posicao, velocidade, em_contato = passo_livre_rk4(
                posicao,
                velocidade,
                raio,
                gravidade,
                dt,
                em_contato,
            )

            tempo += dt

        lista_x.append(posicao[0])
        lista_y.append(posicao[1])

    return (
        tempos,
        np.asarray(lista_x),
        np.asarray(lista_y),
        np.asarray(velocidade) * 0.0,  #Sobrescrito abaixo
        passo_integrador,
    )


# ============================================
# GERAÇÃO DE FIGURAS: ABA 4 — LOOPING
# ============================================
def gerar_figura_looping_corrigido(
    massa,
    raio,
    velocidade_inicial,
    duracao_ms,
    gravidade=9.81,
    modo="trilho",
):
    """
    modo:
        "trilho": o centro do objeto permanece na circunferência.
        "livre": permite perda de contato e movimento balístico.
    """
    resumo = resumo_looping(
        raio,
        velocidade_inicial,
        gravidade,
    )

    x_entrada = raio

    razao = resumo["razao"]

    usar_integracao_livre = (
        modo == "livre"
        and 2.0 < razao < 5.0
    )

    if usar_integracao_livre:
        (
            tempos,
            px,
            py,
            _,
            passo_integrador,
        ) = trajetoria_loop_livre(
            raio,
            velocidade_inicial,
            gravidade,
        )

        vx = np.gradient(px, tempos)
        vy = np.gradient(py, tempos)

    else:
        passo_integrador = None

        tempos, px, py, vx, vy = trajetoria_loop_preso(
            raio,
            velocidade_inicial,
            gravidade,
        )

    ec = 0.5 * massa * (vx**2 + vy**2)
    epg = massa * gravidade * py
    epe = np.zeros_like(ec)

    figura = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.08,
    )

    # Pista reta.
    x_linha = np.linspace(-5.0, x_entrada, 50)
    y_linha = np.zeros_like(x_linha)

    # Loop com parametrização geométrica, não temporal.
    theta_loop = np.linspace(math.pi, -math.pi, 180)

    x_loop = x_entrada + raio * np.sin(theta_loop)
    y_loop = raio + raio * np.cos(theta_loop)

    figura.add_trace(
        go.Scatter(
            x=np.concatenate([x_linha, x_loop]),
            y=np.concatenate([y_linha, y_loop]),
            mode="lines",
            line=dict(color="#7f8c8d", width=4),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    cx, cy = criar_bloco(
        -5.0,
        0.0,
        largura=0.6,
        altura=0.6,
    )

    figura.add_trace(
        go.Scatter(
            x=cx,
            y=cy,
            fill="toself",
            fillcolor="#e74c3c",
            line=dict(color="#c0392b", width=2),
            hovertemplate=(
                "x = %{x:.3f} m"
                "<br>y = %{y:.3f} m"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    figura.add_trace(
        criar_barras_energia(
            ec[:1],
            epg[:1],
            epe[:1],
            ["Ec", "Epg", "Em"],
        ),
        row=1,
        col=2,
    )

    x_minimo = min(-6.0, float(np.min(px) - 1.0))
    x_maximo = max(
        3.0 * raio,
        float(np.max(px) + 1.0),
    )

    y_maximo = max(
        2.0 * raio + 2.0,
        float(np.max(py) + 2.0),
    )

    figura.update_xaxes(
        range=[x_minimo, x_maximo],
        showgrid=False,
        zeroline=False,
        visible=False,
        row=1,
        col=1,
    )

    figura.update_yaxes(
        range=[-1.0, y_maximo],
        showgrid=False,
        zeroline=False,
        visible=False,
        row=1,
        col=1,
    )

    figura.update_yaxes(
        range=[0, max(10.0, float((ec + epg).max()) * 1.25)],
        title="Energia (Joules)",
        row=1,
        col=2,
    )

    publicar_animacao(
        figura,
        tempos,
        ec,
        epg,
        epe,
        duracao_ms,
        st.session_state.reproducao_continua,
        passo_integrador=passo_integrador,
    )

    return resumo


# ============================================
# TÍTULO E NAVEGAÇÃO POR ABAS SUPERIORES
# ============================================
st.markdown(
    """
    <div class="main-title">
        ⚡ Sistemas Conservativos e Dinâmica
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Simulações físicas completas com equações,
        conservação de energia e controle de reprodução
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================
# CONFIGURAÇÕES GLOBAIS
# ============================================
with st.expander(
    "⚙️ Configurações globais da simulação",
    expanded=False,
):
    st.slider(
        "Gravidade (m/s²)",
        min_value=1.0,
        max_value=15.0,
        value=9.81,
        step=0.01,
        key="gravidade_global",
    )

    st.checkbox(
        "Reprodução contínua ao pressionar Play",
        value=False,
        key="reproducao_continua",
        help=(
            "Se ativada, a animação reinicia ao chegar ao fim. "
            "A repetição é visual; a física continua sendo calculada "
            "para o mesmo estado inicial."
        ),
    )

    st.caption(
        """
        Alterar a velocidade ou os parâmetros da simulação faz o
        Streamlit recalcular os estados físicos. Os botões de
        Play/Pause continuam内部控制ando a animação.
        """
    )


# ============================================
# ABAS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Rampa em 'U' (Gravitacional)",
        "2. Sistema Massa-Mola (Elástica)",
        "3. Rampa Inclinada + Mola",
        "4. Brinquedo Looping",
    ]
)


# ============================================
# ABA 1: RAMPA EM U
# ============================================
with tab1:
    st.markdown(
        """
        <div class="concept-card" style="border-left-color: #9b59b6;">
            <b>Princípio e Equações:</b>
            Esfera oscilando livremente em pista sem atrito.
            A energia mecânica total é conservada, convertendo-se
            continuamente entre energia cinética e energia potencial
            gravitacional. A velocidade é obtida de
            <b>v = √[2(Em − Epg)/m]</b> e o tempo é calculado a partir
            de <b>dt = dx/v</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
        $$
        E_m = E_c + E_{pg}
        = \frac{1}{2}mv^2 + mgh
        = \text{constante}
        $$
        """
    )

    col_c1, col_c2 = st.columns([1, 2.5])

    with col_c1:
        st.markdown(
            "<div class='param-box'>",
            unsafe_allow_html=True,
        )

        massa_u = st.slider(
            "Massa da esfera (kg)",
            1.0,
            10.0,
            2.0,
            step=0.5,
            key="mu",
        )

        h_max_u = st.slider(
            "Altura inicial (m)",
            2.0,
            10.0,
            5.0,
            step=0.5,
            key="hu",
        )

        mostrar_controles_velocidade("u")

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with col_c2:
        gerar_figura_rampa_u(
            massa=massa_u,
            altura_max=h_max_u,
            duracao_ms=st.session_state.velocidade_ms,
            gravidade=st.session_state.gravidade_global,
        )


# ============================================
# ABA 2: SISTEMA MASSA–MOLA
# ============================================
with tab2:
    st.markdown(
        """
        <div class="concept-card" style="border-left-color: #2ecc71;">
            <b>Princípio e Equações:</b>
            Bloco oscilando horizontalmente preso a uma mola
            elástica. A mola possui uma extremidade fixa.
            A posição e a velocidade são compatíveis com a
            conservação da energia.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
        $$
        E_m = E_c + E_{pe}
        = \frac{1}{2}mv^2 + \frac{1}{2}kx^2
        = \text{constante}
        $$
        """
    )

    col_m1, col_m2 = st.columns([1, 2.5])

    with col_m1:
        st.markdown(
            "<div class='param-box'>",
            unsafe_allow_html=True,
        )

        massa_m = st.slider(
            "Massa do bloco (kg)",
            1.0,
            10.0,
            2.0,
            step=0.5,
            key="mm",
        )

        k_m = st.slider(
            "Constante elástica (N/m)",
            10,
            100,
            50,
            step=10,
            key="km",
        )

        amp_m = st.slider(
            "Amplitude (m)",
            1.0,
            5.0,
            3.0,
            step=0.5,
            key="ampm",
        )

        mostrar_controles_velocidade("m")

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with col_m2:
        gerar_figura_massa_mola(
            massa=massa_m,
            k_mola=k_m,
            amplitude=amp_m,
            duracao_ms=st.session_state.velocidade_ms,
        )


# ============================================
# ABA 3: RAMPA + MOLA
# ============================================
with tab3:
    st.markdown(
        """
        <div class="concept-card" style="border-left-color: #3498db;">
            <b>Princípio e Equações:</b>
            Bloco solto do alto da rampa.
            A energia potencial gravitacional transforma-se em
            energia cinética e, depois, em energia elástica.
            Na compressão máxima, o bloco para instantaneamente
            e retorna: ele não atravessa a parede.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
        $$
        mgh = \frac{1}{2}kx_{\max}^2
        \implies
        x_{\max} = \sqrt{\frac{2mgh}{k}}
        $$
        """
    )

    col_r1, col_r2 = st.columns([1, 2.5])

    with col_r1:
        st.markdown(
            "<div class='param-box'>",
            unsafe_allow_html=True,
        )

        massa_rm = st.slider(
            "Massa do bloco (kg)",
            1.0,
            10.0,
            2.0,
            step=0.5,
            key="m_rm",
        )

        h_rm = st.slider(
            "Altura inicial (h)",
            1.0,
            8.0,
            4.0,
            step=0.5,
            key="h_rm",
        )

        k_rm = st.slider(
            "Constante da mola (k — N/m)",
            20,
            200,
            100,
            step=10,
            key="k_rm",
        )

        mostrar_controles_velocidade("rm")

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with col_r2:
        gerar_figura_rampa_mola_ref(
            massa=massa_rm,
            altura_max=h_rm,
            k_mola=k_rm,
            duracao_ms=st.session_state.velocidade_ms,
            gravidade=st.session_state.gravidade_global,
        )


# ============================================
# ABA 4: BRINQUEDO LOOPING
# ============================================
with tab4:
    st.markdown(
        """
        <div class="concept-card" style="border-left-color: #e74c3c;">
            <b>Princípio e Equações:</b>
            O carrinho parte de uma linha horizontal com
            velocidade inicial. É necessário verificar tanto
            a energia para alcançar o topo quanto a possibilidade
            de manter contato com a parte interna da pista.
            No modelo livre, a trajetória passa a ser balística
            quando a reação normal exigida se torna negativa.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
        $$
        \frac{1}{2}mv_0^2
        =
        \frac{1}{2}mv_{\text{topo}}^2 + mg(2R)
        $$
        """

        r"""
        $$
        \frac{N_{\text{topo}}}{mg}
        =
        1+\frac{v_{\text{topo}}^2}{Rg}
        $

        $$
        v_{\text{topo}}^2 \geq 5Rg
        \quad\Rightarrow\quad
        v_0^2 \geq 5Rg
        $$
        """
    )

    col_l1, col_l2 = st.columns([1, 2.5])

    with col_l1:
        st.markdown(
            "<div class='param-box'>",
            unsafe_allow_html=True,
        )

        massa_l = st.slider(
            "Massa do carrinho (kg)",
            0.1,
            5.0,
            1.0,
            step=0.1,
            key="m_l",
        )

        raio_l = st.slider(
            "Raio do Looping (R — m)",
            0.5,
            3.0,
            1.0,
            step=0.25,
            key="r_l",
        )

        v_ini_l = st.slider(
            "Velocidade inicial (v₀ — m/s)",
            1.0,
            15.0,
            6.0,
            step=0.5,
            key="v_ini_l",
        )

        modo_loop = st.radio(
            "Modelo de contato",
            options=["trilho", "livre"],
            format_func=lambda valor: {
                "trilho": "Objeto guiado pelo trilho",
                "livre": "Objeto livre — pode se desprender",
            }[valor],
            horizontal=False,
            key="modo_loop",
        )

        mostrar_controles_velocidade("l")

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with col_l2:
        resumo = gerar_figura_looping_corrigido(
            massa=massa_l,
            raio=raio_l,
            velocidade_inicial=v_ini_l,
            duracao_ms=st.session_state.velocidade_ms,
            gravidade=st.session_state.gravidade_global,
            modo=modo_loop,
        )

        st.subheader("📊 Relatório de Viabilidade do Looping")

        col_met1, col_met2 = st.columns(2)

        col_met1.metric(
            "Altura Máxima Atingível (hₘₐₓ)",
            f"{resumo['altura_maxima']:.2f} m",
            help=(
                "Calculada pela conversão total da energia cinética "
                "inicial."
            ),
        )

        col_met2.metric(
            "Altura do Topo do Loop",
            f"{resumo['altura_topo']:.2f} m",
        )

        col_met3, col_met4 = st.columns(2)

        col_met3.metric(
            "Velocidade Mín. no Topo",
            f"{resumo['velocidade_topo_minima']:.2f} m/s",
            help=(
                "Critério N_topo ≥ 0, suficiente para o trilho "
                "não perder contato no topo."
            ),
        )

        col_met4.metric(
            "Velocidade Mín. para Contato Contínuo",
            f"{resumo['velocidade_livre_minima']:.2f} m/s",
            help=(
                "Critério mais restritivo para manter contato "
                "em toda a parte superior do trilho."
            ),
        )

        if not resumo["pode_chegar_ao_topo"]:
            st.error(
                """
                ❌ **Trajetória energeticamente inviável!**
                A velocidade inicial não permite alcançar o topo
                do looping. O carrinho para antes dele e retorna,
                sem atravessar a parede.
                """
            )

        elif modo_loop == "livre" and not resumo["contato_continuo"]:
            st.warning(
                """
                ⚠️ **Há energia suficiente para alcançar o topo,
                mas o contato não é mantido.**

                No modelo de corpo livre, o carrinho se desprende
                antes de completar o trajeto interno. Ele não
                consegue concluir a volta sem uma restrição
                que o obrigue a permanecer sobre a pista.
                """
            )

        else:
            st.success(
                """
                ✅ **Trajetória Viável!**

                O carrinho possui energia suficiente para alcançar
                o topo e retornar à entrada. No modelo selecionado,
                a trajetória pode ser completada.
                """
            )

        if modo_loop == "trilho" and not resumo["contato_continuo"]:
            st.info(
                """
                **Observação sobre o trilho:**
                embora a energia permita completar a volta,
                uma pista interna não rígida não poderia exerts
                uma força normal negativa. Selecione o modelo
                “Objeto livre” para observar a perda de contato.
                """
            )

        st.caption(
            """
            A massa não altera o período desses movimentos sem
            atrito, mas altera as energias e a velocidade de saída
            da rampa em U:增大 a massa aumenta a energia
            transferida, mas não altera a geometria temporal
            naquela pista.
            """
        )


# ============================================
# RODAPÉ
# ============================================
st.markdown("---")

st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.85rem; padding: 1rem;">
        ⚡ <b>Física Visual: Energia e Dinâmica</b>
        — Simulações com equações físicas,
        conservação de energia e controle temporal.
    </div>
    """,
    unsafe_allow_html=True,
)
