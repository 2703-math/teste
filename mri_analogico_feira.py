import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="MRI-62 | Console Experimental", page_icon="◉", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
html,body,[class*="css"]{font-family:'Share Tech Mono',monospace}
.stApp{background:#090d0c;color:#b9d8c1}
section[data-testid="stSidebar"]{background:#0b100e;border-right:1px solid #31513d}
.block-container{max-width:1500px;padding-top:1rem}
.crt{border:2px solid #42634c;border-radius:10px;background:#050807;box-shadow:inset 0 0 40px rgba(86,160,103,.08);padding:16px}
.label{color:#668f70;font-size:.78rem;letter-spacing:.14em}
.title{color:#c8e6cf;letter-spacing:.15em;font-size:1.25rem}
.status{border:1px solid #42634c;padding:7px;color:#aee8b7;background:#0b140e;text-align:center}
.warning{color:#d5c17a}
.stButton button{background:#17231b;color:#bfe4c5;border:1px solid #48694f}
</style>
""", unsafe_allow_html=True)

def phantom(n=256, kind="Corte axial"):
    y,x=np.mgrid[-1:1:complex(n),-1:1:complex(n)]
    r=np.sqrt(x*x+y*y)
    img=np.exp(-((x/.82)**2+(y/.92)**2)*3)
    img+=.55*np.exp(-(((x+.27)/.34)**2+((y+.10)/.47)**2)*3.2)
    img+=.50*np.exp(-(((x-.28)/.34)**2+((y-.08)/.47)**2)*3.2)
    img+=.75*np.exp(-((x/.20)**2+((y+.02)/.27)**2)*5)
    for cx,cy,s,a in [(-.38,-.34,.07,.9),(.37,-.31,.065,.75),(-.40,.34,.08,.65),(.40,.35,.07,.8),(-.05,.48,.06,.55),(.10,-.47,.055,.7)]:
        img+=a*np.exp(-(((x-cx)/s)**2+((y-cy)/s)**2)*3)
    if kind=="Corte cerebral":
        img=np.exp(-((x/.86)**2+(y/.88)**2)*3)
        img+=.9*np.exp(-((x/.58)**2+(y/.62)**2)*2)
        img-=.8*np.exp(-((x/.25)**2+((y+.02)/.34)**2)*4)
        img+=.45*np.exp(-(((x-.25)/.16)**2+((y-.05)/.22)**2)*4)
        img+=.45*np.exp(-(((x+.25)/.16)**2+((y-.05)/.22)**2)*4)
    return np.clip(img*(r<.98),0,1)

def reconstruction(progress, noise, acceleration, kind):
    base=phantom(256,kind)
    k=np.fft.fftshift(np.fft.fft2(base))
    h,w=base.shape; p=max(1,int(progress/100*h/2)); cy=h//2
    mask=np.zeros_like(base); mask[max(0,cy-p):min(h,cy+p+1),:]=1
    if acceleration>1:
        mask[:,::int(acceleration)]=np.maximum(mask[:,::int(acceleration)],1)
    sampled=k*mask
    rec=np.abs(np.fft.ifft2(np.fft.ifftshift(sampled)))
    rec/=max(rec.max(),1e-9)
    rng=np.random.default_rng(7)
    return base,sampled,np.clip(rec+rng.normal(0,noise/100,rec.shape),0,1)

if "running" not in st.session_state: st.session_state.running=False
if "progress" not in st.session_state: st.session_state.progress=8.0
if "kind" not in st.session_state: st.session_state.kind="Corte axial"

st.markdown('<div class="crt"><div class="title">◉ MRI-62 / CONSOLE EXPERIMENTAL DE RESSONÂNCIA MAGNÉTICA</div><div class="label">LABORATÓRIO DE FÍSICA — SIMULAÇÃO DIDÁTICA / MODO ANALÓGICO</div></div>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### CONTROLE DO EXPERIMENTO")
    st.session_state.kind=st.selectbox("TIPO DE CORTE",["Corte axial","Corte cerebral"])
    field=st.slider("CAMPO B₀ — tesla",.2,2.0,1.5,.1)
    rf=st.slider("PULSO RF — %",10,100,70)
    noise=st.slider("RUÍDO DO RECEPTOR — %",0,15,4)
    acceleration=st.slider("DENSIDADE DE AQUISIÇÃO",1.0,4.0,1.0,.5)
    duration=st.slider("VELOCIDADE DA VARREDURA",.05,.50,.12,.01)
    c1,c2=st.columns(2)
    with c1:
        if st.button("▶ INICIAR",use_container_width=True): st.session_state.running=True
    with c2:
        if st.button("■ PARAR",use_container_width=True): st.session_state.running=False
    if st.button("↺ RESETAR",use_container_width=True):
        st.session_state.progress=8.0; st.session_state.running=False; st.rerun()
    st.divider()
    st.code("B0 → RF → GRADIENTES → SINAL → RECONSTRUÇÃO",language=None)

gamma=42.58
larmor=gamma*field
t1=950/max(field,.2); t2=85/max(field,.2)
base,kspace,recon=reconstruction(st.session_state.progress,noise,acceleration,st.session_state.kind)

a,b,c,d=st.columns(4)
a.metric("CAMPO B₀",f"{field:.1f} T")
b.metric("FREQUÊNCIA DE LARMOR",f"{larmor:.1f} MHz")
c.metric("T1 DIDÁTICO",f"{t1:.0f} ms")
d.metric("T2 DIDÁTICO",f"{t2:.0f} ms")

left,right=st.columns([1.25,1])
with left:
    st.markdown('<div class="crt"><div class="label">MONITOR DE IMAGEM / RECONSTRUÇÃO</div>',unsafe_allow_html=True)
    fig=go.Figure(go.Heatmap(z=recon,colorscale=[[0,"#030403"],[.25,"#18301e"],[.55,"#557a5b"],[.78,"#a8c7ad"],[1,"#eef8ef"]],showscale=False))
    fig.update_layout(height=520,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="#050807",plot_bgcolor="#050807",xaxis=dict(visible=False),yaxis=dict(visible=False,scaleanchor="x",scaleratio=1))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.markdown(f'<div class="status">AQUISIÇÃO: {st.session_state.progress:05.1f}% | SINAL: {"ATIVO" if st.session_state.running else "STANDBY"}</div></div>',unsafe_allow_html=True)

with right:
    st.markdown('<div class="crt"><div class="label">OSCILOSCÓPIO — SINAL DE RM</div>',unsafe_allow_html=True)
    t=np.linspace(0,1,900); decay=np.exp(-5.5*t)
    signal=np.sin(2*np.pi*(18+5*rf/100)*t)*decay+.12*np.sin(2*np.pi*83*t)*decay
    rng=np.random.default_rng(12); signal+=rng.normal(0,noise/100,len(t))
    osc=go.Figure(go.Scatter(x=t,y=signal,mode="lines",line=dict(color="#a8d9af",width=1.5)))
    osc.update_layout(height=230,margin=dict(l=5,r=5,t=10,b=5),paper_bgcolor="#050807",plot_bgcolor="#050807",font=dict(color="#8bad92"),xaxis=dict(showgrid=True,gridcolor="#18241b"),yaxis=dict(showgrid=True,gridcolor="#18241b"))
    st.plotly_chart(osc,use_container_width=True,config={"displayModeBar":False})
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="crt"><div class="label">SEQUÊNCIA DO SISTEMA</div>',unsafe_allow_html=True)
    phases=[("01","CAMPO ESTÁTICO B₀",True),("02","PULSO DE RADIOFREQUÊNCIA",rf>0),("03","EXCITAÇÃO DOS PRÓTONS",st.session_state.progress>15),("04","RELAXAÇÃO T1 / T2",st.session_state.progress>30),("05","GRADIENTES ESPACIAIS",st.session_state.progress>45),("06","AQUISIÇÃO DO SINAL",st.session_state.progress>60),("07","RECONSTRUÇÃO",st.session_state.progress>80)]
    for n,name,active in phases:
        col="#bff4c6" if active else "#45614b"
        st.markdown(f'<div style="color:{col};padding:4px 0">[{n}] {name} {"●" if active else "○"}</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

st.markdown("### PAINEL DE AQUISIÇÃO")
p1,p2,p3=st.columns(3)
with p1:
    st.markdown('<div class="crt"><div class="label">K-SPACE / DADOS BRUTOS</div>',unsafe_allow_html=True)
    kval=np.log1p(np.abs(kspace)); kval/=max(kval.max(),1e-9)
    fk=go.Figure(go.Heatmap(z=kval,colorscale="Greys",showscale=False))
    fk.update_layout(height=260,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="#050807",xaxis=dict(visible=False),yaxis=dict(visible=False,scaleanchor="x"))
    st.plotly_chart(fk,use_container_width=True,config={"displayModeBar":False})
    st.markdown('</div>',unsafe_allow_html=True)
with p2:
    st.markdown('<div class="crt"><div class="label">CAMPO MAGNÉTICO / PRÓTONS</div>',unsafe_allow_html=True)
    figspin=go.Figure()
    for i in range(20):
        x0=i%5;y0=i//5;ang=i*.65+st.session_state.progress/45
        figspin.add_trace(go.Scatter(x=[x0,x0+.35*np.cos(ang)],y=[y0,y0+.35*np.sin(ang)],mode="lines",line=dict(color="#9fd4a8",width=2),showlegend=False))
    figspin.update_layout(height=260,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="#050807",plot_bgcolor="#050807",xaxis=dict(range=[-.5,4.8],visible=False),yaxis=dict(range=[-.7,4.8],visible=False,scaleanchor="x"))
    st.plotly_chart(figspin,use_container_width=True,config={"displayModeBar":False})
    st.markdown('</div>',unsafe_allow_html=True)
with p3:
    st.markdown('<div class="crt"><div class="label">REGISTRO DO EXPERIMENTO</div>',unsafe_allow_html=True)
    st.markdown(f"**EQUIPAMENTO:** MRI-62  \n**MODO:** ANALÓGICO / EXPERIMENTAL  \n**CAMPO:** {field:.1f} T  \n**RF:** {rf}%  \n**LARMOR:** {larmor:.1f} MHz  \n**AQUISIÇÃO:** {st.session_state.progress:.1f}%  \n**RUÍDO:** {noise}%")
    st.markdown('<p class="warning">ATENÇÃO: imagem simulada para ensino. Não é exame médico.</p>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

if st.session_state.running:
    st.session_state.progress=min(100,st.session_state.progress+2)
    if st.session_state.progress>=100: st.session_state.running=False
    time.sleep(duration)
    st.rerun()

st.markdown('<div style="text-align:center;color:#49664f;font-size:.75rem;padding:15px">MRI-62 — SIMULAÇÃO EDUCACIONAL • APARELHO FICTÍCIO DE ESTÉTICA ANALÓGICA.</div>',unsafe_allow_html=True)
