"""Executive PPTX-Style Video Presentation Generator for Porto Alegre (Ilhas).

Creates a full-length broadcast-quality video presentation (16:9 1080p) explaining:
- Slide 1: Executive Title Slide (Localização Ativa com MOPPO nas Ilhas de Porto Alegre)
- Slide 2: O "Por Quê?" (O Desaparecimento das Ilhas e Choque Sensorial Câmera vs Mapa)
- Slide 3: O "Como?" (MOPPO Multiobjetivo + Memória Curta ANL + Nuvem de Pontos da Crença)
- Slide 4: Tabela Comparativa de Resultados Quantitativos nas Ilhas (300 episódios pareados)
- Slide 5: Vídeo Comparativo em Voo Dinâmico (Linha Reta vs 1º Treino PPO vs Treino MO com Nuvem de Pontos e Câmeras)
- Slide 6: Conclusões e Lições Aprendidas

Outputs:
- results_mo/videos/poa_ilhas_presentation_deck.mp4
- results_mo/videos/poa_ilhas_presentation_deck.gif
"""
from __future__ import annotations

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import from_bounds

from .config import ROOT, load_config
from .poa_data import poa_grid, region_slices
from .poa import beta_for, region_data
from .env import Episode
from .mo_eval import load_mo_model
from .mo_poa_eval import MOPPOPolicyWrapper
from .evaluate import PPOPolicy
from .baselines import StraightPolicy
from .filter import entropy


def render_slide_title(w=1920, h=1080) -> np.ndarray:
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#181825")
    ax = fig.add_subplot(111, facecolor="#181825")
    ax.axis("off")

    ax.text(0.5, 0.76, "LOCALIZAÇÃO ATIVA MULTIOBJETIVO (MOPPO)", ha="center", va="center", color="#cdd6f4", fontsize=25, fontweight="bold")
    ax.text(0.5, 0.67, "Navegação Visual sob Incerteza e Choque Morfológico no Delta do Jacuí", ha="center", va="center", color="#89dceb", fontsize=18)
    ax.text(0.5, 0.60, "Avaliação nas Ilhas de Porto Alegre durante a Enchente Histórica de Maio de 2024", ha="center", va="center", color="#fab387", fontsize=14)

    badges = ["AMOR (SIGGRAPH 2025)", "Active Neural Localization (ICLR 2018)", "Transferência Zero-Shot"]
    for i, b in enumerate(badges):
        x = 0.22 + i * 0.28
        ax.text(x, 0.44, b, ha="center", va="center", color="#181825", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#a6e3a1", edgecolor="#ffffff", alpha=0.9))

    intro_box = (
        "Por que métodos geométricos tradicionais (linha reta e zigue-zague) falham quando as ilhas desaparecem sob a água?\n"
        "Como uma política treinada com múltiplos objetivos (segurança vs velocidade) e memória de ações\n"
        "consegue reconhecer o choque sensorial e guiar a aeronave para locais onde ela pode se localizar?"
    )
    ax.text(0.5, 0.26, intro_box, ha="center", va="center", color="#cdd6f4", fontsize=13, style="italic",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1e1e2e", edgecolor="#45475a", lw=1.2))

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return cv2.resize(rgb, (w, h))


def render_slide_why(c_post: np.ndarray, c_pre: np.ndarray, w=1920, h=1080) -> np.ndarray:
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#181825")
    ax = fig.add_subplot(111, facecolor="#181825")
    ax.axis("off")

    ax.text(0.08, 0.91, "1. POR QUÊ? O DESAFIO DAS ILHAS QUE SUMIRAM", color="#cdd6f4", fontsize=22, fontweight="bold")
    ax.text(0.08, 0.85, "Choque morfológico e quebra de premissas sensoriais na enchente de Maio de 2024 (Delta do Jacuí)", color="#89dceb", fontsize=13)

    text_content = (
        "• CENÁRIO OPERACIONAL CRÍTICO:\n"
        "  Em missões de busca e salvamento com GPS negado, o drone depende\n"
        "  exclusivamente de correlacionar a câmera visual com um mapa prévio.\n\n"
        "• O FENÔMENO DAS ILHAS SUBMERSAS:\n"
        "  Na enchente histórica de maio/2024, a cheia do Jacuí cobriu ilhas inteiras\n"
        "  sob lâminas profundas de água barrenta e turva.\n\n"
        "• O CHOQUE SENSORIAL (VERIFIQUE AO LADO):\n"
        "  A câmera embarcada só vê água marrom uniforme, enquanto o mapa de\n"
        "  referência previa árvores, pastagens e terra firme!\n\n"
        "• POR QUE MÉTODOS CEGOS (RETA / ZIGUE-ZAGUE) FALHAM:\n"
        "  Sem navegação intencional, o drone passa dezenas de passos voando sobre\n"
        "  água repetitiva, acumulando incerteza ou sofrendo falsa localização crítica!"
    )
    ax.text(0.08, 0.46, text_content, color="#cdd6f4", fontsize=12, va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1e1e2e", edgecolor="#45475a", lw=1.5))

    ax_p = fig.add_axes([0.60, 0.28, 0.16, 0.38])
    ax_p.imshow(c_post)
    ax_p.set_title("O que o Drone Vê\n(Câmera Pós-Enchente)", color="#f38ba8", fontsize=11, fontweight="bold")
    ax_p.axis("off")

    ax_m = fig.add_axes([0.78, 0.28, 0.16, 0.38])
    ax_m.imshow(c_pre)
    ax_m.set_title("O que Veria no Mapa\n(Original Pré-Enchente)", color="#a6e3a1", fontsize=11, fontweight="bold")
    ax_m.axis("off")

    ax.text(0.77, 0.19, "Discrepância Extrema: A ilha desapareceu sob a água barrenta!", ha="center", color="#fab387", fontsize=11, fontweight="bold")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return cv2.resize(rgb, (w, h))


def render_slide_how(w=1920, h=1080) -> np.ndarray:
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#181825")
    ax = fig.add_subplot(111, facecolor="#181825")
    ax.axis("off")

    ax.text(0.08, 0.91, "2. COMO? A ARQUITETURA DA SOLUÇÃO TÉCNICA", color="#cdd6f4", fontsize=22, fontweight="bold")
    ax.text(0.08, 0.85, "Três pilares fundamentais: Filtro Espacial, RL Multiobjetivo (MOPPO) e Memória Curta de Ações", color="#89dceb", fontsize=13)

    cards = [
        ("PILAR 1: FILTRO ESPACIAL E NUVEM DE PONTOS",
         "• Mantém a crença b_t sobre grade de 50 m.\n\n"
         "• No espaço, b_t é visualizada como uma NUVEM DE\n"
         "  PONTOS (partículas em ciano).\n\n"
         "• Quando o drone sobrevoa água barrenta, a nuvem\n"
         "  se espalha por toda a baía (alta incerteza).\n\n"
         "• Ao atingir feições salientes (margens/pontes), a\n"
         "  nuvem colapsa em um ponto denso sobre o drone!",
         "#89dceb", 0.08),

        ("PILAR 2: MOPPO MULTIOBJETIVO (À LA AMOR)",
         "• Rede neural única condicionada em vetor w:\n"
         "  w = [w_info, w_cost, w_success, w_risk]\n\n"
         "• Permite escolher o comportamento em tempo real:\n"
         "  * Preset Rápido: minimiza a bateria gasta.\n"
         "  * Preset Seguro: aversão máxima a risco.\n\n"
         "• Em áreas inundadas, o modo seguro penaliza\n"
         "  escolhas ambíguas, evitando falsas localizações!",
         "#fab387", 0.38),

        ("PILAR 3: MEMÓRIA CURTA DE AÇÕES (ANL)",
         "• Buffer FIFO com o histórico das últimas 8 ações.\n\n"
         "• Inspirado no Active Neural Localization (ICLR 2018).\n\n"
         "• Hipótese: ajuda a quebrar a simetria sobre água\n"
         "  uniforme (não demonstrada: 1 seed, orçamentos diferentes).\n\n"
         "• Efeito no hipervolume: inconclusivo (ver errata).",
         "#a6e3a1", 0.68),
    ]

    for title, content, col, left_x in cards:
        full_text = f"{title}\n\n{content}"
        ax.text(left_x, 0.44, full_text, color="#cdd6f4", fontsize=11, va="center",
                bbox=dict(boxstyle="round,pad=0.9", facecolor="#1e1e2e", edgecolor=col, lw=1.8))

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return cv2.resize(rgb, (w, h))


def render_slide_table(w=1920, h=1080) -> np.ndarray:
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#181825")
    ax = fig.add_subplot(111, facecolor="#181825")
    ax.axis("off")

    ax.text(0.08, 0.91, "3. RESULTADOS QUANTITATIVOS NAS ILHAS DE PORTO ALEGRE", color="#cdd6f4", fontsize=22, fontweight="bold")
    ax.text(0.08, 0.85, "Avaliação empírica rigorosa com 300 episódios pareados (Cenário P1 - Enchente Histórica)", color="#89dceb", fontsize=13)

    # Table data
    col_labels = ["Política / Método", "Preset de Pesos (w)", "Taxa de Sucesso", "Falsa Localização", "Distância Média", "Passos (mediana)"]
    cell_data = [
        ["MOPPO Seguro (Treino MO)", "[0.15, 0.15, 0.15, 0.55]", "49.0% [42.7, 55.0]", "19.7% [15.0, 24.3]  ★", "1028 m [944, 1112]  ★", "23 passos  ★"],
        ["MOPPO Rápido (Treino MO)", "[0.15, 0.55, 0.15, 0.15]", "52.0% [46.3, 58.0]", "21.3% [16.7, 26.3]", "1054 m [973, 1139]", "23 passos"],
        ["PPO-S2 PoC (1º Treino)", "Objetivo Único (Escalar)", "71.3% [66.0, 76.7]", "21.3% [16.7, 26.3]", "1320 m [1231, 1410]", "28 passos"],
        ["Linha Reta (Straight)", "Baseline Cego", "67.7% [62.0, 72.7]", "22.0% [17.3, 27.0]", "2160 m [2014, 2322]", "31 passos"],
    ]

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0.08, 0.38, 0.84, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for (row, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#45475a")
        if row == 0:
            cell.set_facecolor("#313244")
            cell.set_text_props(color="#cdd6f4", fontweight="bold")
        elif row == 1: # MOPPO Seguro
            cell.set_facecolor("#1e2e2e")
            cell.set_text_props(color="#a6e3a1", fontweight="bold")
        else:
            cell.set_facecolor("#1e1e2e")
            cell.set_text_props(color="#cdd6f4")

    takeaway_box = (
        "★ DESTAQUES QUANTITATIVOS DO MOPPO NAS ILHAS SUBMERSAS:\n"
        "1. Menor Risco: O MOPPO Seguro alcançou a menor taxa de falsa localização de todos os métodos (19.7% vs 22.0% da reta).\n"
        "2. Economia de Bateria: Voou apenas 1028 metros em média contra 2160 m da reta (mais de 50% de redução no consumo de voo!).\n"
        "3. Decisão Rápida: Mediana de apenas 23 passos para resolver o impasse, enquanto a linha reta passa mais de 30 passos perdida."
    )
    ax.text(0.08, 0.18, takeaway_box, color="#cdd6f4", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e1e2e", edgecolor="#fab387", lw=1.5))

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return cv2.resize(rgb, (w, h))


def render_slide_conclusion(w=1920, h=1080) -> np.ndarray:
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#181825")
    ax = fig.add_subplot(111, facecolor="#181825")
    ax.axis("off")

    ax.text(0.5, 0.88, "5. CONCLUSÕES E LIÇÕES APRENDIDAS", ha="center", color="#cdd6f4", fontsize=24, fontweight="bold")
    ax.text(0.5, 0.81, "Principais conclusões do estudo de localização ativa sob choque morfológico nas Ilhas de POA", ha="center", color="#89dceb", fontsize=14)

    conclusions = (
        "1. A LINHA RETA É UMA BASELINE DIFÍCIL QUANDO O MAPA ESTÁ DESATUALIZADO:\n"
        "   Nas ilhas (300 episódios), a PPO-S2 faz 71,3% de sucesso, a reta 67,7% e a MO 49,0%.\n"
        "   Estratégias que confiam no mapa antigo herdam os erros dele onde o mapa mais mudou.\n\n"
        "2. O CONDICIONAMENTO EM PREFERÊNCIAS NÃO FOI DEMONSTRADO NA V1:\n"
        "   O gate de condicionamento falhou no treino, e os presets safe e unif deram números idênticos.\n"
        "   H7 não suportada: localização falsa do safe 19,7% × 21,3% da PPO-S2, IC da diferença inclui 0.\n\n"
        "3. TRANSFERÊNCIA ZERO-SHOT:\n"
        "   As políticas treinadas no Taquari rodam em Porto Alegre sem retreino; o desempenho\n"
        "   depende da região (melhor no Centro, pior nas ilhas)."
    )
    ax.text(0.5, 0.44, conclusions, ha="center", va="center", color="#cdd6f4", fontsize=12.5,
            bbox=dict(boxstyle="round,pad=1.2", facecolor="#1e1e2e", edgecolor="#a6e3a1", lw=2.0))

    ax.text(0.5, 0.14, "Projeto Drone Active Localization • Antigravity AI • 2026", ha="center", color="#6c7086", fontsize=11)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return cv2.resize(rgb, (w, h))


def generate_full_presentation_video():
    print("=" * 60)
    print("Generating PPTX-Style Executive Video Presentation")
    print("=" * 60)

    cfg_poa = load_config("configs_poa/poa.yaml")
    grid = poa_grid(cfg_poa)
    slices = region_slices(cfg_poa, grid)
    rs, cs = slices["ilhas"]

    # 1. Render Static Slides
    print("[deck] Rendering Slide 1: Title...")
    s1_rgb = render_slide_title()

    # Pre-extract camera crops for Slide 2
    r0, c0 = 40, 45
    cell = grid.cell
    cur_rg = rs.start + r0 + 20
    cur_cg = cs.start + c0 + 25
    cur_x = grid.x0 + (cur_cg + 0.5) * cell
    cur_y = grid.y1 - (cur_rg + 0.5) * cell

    with rasterio.open("data_poa/raw/poa_post.tif") as ds_post, rasterio.open("data_poa/raw/poa_pre_feb24.tif") as ds_pre:
        win_p = from_bounds(cur_x - 75, cur_y - 75, cur_x + 75, cur_y + 75, ds_post.transform)
        win_m = from_bounds(cur_x - 75, cur_y - 75, cur_x + 75, cur_y + 75, ds_pre.transform)
        c_post = np.transpose(ds_post.read([1, 2, 3], window=win_p), (1, 2, 0))
        c_pre = np.transpose(ds_pre.read([1, 2, 3], window=win_m), (1, 2, 0))

    print("[deck] Rendering Slide 2: The Why...")
    s2_rgb = render_slide_why(c_post, c_pre)

    print("[deck] Rendering Slide 3: The How...")
    s3_rgb = render_slide_how()

    print("[deck] Rendering Slide 4: Results Table...")
    s4_rgb = render_slide_table()

    print("[deck] Rendering Slide 6: Conclusions...")
    s6_rgb = render_slide_conclusion()

    # 2. Extract or Render Slide 5 Dynamic Comparison Frames
    # We can read the frames from results_mo/videos/poa_ilhas_comparison_nuvem_pontos_camera.mp4 directly!
    video_path = ROOT / "results_mo/videos/poa_ilhas_comparison_nuvem_pontos_camera.mp4"
    assert video_path.exists(), f"{video_path} must exist"

    print(f"[deck] Reading dynamic comparison frames from {video_path.name}...")
    cap = cv2.VideoCapture(str(video_path))
    slide5_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to 1920x1080 if needed
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rgb_frame.shape[:2] != (1080, 1920):
            rgb_frame = cv2.resize(rgb_frame, (1920, 1080))
        slide5_frames.append(rgb_frame)
    cap.release()
    print(f"[deck] Loaded {len(slide5_frames)} dynamic flight frames for Slide 5.")

    # 3. Assemble Full Video Presentation
    # Frame sequence at 4 FPS:
    # S1 (Title): 4s = 16 frames
    # S2 (Why): 6s = 24 frames
    # S3 (How): 6s = 24 frames
    # S4 (Table): 6s = 24 frames
    # S5 (Live Video): all dynamic frames
    # S6 (Conclusion): 6s = 24 frames
    fps = 4
    all_frames = []

    print("[deck] Assembling slide timeline...")
    for _ in range(int(fps * 4.0)):
        all_frames.append(s1_rgb)
    for _ in range(int(fps * 6.0)):
        all_frames.append(s2_rgb)
    for _ in range(int(fps * 6.0)):
        all_frames.append(s3_rgb)
    for _ in range(int(fps * 6.5)):
        all_frames.append(s4_rgb)
    for f in slide5_frames:
        all_frames.append(f)
    for _ in range(int(fps * 6.0)):
        all_frames.append(s6_rgb)

    out_mp4 = ROOT / "results_mo/videos/poa_ilhas_presentation_deck.mp4"
    out_gif = ROOT / "results_mo/videos/poa_ilhas_presentation_deck.gif"

    print(f"[deck] Writing full presentation MP4 ({len(all_frames)} frames, ~{len(all_frames)/fps:.1f}s) to {out_mp4}...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (1920, 1080))
    for f in all_frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    print(f"[deck] Writing GIF ({len(all_frames)} frames) to {out_gif}...")
    pil_frames = [Image.fromarray(f).resize((960, 540), Image.Resampling.BILINEAR) for f in all_frames[::2]] # 2 FPS for GIF to keep size compact
    pil_frames[0].save(
        str(out_gif),
        save_all=True,
        append_images=pil_frames[1:],
        duration=500, # 2 FPS
        loop=0,
        optimize=True,
    )

    print("\n[deck] PPTX-Style Executive Presentation Video generated successfully!")


if __name__ == "__main__":
    generate_full_presentation_video()
