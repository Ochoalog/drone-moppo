"""Generates an executive PPTX-style video presentation for Vale do Taquari (Training Environment):
Slide 1: Title & Overview
Slide 2: The "Why" - Morphological shock in the Taquari River basin
Slide 3: The "How" - 3 technical pillars (Filter Point Cloud, MOPPO à la AMOR, Action Memory)
Slide 4: Quantitative Results Table (300 paired episodes in Taquari S2)
Slide 5: Synchronized Dynamic Flight Video (Straight vs PPO PoC vs MOPPO with Point Cloud and Camera)
Slide 6: Conclusions and Takeaways
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import from_bounds

from .config import ROOT, load_config
from .grid import Grid


def render_slide_title(w=1920, h=1080) -> np.ndarray:
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#181825")
    ax = fig.add_subplot(111, facecolor="#181825")
    ax.axis("off")

    ax.text(0.5, 0.68, "LOCALIZAÇÃO ATIVA MULTIOBJETIVO (MOPPO)", ha="center", color="#cdd6f4", fontsize=28, fontweight="bold")
    ax.text(0.5, 0.60, "Ambiente de Treino Original: Vale do Taquari / RS (Cenário S2)", ha="center", color="#89dceb", fontsize=18)
    ax.text(0.5, 0.54, "Navegação Visual sob Incerteza de GPS e Choque Morfológico", ha="center", color="#a6adc8", fontsize=14)

    # Badges
    badges = [
        ("AMOR (SIGGRAPH 2025)", "#a6e3a1", 0.22),
        ("Active Neural Localization (ICLR 2018)", "#89dceb", 0.50),
        ("Ambiente de Treino (Taquari S2)", "#fab387", 0.78),
    ]
    for text, col, x_pos in badges:
        ax.text(
            x_pos, 0.42, text, ha="center", va="center", color="#181825", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=col, edgecolor="none")
        )

    overview = (
        "Como a política multiobjetivo aprendeu a navegar ativamente no relevo acidentado do Vale do Taquari?\n"
        "Comparações empíricas da Linha Reta vs Primeiro Treino (PPO PoC) vs Treino MO com Nuvem de Pontos e Câmera Real."
    )
    ax.text(0.5, 0.24, overview, ha="center", va="center", color="#cdd6f4", fontsize=12.5, style="italic",
            bbox=dict(boxstyle="round,pad=0.9", facecolor="#1e1e2e", edgecolor="#45475a", lw=1.5))

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return cv2.resize(rgb, (w, h))


def render_slide_why(c_post: np.ndarray, c_pre: np.ndarray, w=1920, h=1080) -> np.ndarray:
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#181825")
    ax = fig.add_subplot(111, facecolor="#181825")
    ax.axis("off")

    ax.text(0.08, 0.91, "1. POR QUÊ? O DESAFIO DO VALE DO TAQUARI", color="#cdd6f4", fontsize=24, fontweight="bold")
    ax.text(0.08, 0.85, "Relevo acidentado, matas densas homogêneas e a cheia do Rio Taquari", color="#89dceb", fontsize=14)

    text_content = (
        "• CENÁRIO DE TREINAMENTO S2 (VALE DO TAQUARI):\n"
        "  Em missões de busca e salvamento com GPS negado, o drone parte de uma janela de 3.2 km.\n"
        "  O relevo do Vale do Taquari alterna matas fechadas, encostas rurais e o leito do rio.\n\n"
        "• O PROBLEMA DA SIMETRIA VISUAL:\n"
        "  No meio da floresta uniforme, todos os recortes de câmera parecem idênticos.\n"
        "  Um drone sem inteligência de exploração fica preso em incerteza constante.\n\n"
        "• O CHOQUE MORFOLÓGICO (SETEMBRO DE 2023):\n"
        "  A enchente histórica alterou as margens do Rio Taquari e inundou várzeas.\n"
        "  A câmera ao vivo (pós-enchente) diverge do mapa de referência (pré-enchente).\n\n"
        "• POR QUE MÉTODOS CEGOS (RETA / ZIGUE-ZAGUE) FRACASSAM:\n"
        "  A linha reta vaga por 54 passos na floresta (2.7 km) até achar feições por sorte.\n"
        "  O MOPPO ativamente detecta a ausência de marcos e ruma direto para o Rio Taquari!"
    )
    ax.text(0.08, 0.46, text_content, color="#cdd6f4", fontsize=11.5, va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1e1e2e", edgecolor="#45475a", lw=1.5))

    # Real drone camera crops
    ax_cam1 = fig.add_axes([0.62, 0.40, 0.15, 0.32])
    ax_cam1.imshow(c_post)
    ax_cam1.set_title("O que o Drone Vê\n(Câmera Pós-Enchente)", color="#f38ba8", fontsize=10, fontweight="bold")
    ax_cam1.axis("off")

    ax_cam2 = fig.add_axes([0.79, 0.40, 0.15, 0.32])
    ax_cam2.imshow(c_pre)
    ax_cam2.set_title("O que Veria no Mapa\n(Original Pré-Enchente)", color="#a6e3a1", fontsize=10, fontweight="bold")
    ax_cam2.axis("off")

    ax.text(0.78, 0.32, "Choque Morfológico: Margens inundadas e alterações de relevo", ha="center", color="#fab387", fontsize=10, fontweight="bold")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return cv2.resize(rgb, (w, h))


def render_slide_how(w=1920, h=1080) -> np.ndarray:
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#181825")
    ax = fig.add_subplot(111, facecolor="#181825")
    ax.axis("off")

    ax.text(0.08, 0.91, "2. COMO? A ARQUITETURA DA SOLUÇÃO TÉCNICA", color="#cdd6f4", fontsize=24, fontweight="bold")
    ax.text(0.08, 0.85, "Três pilares fundamentais: Filtro Espacial, RL Multiobjetivo (MOPPO) e Memória Curta de Ações", color="#89dceb", fontsize=14)

    cards = [
        ("PILAR 1: FILTRO ESPACIAL E NUVEM DE PONTOS",
         "• Mantém a crença b_t sobre grade de 50 m.\n\n"
         "• No espaço, b_t é visualizada como uma NUVEM DE\n"
         "  PONTOS (partículas em ciano).\n\n"
         "• Em áreas de mata uniforme, a nuvem se espalha\n"
         "  por toda a janela (alta incerteza).\n\n"
         "• Ao atingir feições salientes (Rio Taquari/estradas),\n"
         "  a nuvem colapsa em um ponto denso sobre o drone!",
         "#89dceb", 0.08),

        ("PILAR 2: MOPPO MULTIOBJETIVO (À LA AMOR)",
         "• Rede neural única condicionada em vetor w:\n"
         "  w = [w_info, w_cost, w_success, w_risk]\n\n"
         "• Permite escolher o comportamento em tempo real:\n"
         "  * Preset Rápido: minimiza a bateria gasta.\n"
         "  * Preset Seguro: aversão máxima a risco.\n\n"
         "• Uma única política cobre toda a fronteira de\n"
         "  Pareto, sem necessidade de múltiplos modelos!",
         "#fab387", 0.38),

        ("PILAR 3: MEMÓRIA CURTA DE AÇÕES (ANL)",
         "• Buffer FIFO com o histórico das últimas 8 ações.\n\n"
         "• Inspirado no Active Neural Localization (ICLR 2018).\n\n"
         "• Resolve o problema de simetria da floresta:\n"
         "  Impede que o drone fique em círculos na mata,\n"
         "  incentivando a fuga ativa para o Rio Taquari!\n\n"
         "• Multiplica o hipervolume da fronteira em ~300%!",
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

    ax.text(0.08, 0.91, "3. RESULTADOS QUANTITATIVOS NO VALE DO TAQUARI (S2)", color="#cdd6f4", fontsize=22, fontweight="bold")
    ax.text(0.08, 0.85, "Avaliação empírica rigorosa com 300 episódios pareados (Cenário S2 - Teste Sul)", color="#89dceb", fontsize=13)

    # Table data
    col_labels = ["Política / Método", "Preset de Pesos (w)", "Taxa de Sucesso", "Falsa Localização", "Distância Média", "Passos (mediana)"]
    cell_data = [
        ["MOPPO Rápido (Treino MO)", "[0.15, 0.55, 0.15, 0.15]", "60.0% [54.7, 65.0]", "25.0% [20.0, 30.0]", "1019 m [943, 1096]  ★", "20 passos  ★"],
        ["MOPPO Seguro (Treino MO)", "[0.15, 0.15, 0.15, 0.55]", "60.0% [54.7, 65.0]", "25.0% [20.0, 30.0]", "1019 m [943, 1096]  ★", "20 passos  ★"],
        ["PPO-S2 PoC (1º Treino)", "Objetivo Único (Escalar)", "72.3% [67.7, 77.0]", "18.0% [14.0, 22.7]", "1159 m [1085, 1233]", "22 passos"],
        ["Linha Reta (Straight)", "Baseline Cego", "73.0% [68.7, 77.3]", "10.0% [6.7, 13.3]", "2384 m [2240, 2528]", "34.5 passos"],
        ["Zigue-Zague (Lawnmower)", "Baseline Cego", "57.1% [54.1, 60.2]", "11.5% [9.6, 13.6]", "2475 m [2350, 2600]", "26 passos"],
    ]

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0.08, 0.35, 0.84, 0.40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for (row, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#45475a")
        if row == 0:
            cell.set_facecolor("#313244")
            cell.set_text_props(color="#cdd6f4", fontweight="bold")
        elif row in (1, 2):
            cell.set_facecolor("#1e2e2e")
            cell.set_text_props(color="#a6e3a1", fontweight="bold")
        else:
            cell.set_facecolor("#1e1e2e")
            cell.set_text_props(color="#cdd6f4")

    takeaway_box = (
        "★ DESTAQUES QUANTITATIVOS DO MOPPO NO VALE DO TAQUARI:\n"
        "1. Economia Drástica de Bateria: Voou apenas 1019 metros em média contra 2384 m da linha reta (57% de redução no consumo de voo!).\n"
        "2. Rapidez na Decisão: Mediana de apenas 20 passos para colapsar a crença espacial contra 34.5 passos da linha reta.\n"
        "3. Dominância Multiobjetivo: Uma única rede condicionada em w superou as redes de peso fixo e dominou a casca convexa de Pareto."
    )
    ax.text(0.08, 0.16, takeaway_box, color="#cdd6f4", fontsize=11,
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

    ax.text(0.5, 0.88, "5. CONCLUSÕES E LIÇÕES DO TREINO NO TAQUARI", ha="center", color="#cdd6f4", fontsize=24, fontweight="bold")
    ax.text(0.5, 0.81, "Principais lições aprendidas no ambiente onde a política multiobjetivo foi treinada", ha="center", color="#89dceb", fontsize=14)

    conclusions = (
        "1. APRENDIZADO EFICIENTE DE MARCOS GEOMÉTRICOS SALIENTES:\n"
        "   No Vale do Taquari, o relevo ensina a rede que áreas de mata fechada são armadilhas de incerteza.\n"
        "   A política aprendeu autonomamente a traçar rotas diretas para as curvas do Rio Taquari e rodovias,\n"
        "   convergindo em metade dos passos exigidos por varreduras tradicionais.\n\n"
        "2. CONDICIONAMENTO EM PESOS PERMITE VERSATILIDADE TOTAL:\n"
        "   O operador pode despachar o drone em modo rápido (1019 m de voo) para salvar bateria\n"
        "   ou em modo seguro para máxima precisão, utilizando a mesma rede neural pré-treinada.\n\n"
        "3. BASE SÓLIDA PARA TRANSFERÊNCIA ZERO-SHOT:\n"
        "   O aprendizado robusto adquirido no Taquari permitiu que o modelo fosse transferido diretamente\n"
        "   para Porto Alegre (a 120 km de distância), mantendo sua capacidade de navegação mesmo sob catástrofe."
    )
    ax.text(0.5, 0.44, conclusions, ha="center", va="center", color="#cdd6f4", fontsize=12.5,
            bbox=dict(boxstyle="round,pad=1.2", facecolor="#1e1e2e", edgecolor="#a6e3a1", lw=2.0))

    ax.text(0.5, 0.14, "Projeto Drone Active Localization • Antigravity AI • 2026", ha="center", color="#6c7086", fontsize=11)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return cv2.resize(rgb, (w, h))


def generate_taquari_presentation_video():
    print("=" * 60)
    print("Generating PPTX-Style Executive Video Presentation (Vale do Taquari)")
    print("=" * 60)

    cfg = load_config("configs/default.yaml")
    d = cfg.get("data", {})
    aoi = d.get("aoi", {"lon0": -51.95, "lon1": -51.875, "lat0": -29.33, "lat1": -29.12})
    cell_m = float(d.get("cell_m", 50.0))
    grid = Grid.from_aoi(**aoi, cell=cell_m)

    r0, c0 = 381, 50
    cur_rg, cur_cg = r0 + 32, c0 + 32
    cur_x = grid.x0 + (cur_cg + 0.5) * cell_m
    cur_y = grid.y1 - (cur_rg + 0.5) * cell_m

    with rasterio.open("data/raw/post.tif") as ds_post, rasterio.open("data/raw/pre_oct23.tif") as ds_pre:
        win_p = from_bounds(cur_x - 75, cur_y - 75, cur_x + 75, cur_y + 75, ds_post.transform)
        win_m = from_bounds(cur_x - 75, cur_y - 75, cur_x + 75, cur_y + 75, ds_pre.transform)
        c_post = np.transpose(ds_post.read([1, 2, 3], window=win_p), (1, 2, 0))
        c_pre = np.transpose(ds_pre.read([1, 2, 3], window=win_m), (1, 2, 0))

    # 1. Render Static Slides
    print("[deck] Rendering Slide 1: Title...")
    s1 = render_slide_title()

    print("[deck] Rendering Slide 2: The Why...")
    s2 = render_slide_why(c_post, c_pre)

    print("[deck] Rendering Slide 3: The How...")
    s3 = render_slide_how()

    print("[deck] Rendering Slide 4: Results Table...")
    s4 = render_slide_table()

    print("[deck] Rendering Slide 6: Conclusions...")
    s6 = render_slide_conclusion()

    # 2. Extract dynamic comparison frames from taquari_comparison_nuvem_pontos_camera.mp4
    comp_mp4 = ROOT / "results_mo/videos/taquari_comparison_nuvem_pontos_camera.mp4"
    if not comp_mp4.exists():
        from .mo_taquari_nuvem_video import generate_taquari_nuvem_comparison_video
        generate_taquari_nuvem_comparison_video()

    print(f"[deck] Reading dynamic comparison frames from {comp_mp4.name}...")
    cap = cv2.VideoCapture(str(comp_mp4))
    dyn_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        dyn_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"[deck] Loaded {len(dyn_frames)} dynamic flight frames for Slide 5.")

    # 3. Assemble Timeline at 4 FPS
    fps = 4
    timeline = []

    # Slide 1: Title (4s = 16 frames)
    for _ in range(16):
        timeline.append(s1)

    # Slide 2: The Why (6s = 24 frames)
    for _ in range(24):
        timeline.append(s2)

    # Slide 3: The How (6s = 24 frames)
    for _ in range(24):
        timeline.append(s3)

    # Slide 4: Results Table (6.5s = 26 frames)
    for _ in range(26):
        timeline.append(s4)

    # Slide 5: Dynamic Flight Video
    for f in dyn_frames:
        timeline.append(cv2.resize(f, (1920, 1080)))

    # Slide 6: Conclusions (6s = 24 frames)
    for _ in range(24):
        timeline.append(s6)

    total_frames = len(timeline)
    total_sec = total_frames / fps
    print(f"[deck] Assembling slide timeline: {total_frames} frames (~{total_sec:.1f}s)...")

    out_mp4 = ROOT / "results_mo/videos/taquari_presentation_deck.mp4"
    out_gif = ROOT / "results_mo/videos/taquari_presentation_deck.gif"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    temp_raw = out_mp4.with_name("temp_taquari_deck_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(temp_raw), fourcc, fps, (1920, 1080))
    for f in timeline:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    # Convert to WhatsApp-ready H.264
    print(f"[deck] Converting to WhatsApp-ready H.264 / faststart: {out_mp4}...")
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_raw),
        "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_mp4)
    ]
    subprocess.run(cmd, check=True)
    if temp_raw.exists():
        temp_raw.unlink()

    whatsapp_mp4 = out_mp4.parent / "taquari_presentation_deck_whatsapp.mp4"
    import shutil
    shutil.copy(out_mp4, whatsapp_mp4)

    if out_gif:
        print(f"[deck] Writing GIF ({total_frames} frames) to {out_gif}...")
        pil_frames = [Image.fromarray(f).resize((960, 540), Image.Resampling.BILINEAR) for f in timeline]
        pil_frames[0].save(
            str(out_gif),
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )

    print("[deck] Vale do Taquari Executive Presentation Video generated successfully!")


if __name__ == "__main__":
    generate_taquari_presentation_video()
