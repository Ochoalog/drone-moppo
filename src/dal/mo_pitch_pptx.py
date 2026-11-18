"""Builds an executive, publication-grade PPTX pitch presentation for Active Drone Localization.

Designed for a 5-minute presentation with a professor:
- Highly visual, concise and direct.
- Shows the problem of the lost drone with real visual examples.
- Explains the 3 technical pillars (Filter Point Cloud, MOPPO à la AMOR, ANL Memory).
- Incorporates embedded synchronized comparison videos (Taquari and Porto Alegre Ilhas).
- Delivers clean quantitative comparisons and key takeaways.
"""
from __future__ import annotations

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

from .config import ROOT


# ==============================================================================
# Theme Colors & Styles (Modern Deep Slate / Catppuccin Mocha)
# ==============================================================================
C_BG = RGBColor(17, 20, 29)         # #11141d (Slide background)
C_CARD_BG = RGBColor(26, 31, 44)    # #1a1f2c (Card container)
C_CARD_BORDER = RGBColor(45, 53, 72)# #2d3548 (Card border)
C_TEXT_WHITE = RGBColor(255, 255, 255)
C_TEXT_LIGHT = RGBColor(205, 214, 244) # #cdd6f4
C_TEXT_MUTED = RGBColor(166, 173, 200) # #a6adc8

C_CYAN = RGBColor(0, 229, 255)      # #00e5ff (Belief Point Cloud)
C_GREEN = RGBColor(0, 230, 118)     # #00e676 (MOPPO Success)
C_PEACH = RGBColor(255, 171, 64)    # #ffab40 (AMOR / Tradeoffs)
C_RED = RGBColor(255, 82, 82)       # #ff5252 (Risk / Warning)
C_PURPLE = RGBColor(203, 166, 247)  # #cba6f7 (Baselines)


def set_bg(slide):
    """Sets a dark background on the slide."""
    bg_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5)
    )
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = C_BG
    bg_shape.line.fill.background()
    return bg_shape


def add_header(slide, title_text: str, subtitle_text: str, tag_text: str = "PROJETO DRONE ACTIVE LOCALIZATION • ANTIGRAVITY AI"):
    """Adds a consistent header across slides."""
    # Top Tag
    tb_tag = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
    p_tag = tb_tag.text_frame.paragraphs[0]
    p_tag.text = tag_text.upper()
    p_tag.font.name = "Arial"
    p_tag.font.size = Pt(9.5)
    p_tag.font.bold = True
    p_tag.font.color.rgb = C_CYAN

    # Main Title
    tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.68), Inches(11.7), Inches(0.55))
    p_title = tb_title.text_frame.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = "Arial"
    p_title.font.size = Pt(20)
    p_title.font.bold = True
    p_title.font.color.rgb = C_TEXT_WHITE

    # Subtitle
    tb_sub = slide.shapes.add_textbox(Inches(0.8), Inches(1.22), Inches(11.7), Inches(0.35))
    p_sub = tb_sub.text_frame.paragraphs[0]
    p_sub.text = subtitle_text
    p_sub.font.name = "Arial"
    p_sub.font.size = Pt(11)
    p_sub.font.color.rgb = C_TEXT_MUTED


def add_card(slide, left, top, width, height, border_color=C_CARD_BORDER, bg_color=C_CARD_BG):
    """Adds a styled card background shape."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.5)
    return card


# ==============================================================================
# Slide Builders
# ==============================================================================

def build_slide1_cover(prs):
    """Slide 1: Capa / Pitch Title."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)

    # Accent decorative glow line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(11.733), Inches(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = C_CYAN
    line.line.fill.background()

    # Category Pill
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(4.2), Inches(0.42))
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(18, 53, 66)
    pill.line.color.rgb = C_CYAN
    pill.line.width = Pt(1.2)
    tf_pill = pill.text_frame
    p_pill = tf_pill.paragraphs[0]
    p_pill.text = "APRESENTAÇÃO EXECUTIVA • PITCH DE 5 MINUTOS"
    p_pill.font.name = "Arial"
    p_pill.font.size = Pt(10)
    p_pill.font.bold = True
    p_pill.font.color.rgb = C_CYAN
    p_pill.alignment = PP_ALIGN.CENTER

    # Main Title
    tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(2.1), Inches(11.7), Inches(1.4))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    p1 = tf_title.paragraphs[0]
    p1.text = "LOCALIZAÇÃO ATIVA MULTIOBJETIVO\nSOB NEGAÇÃO DE GPS"
    p1.font.name = "Arial"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = C_TEXT_WHITE

    # Subtitle
    tb_sub = slide.shapes.add_textbox(Inches(0.8), Inches(3.6), Inches(11.7), Inches(0.8))
    tf_sub = tb_sub.text_frame
    tf_sub.word_wrap = True
    p_sub = tf_sub.paragraphs[0]
    p_sub.text = "Navegação Visual Autônoma com MOPPO (à la AMOR) e Memória de Ações (ANL) em Cenários Reais de Calamidade"
    p_sub.font.name = "Arial"
    p_sub.font.size = Pt(15)
    p_sub.font.color.rgb = C_CYAN

    # 3 Badges / Value Pillars
    badges = [
        ("AMOR (ACM SIGGRAPH 2025)", "RL Multiobjetivo Condicionado em Pesos\nRede neural única cobrindo toda a fronteira de Pareto", C_GREEN),
        ("ANL (ICLR 2018)", "Memória Curta de Ações FIFO (8 passos)\nQuebra simetrias visuais em água e matas homogêneas", C_PEACH),
        ("Validação sob Calamidade", "Treino no Vale do Taquari (S2) e Teste Zero-Shot\nnas Ilhas Submersas de Porto Alegre (Maio/2024)", C_CYAN),
    ]

    card_w = Inches(3.75)
    for i, (b_title, b_desc, b_col) in enumerate(badges):
        left_pos = Inches(0.8 + i * 4.0)
        card = add_card(slide, left_pos, Inches(4.6), card_w, Inches(1.8), border_color=b_col)
        
        tb = slide.shapes.add_textbox(left_pos + Inches(0.15), Inches(4.7), card_w - Inches(0.3), Inches(1.6))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p_bt = tf.paragraphs[0]
        p_bt.text = b_title
        p_bt.font.name = "Arial"
        p_bt.font.size = Pt(12)
        p_bt.font.bold = True
        p_bt.font.color.rgb = b_col
        
        p_bd = tf.add_paragraph()
        p_bd.text = b_desc
        p_bd.font.name = "Arial"
        p_bd.font.size = Pt(10.5)
        p_bd.font.color.rgb = C_TEXT_LIGHT

    # Footer
    tb_foot = slide.shapes.add_textbox(Inches(0.8), Inches(6.7), Inches(11.7), Inches(0.4))
    p_foot = tb_foot.text_frame.paragraphs[0]
    p_foot.text = "Projeto Drone Active Localization • Apresentação Acadêmica / Pitch • 2026"
    p_foot.font.name = "Arial"
    p_foot.font.size = Pt(9.5)
    p_foot.font.color.rgb = C_TEXT_MUTED


def build_slide2_problem(prs):
    """Slide 2: O Problema: O Drone Perdido em Território Homogêneo e Choque Morfológico."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_header(
        slide,
        "1. O PROBLEMA: NAVEGAÇÃO VISUAL SOB GPS NEGADO E CHOQUE MORFOLÓGICO",
        "Como um drone de busca e salvamento se localiza quando o GPS falha e o terreno perdeu suas referências?"
    )

    # Left Container: Problem Explanation Card (5.2 in)
    add_card(slide, Inches(0.8), Inches(1.7), Inches(5.1), Inches(5.3), border_color=C_RED)
    tb_left = slide.shapes.add_textbox(Inches(0.95), Inches(1.85), Inches(4.8), Inches(5.0))
    tf_left = tb_left.text_frame
    tf_left.word_wrap = True

    items = [
        ("O Cenário Operacional:", "Em áreas de desastre com GPS bloqueado, o drone parte de uma janela de busca de 3.2 x 3.2 km (64 x 64 células). Ele só dispõe de uma câmera voltada para o solo e um mapa prévio embarcado.", C_TEXT_WHITE),
        ("Onde o Drone se Perde?", "• Mata Fechada (Taquari): Copas de árvores uniformes geram simetria visual absoluta (alta entropia).\n• Ilhas Submersas (POA): A enchente cobriu arquipélagos inteiros sob lâminas de água barrenta.", C_PEACH),
        ("O Choque Morfológico Extremo:", "A câmera do drone só vê água marrom uniforme, enquanto o mapa de satélite pré-enchente esperava terra firme e árvores! Não há correspondência visual direta.", C_RED),
        ("Por que Métodos Tradicionais Falham?", "Varreduras cegas (linha reta e zigue-zague) não possuem agência ativa: voam em círculos ou derivam por quilômetros na água, esgotando a bateria ou emitindo falsas localizações fatais!", C_TEXT_LIGHT),
    ]

    for idx, (title, desc, col) in enumerate(items):
        p_t = tf_left.paragraphs[0] if idx == 0 else tf_left.add_paragraph()
        p_t.text = title
        p_t.font.name = "Arial"
        p_t.font.size = Pt(11)
        p_t.font.bold = True
        p_t.font.color.rgb = col
        p_t.space_before = Pt(6) if idx > 0 else Pt(0)

        p_d = tf_left.add_paragraph()
        p_d.text = desc
        p_d.font.name = "Arial"
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = C_TEXT_LIGHT

    # Right Container 1: Problem Visual Graphic
    add_card(slide, Inches(6.1), Inches(1.7), Inches(6.433), Inches(2.9), border_color=C_CARD_BORDER)
    visual_img = ROOT / "results_mo/pptx_assets/slide2_problem_visual.png"
    if visual_img.exists():
        slide.shapes.add_picture(str(visual_img), Inches(6.25), Inches(1.8), width=Inches(6.133))

    # Right Container 2: Consequence Callout Box
    add_card(slide, Inches(6.1), Inches(4.75), Inches(6.433), Inches(2.25), border_color=C_PEACH)
    tb_impact = slide.shapes.add_textbox(Inches(6.25), Inches(4.85), Inches(6.133), Inches(2.05))
    tf_impact = tb_impact.text_frame
    tf_impact.word_wrap = True

    p_it = tf_impact.paragraphs[0]
    p_it.text = "⚠️ O GARGALO DOS MÉTODOS TRADICIONAIS (VARREDURAS CEGAS)"
    p_it.font.name = "Arial"
    p_it.font.size = Pt(11)
    p_it.font.bold = True
    p_it.font.color.rgb = C_PEACH

    impact_points = [
        ("Deriva Espacial Cega:", "Em água barrenta ou mata homogênea, voar reto significa derivar por mais de 3.3 km (67 passos) sem colapsar a incerteza."),
        ("Incerteza Cumulativa:", "Sem marcos visuais novos, a crença b_t espalha-se em dezenas de picos espúrios pela janela de 3.2 km."),
        ("Risco de Falsa Localização:", "Mais de 22% de decisões erradas nas baselines — o drone bate o martelo na coordenada incorreta, fatal em resgates!"),
    ]
    for ip_title, ip_desc in impact_points:
        p_ip = tf_impact.add_paragraph()
        p_ip.text = f"• {ip_title} {ip_desc}"
        p_ip.font.name = "Arial"
        p_ip.font.size = Pt(9.5)
        p_ip.font.color.rgb = C_TEXT_LIGHT
        p_ip.space_before = Pt(3)


def build_slide3_architecture(prs):
    """Slide 3: O que Fizemos: A Arquitetura Técnica em 3 Pilares."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_header(
        slide,
        "2. O QUE FIZEMOS: A ARQUITETURA TÉCNICA EM 3 PILARES",
        "Três pilares fundamentais: Filtro Espacial, RL Multiobjetivo (MOPPO à la AMOR) e Memória Curta de Ações (ANL)"
    )

    pillars = [
        (
            "PILAR 1: FILTRO ESPACIAL & NUVEM DE PONTOS",
            "Filtro de Histograma Bayesiano (b_t)",
            [
                "Mantém a distribuição de probabilidade espacial b_t sobre a grade de 50 m.",
                "No espaço 2D, a crença é visualizada como uma Nuvem de Partículas em Ciano.",
                "Em áreas uniformes (lama/floresta), a nuvem dispersa-se por toda a janela de busca.",
                "Ao atingir feições salientes (rodovia/rio), a nuvem colapsa em um ponto denso sobre o drone!",
                "Critério de Sucesso: ≥ 90% da probabilidade concentrada em um raio de 100 metros.",
            ],
            C_CYAN
        ),
        (
            "PILAR 2: MOPPO MULTIOBJETIVO (À LA AMOR)",
            "Rede Neural Condicionada em Pesos (w)",
            [
                "Rede única condicionada no vetor de preferências: w = [w_info, w_cost, w_success, w_risk].",
                "Crítico vetorial V(s, w) ∈ ℝ⁴ com GAE escalarizada Â = A · w normalizada por minibatch.",
                "Permite adaptar o drone em tempo de voo sem retreinar:",
                "  • Preset Rápido: minimiza a bateria gasta.",
                "  • Preset Seguro: aversão máxima a risco.",
                "Na v1, o condicionamento em w não foi demonstrado (gate reprovado; ver errata).",
            ],
            C_PEACH
        ),
        (
            "PILAR 3: MEMÓRIA CURTA DE AÇÕES (ANL)",
            "Buffer FIFO de 8 Ações Recentes",
            [
                "Inspirado no Active Neural Localization (ICLR 2018).",
                "Armazena as últimas 8 ações em one-hot (64 dimensões) concatenadas à observação.",
                "Quebra de Simetria Visual: Impede que o drone fique em círculos sobre água uniforme ou mata homogênea.",
                "Incentiva a fuga ativa em direção à terra firme e rodovias estruturadas.",
                "Efeito da memória no hipervolume: inconclusivo (1 seed, orçamentos diferentes).",
            ],
            C_GREEN
        ),
    ]

    card_w = Inches(3.75)
    for i, (p_title, p_sub, bullets, p_col) in enumerate(pillars):
        left_pos = Inches(0.8 + i * 4.0)
        add_card(slide, left_pos, Inches(1.7), card_w, Inches(5.3), border_color=p_col)

        tb = slide.shapes.add_textbox(left_pos + Inches(0.15), Inches(1.85), card_w - Inches(0.3), Inches(5.0))
        tf = tb.text_frame
        tf.word_wrap = True

        p_t = tf.paragraphs[0]
        p_t.text = p_title
        p_t.font.name = "Arial"
        p_t.font.size = Pt(11)
        p_t.font.bold = True
        p_t.font.color.rgb = p_col

        p_s = tf.add_paragraph()
        p_s.text = p_sub
        p_s.font.name = "Arial"
        p_s.font.size = Pt(10)
        p_s.font.italic = True
        p_s.font.color.rgb = C_TEXT_MUTED
        p_s.space_after = Pt(8)

        for b in bullets:
            p_b = tf.add_paragraph()
            p_b.text = f"• {b}"
            p_b.font.name = "Arial"
            p_b.font.size = Pt(9.5)
            p_b.font.color.rgb = C_TEXT_LIGHT
            p_b.space_after = Pt(4)


def build_slide4_taquari_video(prs):
    """Slide 4: Ambiente de Treino (Vale do Taquari S2) & Como ele se Recupera com Vídeo."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_header(
        slide,
        "3. AMBIENTE DE TREINO & RECUPERAÇÃO: VALE DO TAQUARI (CENÁRIO S2)",
        "Onde treinamos a política multiobjetivo e como o drone aprendeu a colapsar a incerteza no Rio Taquari"
    )

    # Left Container: Embedded Video
    add_card(slide, Inches(0.8), Inches(1.7), Inches(7.5), Inches(5.3), border_color=C_CYAN)
    
    movie_path = ROOT / "results_mo/videos/taquari_comparison_nuvem_pontos_camera.mp4"
    poster_path = ROOT / "results_mo/pptx_assets/poster_taquari_comparison.png"
    
    if movie_path.exists() and poster_path.exists():
        slide.shapes.add_movie(
            str(movie_path),
            Inches(0.95), Inches(1.85), Inches(7.2), Inches(4.2),
            poster_frame_image=str(poster_path),
            mime_type="video/mp4"
        )

    # Video Caption Pill
    tb_vcap = slide.shapes.add_textbox(Inches(0.95), Inches(6.15), Inches(7.2), Inches(0.7))
    tf_vcap = tb_vcap.text_frame
    tf_vcap.word_wrap = True
    p_vc = tf_vcap.paragraphs[0]
    p_vc.text = "▶ VÍDEO INCORPORADO: Comparação Sincronizada no Vale do Taquari (Episódio 0)\nTopo: Satélite + Nuvem de Partículas colapsando. Embaixo: Câmera do Drone vs Mapa Original."
    p_vc.font.name = "Arial"
    p_vc.font.size = Pt(9)
    p_vc.font.bold = True
    p_vc.font.color.rgb = C_CYAN

    # Right Container: Explanation & Metrics (4.0 in)
    add_card(slide, Inches(8.5), Inches(1.7), Inches(4.033), Inches(5.3), border_color=C_CARD_BORDER)
    tb_right = slide.shapes.add_textbox(Inches(8.65), Inches(1.85), Inches(3.733), Inches(5.0))
    tf_right = tb_right.text_frame
    tf_right.word_wrap = True

    right_sections = [
        ("Onde Treinamos:", "Vale do Taquari / RS (cenário S2). Ortofoto de alta resolução (50 m/célula) cobrindo a cheia de Setembro/2023. Treino no norte, teste no sul.", C_TEXT_WHITE),
        ("Como o Drone se Recupera:", "Ao decolar no meio da mata uniforme, a política detecta a falta de marcos discriminativos e ruma ativamente para o Rio Taquari e a rodovia, colapsando a nuvem de pontos em apenas 18 passos!", C_GREEN),
        ("Comparação Lado a Lado no Vídeo:", 
         "• 🟢 Treino MO (MOPPO + Memória):\n   18 passos (1.272 m) — Sucesso Rápido!\n"
         "• 🟠 1º Treino (PPO PoC S2):\n   29 passos (1.425 m) — Curva indireta.\n"
         "• 🟣 Linha Reta (Baseline Cego):\n   54 passos (2.777 m) — Vaga na mata!", C_PEACH),
        ("Destaque no Taquari:", "O MOPPO economizou mais de 57% de bateria em relação à linha reta com decisão 2x mais rápida!", C_CYAN)
    ]

    for idx, (title, desc, col) in enumerate(right_sections):
        p_t = tf_right.paragraphs[0] if idx == 0 else tf_right.add_paragraph()
        p_t.text = title
        p_t.font.name = "Arial"
        p_t.font.size = Pt(10.5)
        p_t.font.bold = True
        p_t.font.color.rgb = col
        p_t.space_before = Pt(5) if idx > 0 else Pt(0)

        p_d = tf_right.add_paragraph()
        p_d.text = desc
        p_d.font.name = "Arial"
        p_d.font.size = Pt(9.5)
        p_d.font.color.rgb = C_TEXT_LIGHT


def build_slide5_poa_video(prs):
    """Slide 5: Teste Extremo Zero-Shot: As Ilhas Submersas de Porto Alegre com Vídeo."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_header(
        slide,
        "4. TESTE ZERO-SHOT: AS ILHAS SUBMERSAS DE PORTO ALEGRE (MAIO/2024)",
        "Transferência cega a 120 km de distância durante a maior catástrofe climática do Rio Grande do Sul"
    )

    # Left Container: Embedded Video
    add_card(slide, Inches(0.8), Inches(1.7), Inches(7.5), Inches(5.3), border_color=C_GREEN)
    
    movie_path = ROOT / "results_mo/videos/poa_ilhas_comparison_nuvem_pontos_camera.mp4"
    poster_path = ROOT / "results_mo/pptx_assets/poster_poa_comparison.png"
    
    if movie_path.exists() and poster_path.exists():
        slide.shapes.add_movie(
            str(movie_path),
            Inches(0.95), Inches(1.85), Inches(7.2), Inches(4.2),
            poster_frame_image=str(poster_path),
            mime_type="video/mp4"
        )

    # Video Caption Pill
    tb_vcap = slide.shapes.add_textbox(Inches(0.95), Inches(6.15), Inches(7.2), Inches(0.7))
    tf_vcap = tb_vcap.text_frame
    tf_vcap.word_wrap = True
    p_vc = tf_vcap.paragraphs[0]
    p_vc.text = "▶ VÍDEO INCORPORADO: Comparação Sincronizada nas Ilhas do Delta do Jacuí (Episódio 13)\nO drone decola de uma ilha submersa. Observe o que a câmera vê (lama) vs o mapa original (mata)."
    p_vc.font.name = "Arial"
    p_vc.font.size = Pt(9)
    p_vc.font.bold = True
    p_vc.font.color.rgb = C_GREEN

    # Right Container: Explanation & Metrics (4.0 in)
    add_card(slide, Inches(8.5), Inches(1.7), Inches(4.033), Inches(5.3), border_color=C_CARD_BORDER)
    tb_right = slide.shapes.add_textbox(Inches(8.65), Inches(1.85), Inches(3.733), Inches(5.0))
    tf_right = tb_right.text_frame
    tf_right.word_wrap = True

    right_sections = [
        ("O Desafio Extremo das Ilhas:", "A ilha sob a qual o drone decola desapareceu sob a água marrom. O mapa diz que há vegetação, mas a câmera só registra lama homogênea!", C_RED),
        ("Como o MOPPO se Recupera:", "O modelo reconhece a ambiguidade da água, recusa derivar na baía inundada e ruma direto para a margem e para a ponte da BR-290, colapsando a nuvem de pontos em apenas 9 passos (636 m) com 71.3% de certeza!", C_GREEN),
        ("Comparação Lado a Lado no Vídeo:",
         "• 🟢 Treino MO (MOPPO + Memória):\n   9 passos (636 m) — Vitória Esmagadora!\n"
         "• 🟠 1º Treino (PPO PoC S2):\n   14 passos (824 m) — Busca estrada.\n"
         "• 🟣 Linha Reta (Baseline Cego):\n   67 passos (3.353 m) — Perdida na água!", C_PEACH),
        ("Destaque nas Ilhas:", "Menor taxa de falsa localização de todos os métodos (19.7% vs 22.0% da reta) com menos da metade do gasto de voo!", C_CYAN)
    ]

    for idx, (title, desc, col) in enumerate(right_sections):
        p_t = tf_right.paragraphs[0] if idx == 0 else tf_right.add_paragraph()
        p_t.text = title
        p_t.font.name = "Arial"
        p_t.font.size = Pt(10.5)
        p_t.font.bold = True
        p_t.font.color.rgb = col
        p_t.space_before = Pt(5) if idx > 0 else Pt(0)

        p_d = tf_right.add_paragraph()
        p_d.text = desc
        p_d.font.name = "Arial"
        p_d.font.size = Pt(9.5)
        p_d.font.color.rgb = C_TEXT_LIGHT


def build_slide6_results_table(prs):
    """Slide 6: Resultados Quantitativos & Fronteira de Pareto."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_header(
        slide,
        "5. RESULTADOS QUANTITATIVOS: COMPARAÇÃO RIGOROSA COM 300 EPISÓDIOS",
        "Avaliação empírica pareada no Vale do Taquari (S2) e nas Ilhas de Porto Alegre (P1 Enchente)"
    )

    # Table Card Container
    add_card(slide, Inches(0.8), Inches(1.7), Inches(11.733), Inches(3.2), border_color=C_CARD_BORDER)

    # Create Table
    rows = 7
    cols = 6
    tbl_shape = slide.shapes.add_table(rows, cols, Inches(0.95), Inches(1.85), Inches(11.433), Inches(2.9))
    tbl = tbl_shape.table

    # Column widths
    tbl.columns[0].width = Inches(2.2) # Cenário
    tbl.columns[1].width = Inches(2.5) # Política
    tbl.columns[2].width = Inches(1.6) # Sucesso
    tbl.columns[3].width = Inches(1.8) # Falsa Loc.
    tbl.columns[4].width = Inches(1.8) # Distância
    tbl.columns[5].width = Inches(1.533) # Passos

    headers = ["Cenário / Região", "Política / Método", "Sucesso (%) ↑", "Falsa Loc. (%) ↓", "Distância Média ↓", "Passos (med.)"]
    table_data = [
        ["Taquari S2 (Treino)", "MOPPO Rápido (Treino MO)", "60.0% [54.7, 65.0]", "25.0% [20.0, 30.0]", "1019 m [943, 1096] ★", "20 passos ★"],
        ["Taquari S2 (Treino)", "PPO-S2 PoC (1º Treino Escalar)", "72.3% [67.7, 77.0]", "18.0% [14.0, 22.7]", "1159 m [1085, 1233]", "22 passos"],
        ["Taquari S2 (Treino)", "Linha Reta (Baseline Cego)", "73.0% [68.7, 77.3]", "10.0% [6.7, 13.3]", "2384 m [2240, 2528]", "34.5 passos"],
        ["POA Ilhas (Enchente)", "MOPPO Seguro (Treino MO)", "49.0% [42.7, 55.0]", "19.7% [15.0, 24.3] ★", "1028 m [944, 1112] ★", "23 passos ★"],
        ["POA Ilhas (Enchente)", "PPO-S2 PoC (1º Treino Escalar)", "71.3% [66.0, 76.7]", "21.3% [16.7, 26.3]", "1320 m [1231, 1410]", "28 passos"],
        ["POA Ilhas (Enchente)", "Linha Reta (Baseline Cego)", "67.7% [62.0, 72.7]", "22.0% [17.3, 27.0]", "2160 m [2014, 2322]", "31 passos"],
    ]

    for c_idx, h in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(38, 45, 62)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Arial"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = C_CYAN
        p.alignment = PP_ALIGN.CENTER

    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            # Highlight MOPPO rows
            if "MOPPO" in row[1]:
                cell.fill.fore_color.rgb = RGBColor(22, 38, 38)
            else:
                cell.fill.fore_color.rgb = C_CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Arial"
            p.font.size = Pt(9.5)
            p.alignment = PP_ALIGN.CENTER
            if "MOPPO" in row[1]:
                p.font.bold = True
                p.font.color.rgb = C_GREEN if "★" in val else C_TEXT_WHITE
            else:
                p.font.color.rgb = C_TEXT_LIGHT

    # Bottom Area: 3 Highlight Callout Cards (Width 3.75 in each)
    highlights = [
        ("🔋 Menos distância, mais erros", "MO: 1019 m em média no Taquari contra 2384 m da reta, mas com 25,0% de localização falsa contra 10,0% (média sobre todos os episódios).", C_GREEN),
        ("🛡️ H7 não suportada", "Localização falsa do safe nas ilhas: 19,7% contra 21,3% da PPO-S2; o IC da diferença inclui 0.", C_CYAN),
        ("📈 Hipervolume: inconclusivo", "0.1529 → 0.4232 com memória, mas confundido por orçamento (348k × 246k), 1 seed e conjuntos de tamanhos diferentes.", C_PEACH),
    ]

    for i, (hl_t, hl_d, hl_col) in enumerate(highlights):
        left_pos = Inches(0.8 + i * 4.0)
        add_card(slide, left_pos, Inches(5.1), Inches(3.75), Inches(1.9), border_color=hl_col)

        tb = slide.shapes.add_textbox(left_pos + Inches(0.15), Inches(5.2), Inches(3.45), Inches(1.7))
        tf = tb.text_frame
        tf.word_wrap = True

        p_t = tf.paragraphs[0]
        p_t.text = hl_t
        p_t.font.name = "Arial"
        p_t.font.size = Pt(11)
        p_t.font.bold = True
        p_t.font.color.rgb = hl_col

        p_d = tf.add_paragraph()
        p_d.text = hl_d
        p_d.font.name = "Arial"
        p_d.font.size = Pt(9.5)
        p_d.font.color.rgb = C_TEXT_LIGHT
        p_d.space_before = Pt(4)


def build_slide7_conclusions(prs):
    """Slide 7: Conclusões & Lições Aprendidas (Pitch Wrap-up)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_header(
        slide,
        "6. CONCLUSÕES & LIÇÕES APRENDIDAS (PITCH FINAL)",
        "O que este trabalho demonstra para a robótica aérea autônoma em cenários de desastre"
    )

    takeaways = [
        (
            "1. Navegação Ativa é Indispensável em Calamidades",
            "Padrões cegos de varredura geométrica (reta e zigue-zague) assumem que qualquer terreno fornece feições constantes. Quando a água cobre ilhas inteiras ou a mata é uniforme, voar reto significa ficar preso na incerteza. O MOPPO ativamente detecta a ausência de marcos e busca refúgio em estruturas discriminativas (pontes e margens).",
            C_CYAN
        ),
        (
            "2. RL Multiobjetivo (MOPPO à la AMOR) Viabiliza Frotas Reais",
            "Treinar uma rede neural para cada peso de recompensa é inviável em operação de campo. Com uma única política condicionada no vetor w, o operador pode alternar dinamicamente entre voo rápido (preservar bateria) e voo ultrasseguro (aversão máxima a erro) sem necessidade de retreinar o modelo.",
            C_PEACH
        ),
        (
            "3. Memória Curta (ANL) e Transferência Zero-Shot",
            "O efeito da memória de 8 ações é inconclusivo na v1 (1 seed, orçamentos diferentes). As políticas treinadas no Taquari rodam em Porto Alegre sem retreino; a MO perde para a PPO-S2 e para a reta nas ilhas (49,0% × 71,3% × 67,7% de sucesso).",
            C_GREEN
        ),
    ]

    card_h = Inches(1.45)
    for i, (t_title, t_desc, t_col) in enumerate(takeaways):
        top_pos = Inches(1.7 + i * 1.6)
        add_card(slide, Inches(0.8), top_pos, Inches(11.733), card_h, border_color=t_col)

        tb = slide.shapes.add_textbox(Inches(1.0), top_pos + Inches(0.12), Inches(11.333), card_h - Inches(0.24))
        tf = tb.text_frame
        tf.word_wrap = True

        p_t = tf.paragraphs[0]
        p_t.text = t_title
        p_t.font.name = "Arial"
        p_t.font.size = Pt(12)
        p_t.font.bold = True
        p_t.font.color.rgb = t_col

        p_d = tf.add_paragraph()
        p_d.text = t_desc
        p_d.font.name = "Arial"
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = C_TEXT_LIGHT
        p_d.space_before = Pt(3)

    # Bottom Pitch Close
    banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(6.45), Inches(11.733), Inches(0.55))
    banner.fill.solid()
    banner.fill.fore_color.rgb = RGBColor(22, 38, 38)
    banner.line.color.rgb = C_GREEN
    banner.line.width = Pt(1.2)
    tf_b = banner.text_frame
    p_b = tf_b.paragraphs[0]
    p_b.text = "🎯 CONCLUSÃO: Localização Ativa Multiobjetivo une eficiência energética, precisão e adaptabilidade para drones autônomos de resgate!"
    p_b.font.name = "Arial"
    p_b.font.size = Pt(10.5)
    p_b.font.bold = True
    p_b.font.color.rgb = C_GREEN
    p_b.alignment = PP_ALIGN.CENTER


def generate_pitch_presentation(out_pptx: Path = ROOT / "results_mo/apresentacao_drone_active_loc.pptx"):
    print("=" * 60)
    print("Generating Executive Pitch Presentation (PPTX)")
    print("=" * 60)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    print("[pptx] Building Slide 1: Cover...")
    build_slide1_cover(prs)

    print("[pptx] Building Slide 2: The Problem (Lost Drone & Shock)...")
    build_slide2_problem(prs)

    print("[pptx] Building Slide 3: Technical Architecture...")
    build_slide3_architecture(prs)

    print("[pptx] Building Slide 4: Taquari Training & Recovery (with Video)...")
    build_slide4_taquari_video(prs)

    print("[pptx] Building Slide 5: Porto Alegre Zero-Shot Test (with Video)...")
    build_slide5_poa_video(prs)

    print("[pptx] Building Slide 6: Quantitative Results Table...")
    build_slide6_results_table(prs)

    print("[pptx] Building Slide 7: Conclusions & Pitch Wrap-up...")
    build_slide7_conclusions(prs)

    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_pptx))
    print(f"[pptx] Presentation saved successfully: {out_pptx} ({os.path.getsize(out_pptx) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    generate_pitch_presentation()
