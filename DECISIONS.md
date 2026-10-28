# DECISIONS

Decisões tomadas durante a implementação (decisão, motivo, alternativa descartada).

## Ambiente
- **CPU apenas.** `nvidia-smi` falha (sem driver); CPU i7-10510U (4 núcleos / 8 threads), 15 GB RAM, ~4 GB livres em disco. Torch CPU (`download.pytorch.org/whl/cpu`). *Alternativa:* nenhuma.
- **`uv` + `.venv`, Python 3.12**; versões fixadas em `requirements.txt` (freeze) e `pyproject.toml`.

## Fase 1: dados
- **AOI em UTM:** a maior caixa alinhada à grade de 50 m **dentro** do retângulo lon/lat (x 407800–414850, y 6755150–6778300 → 141 × 463 células). *Motivo:* o retângulo lon/lat não é alinhado em UTM; a caixa interna garante que tudo está dentro da AOI pedida. *Alternativa:* caixa envolvente (incluiria pontos fora da AOI).
- **Leitura no overview fator 4 (1,2207 m/px) e miniatura no fator 16 (4,88 m/px)**, em blocos de 2048 px lidos em paralelo (8 threads, um handle por thread), recortados aos limites do dataset e completados com zeros. As três datas têm a mesma grade de pixels (origens diferem por múltiplos de 5000 m), então não há reamostragem. Download total ≈ 35 s.
- **Armazenamento como GeoTIFF JPEG q95** (~50–60 MB por data) em vez de `.npy` (355 MB por data). *Motivo:* sobram só ~4 GB de disco; a fonte já é JPEG, então a perda extra é desprezível.
- **Pixel válido = max(R,G,B) > 3** (e não ≠ 0). *Motivo:* o ringing do JPEG deixa valores 1–3 no nodata.
- **Frações por célula calculadas sobre a footprint de 150 m** (e não sobre o quadrado de 50 m). *Motivo:* o descritor vê a footprint inteira; nuvem nas vizinhanças corrompe o descritor.
- **Máscara de nuvem refinada:** HSV (V > 200, S < 40) → **abertura com disco de 15 px (~18 m)** → dilatação com disco de 41 px (~50 m). *Motivo:* o detector bruto marcava os telhados brancos da área urbana (manchas idênticas em `post` e `pre_mar23`), o que invalidaria justamente as células mais distintivas. A abertura remove objetos pequenos e mantém as nuvens (centenas de metros). A dilatação grande cobre bordas suaves. Resultado: nuvem bruta (como em §3) post 1,0%, mar23 0,85%, oct23 15,6%. Refinada: post 0,16%, mar23 0,12%, oct23 17,2%. *Alternativa descartada:* detector bruto com dilatação 7 px (descartava ~4% das células, quase todas urbanas).
- **Teste de §3:** a fração de nuvem é comparada (±30%) usando o detector **bruto**, que é o que reproduz a medição manual. A máscara refinada tem um teste próprio.

## Alinhamento
- Correlação de fase (OpenCV, janela de Hanning) em magnitude de Sobel, nos **60 tiles de 600 m mais estruturados** (maior gradiente médio em `post`) válidos nas duas datas. Usa a mediana dos tiles com resposta ≥ 0,05. Convenção verificada com deslocamento sintético (+3,3 px → +3,16 px estimado).
- **Resultados:** `pre_mar23` (2,22 E, 1,22 N) m, |d| = 2,5 m (MAD em x = 2,2 m); `pre_oct23` (4,92 E, 0,85 N) m, |d| = 5,0 m. Os dois passam de 2 m, então **a correção é aplicada** (as crops do mapa são amostradas em p + offset).
- `pre_mar23` deu um pouco acima dos 1,5–2 m de §3, com dispersão espacial grande em x (relevo e ortorretificação diferentes). Isso é ≪ 50 m (célula) e foi mantido com o offset global. *Alternativa:* campo de deslocamento local (desnecessário nessa escala).
