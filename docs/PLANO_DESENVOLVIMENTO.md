# Plano de Desenvolvimento v4: Submissão ICUAS 2027

**Projeto:** localização global ativa de drone sem GNSS sobre mapas desatualizados por enchentes (*When Does Learning to Act Pay Off? Belief-Space RL for UAV Localization over Outdated Post-Flood Maps*)
**Conferência alvo:** **ICUAS 2027** — International Conference on Unmanned Aircraft Systems (14–17 jun 2027, Fortaleza, CE, Brasil; https://uasconferences.com/2027_icuas/)
**Deadline de submissão:** **05/02/2027** (revisão *double-blind*, IEEE 2 colunas, limite de **8 páginas**)
**Versão:** v4, de 27/11/2026. Substitui a v3 (preservada em `docs/PLANO_DESENVOLVIMENTO_v3.md`).

---

## 0. Objetivo da v4 (por que a v3 ainda não basta para o ICUAS)

O artigo v3 (entregue na disciplina em 25/11/2026) já possui rigor estatístico de conferência internacional (episódios pareados, bootstrap hierárquico seeds → janelas, *learning gates*, hipóteses pré-especificadas e resultados negativos honestos). Porém, em uma conferência de robótica aérea como o ICUAS, há **duas lacunas principais** que revisores certamente apontarão:

1. **Todas as observações atuais são *satélite → satélite* (Maxar pré vs. Maxar pós).** Falta demonstrar o pipeline (percepção DINOv2 + filtro bayesiano + políticas ativas) sobre **imagens reais capturadas por câmera de UAV** (*cross-view domain gap*: lente, iluminação, escala, rotação/yaw e textura aérea real).
2. **A ambiguidade controlada (RQ4 / Experimento E) usa ruído gaussiano sintético ($\sigma$).** Falta ancorar esse espectro de ambiguidade na queda real de matching *drone → satélite*.
3. **Requisitos formais de UAV/ICUAS:** medição de latência em tempo real por passo de decisão, anonimização *double-blind*, fechamento de referências (`VERIFICAR` em `paper/refs.bib`) e ajuste rigoroso ao limite de **8 páginas**.

A v4 fecha essas lacunas com **dois planos de dados reais de UAV** (Planos 1 e 2) + fechamento editorial (Plano 3), sem quebrar nenhum artefato congelado da v1/v3.

---

## 1. Princípios mantidos da v3

1. **Nada congelado muda:** `results/`, `results_poa/`, `results_v3/` e `paper/tables/v1_*.csv` permanecem intactos. Todo novo artefato da v4 vai para `data_v4/`, `results_v4/` e `scripts/v4_*` (ou `src/dal/v4/`).
2. **Sem vazamento:** as políticas PPO e os pesos fine-tuned do DINOv2 (`dinov2_vits14_ftA`, treinados apenas na faixa de treino do Taquari) são avaliados estritamente **zero-shot** nos novos dados de UAV.
3. **Critérios antes da avaliação:** hipóteses e critérios dos Planos 1 e 2 são registrados antes da avaliação final nos dados de teste.
4. **Separação editorial limpa com o paper de λ-BRIEF/SC-BRIEF:** os voos de `/storage/UAVs Localization/Datasets/` (`UFRGS`, `ADM1–3`) são usados aqui **exclusivamente** para validar o *cross-view domain gap* do DINOv2 e o histogram filter da arquitetura ANL (ancorando a RQ4), sem repetir resultados de localização global com descritores binários/multiespectrais.

---

## 2. Plano 1 — Validação *Cross-View* Real e Filtro Passivo nos Voos Reais (`UFRGS` e `ADM1–3`)

**Dados já disponíveis no disco** (`/storage/UAVs Localization/Datasets/`):
- **UFRGS** (`UFRGS VET/voo_100m/`): 826 m, 224 frames avaliados (534 totais), altitude 82–101 m, DJI Matrice 100 + Parrot Sequoia RGB ($910\times 1234$), 58 mapas de satélite (2002–2018, ~0,48 km²).
- **ADM1 (`0035`), ADM2 (`0036`), ADM3 (`0040`)** em **Arroio do Meio, RS (Vale do Taquari!)**: 716–1610 m, altitudes 32–191 m, 27 mapas de satélite (2006–2018, ~0,9 km²).
- Ground truth completo: `traj_truth.txt` (x, y, escala, yaw) e `traj_angles.txt` (roll, pitch, yaw).

### Entregas do Plano 1
1. **Medição do *Domain Gap* Real *Drone → Satélite* (`P1-retrieval`):**
   - Extrair descritores **DINOv2 zero-shot (GeM / grid4)**, **DINOv2 fine-tuned (`ftA`)** e **hand-crafted** nos pares reais *(frame RGB de drone, recorte de satélite correspondente)* em mapas recentes (2018) e desatualizados (2006/2011/2002).
   - Medir *top-1 retrieval* e distribuição de similaridade cosseno real, mapeando em qual nível de ruído sintético **L0–L3** (RQ4) um voo real de drone se situa.
2. **Validação do Filtro Bayesiano em Trajetórias Reais (`P1-filter`):**
   - Rodar o histogram filter do `dal_v3` ao longo das trajetórias reais (sem GPS, usando apenas odometria relativa + ruído) em mapas recentes vs. antigos, reportando taxa de convergência, erro final (m) e falsas localizações.
3. **Gate P1-G0 (integridade e alinhamento):**
   - Conferir escala métrica (`px/m`), orientação (`yaw`) e alinhamento visual de pelo menos 5 frames por voo antes de extrair descritores em massa.

---

## 3. Plano 2 — Site de Teste Ativo com Ortomosaicos Reais de UAV Pós-Enchente (Canoas / Mathias Velho)

**Descoberta chave no OpenAerialMap (27/11/2026):**
O Corpo de Bombeiros do Estado de São Paulo (**CBPMESP**) publicou no OpenAerialMap ortomosaicos GeoTIFF reais capturados por UAV durante a resposta às enchentes de maio/2024 em **Mathias Velho, Canoas, RS** — **100% dentro da `CANOAS_AOI_UTM` já usada na v3**, permitindo usar exatamente o mesmo mapa de satélite pré-enchente Maxar (`poa_pre_feb24`, 02/02/2024)!

| Dataset OpenAerialMap (UAV) | Data | GSD | Área aprox. | UTM 22S (`x`, `y`) | Tamanho |
|---|---|---|---|---|---|
| `20240526_mathiasvelho_setorleste_rgb` (`6653e06349dd860001fae223`) | 2024-05-26 | 4,1 cm/px | ~2,60 km² | `x=[480501, 482541]`, `y=[6691344, 6692626]` | 390,5 MB |
| `20240525 - Canoas - Mathias Velho - RS - Setor Leste` (`66531d5249dd860001fae1fd`) | 2024-05-25 | 15,1 cm/px | ~0,77 km² | `x=[479484, 480602]`, `y=[6691962, 6692652]` | 9,5 MB |
| `20240525 - Canoas - Mathias Velho - RS - Setor Oeste` (`6653094e49dd860001fae1f9`) | 2024-05-25 | 15,1 cm/px | ~1,98 km² | `x=[476689, 477798]`, `y=[6691093, 6692884]` | 22,4 MB |

*(Bônus opcional: ortomosaicos UAV CBPMESP de Cruzeiro do Sul/RS — Vale do Taquari — 22–24/05/2024, ~1,0 km², 2–3 cm/px.)*

### Por que este experimento eleva o artigo ao padrão ICUAS
- Permite comparar lado a lado, **na mesma cidade e na mesma enchente**:
  1. **Canoas Satélite → Satélite** (Maxar 09/05 vs. Maxar 02/02 — já na v3);
  2. **Canoas UAV Real → Satélite** (Ortomosaico UAV CBPMESP 25–26/05 vs. Maxar 02/02 — novo na v4).
- Como a área contínua do UAV é de ~2,0 × 1,2 km (Setor Leste) e ~1,1 × 1,8 km (Setor Oeste), adaptamos o tamanho da janela de busca no teste UAV (ex.: $1{,}0 \times 1{,}0\text{ km}$ ou $1{,}2 \times 1{,}2\text{ km}$ em grade de 25 m, $40\times 40$ ou $48\times 48$ células — ou janelas de $1{,}6 \times 1{,}6\text{ km}$ unindo Leste 25/05 + 26/05), preservando o mesmo passo de 50 m, o mesmo filtro bayesiano e avaliação pareada de **PPO (zero-shot)** vs. **Straight**, **Straight 150 m**, **Lawnmower**, **Greedy (map)** e **Random**.

### Entregas e Gates do Plano 2
1. **Gate P2-G0 (qualidade e registro do ortomosaico UAV):**
   - Baixar os GeoTIFFs CBPMESP em `data_v4/uav_canoas/`.
   - Reprojetar/verificar CRS para `EPSG:32722` (UTM 22S), calcular fração de pixels válidos (sem borda preta/nodata) e verificar alinhamento espacial (deslocamento `dx, dy`) contra o Maxar pré-evento `poa_pre_feb24`.
2. **Extração de descritores *UAV → Satélite* (`P2-extract`):**
   - Gerar specs de grade 25 m (altitudes 60, 100 e 150 m; 4 variantes de observação com jitter/yaw/roll/pitch/fotometria) sobre os ortomosaicos UAV reais.
   - Extrair descritores DINOv2 (`dinov2_vits14_ftA` + zero-shot) no Colab (GPU T4) ou CPU local (área pequena: ~3.000–5.000 células válidas, muito rápido!).
3. **Avaliação Ativa Pareada em Observações Reais de UAV (`P2-eval`):**
   - Rodar episódios pareados comparando `Random`, `Greedy (map)`, `Lawnmower`, `Straight (100 m)`, `Straight (150 m)` e `PPO` (10 seeds zero-shot de Taquari) com bootstrap hierárquico.
   - *(Opcional, se houver folga de GPU)*: testar se o ranking de métodos em **UAV Real → Satélite** reproduz o ranking de **Satélite → Satélite** (Canoas) e **Ruído Sintético (L3)**.

---

## 4. Plano 3 — Requisitos Formais e Polimento ICUAS 2027

1. **Benchmark de latência em tempo real (`P3-latency`):**
   - Medir tempo médio e p95 por passo de decisão (pré-processamento + DINOv2 ViT-S/14 + atualização do histogram filter + forward PPO) em CPU e GPU, comparando com o orçamento físico de 6 s por passo de 50 m.
2. **Revisão de referências (`paper/refs.bib`):**
   - Resolver todos os 8 `note={VERIFICAR}` com citações completas e conferidas (venue, ano, páginas/DOI/URL).
3. **Anonimização *double-blind*:**
   - Remover `UFRGS` e qualquer identificador de autor em `paper/main.tex` e links externos.
4. **Ajuste de páginas (limite estrito de 8 páginas IEEEtran):**
   - Compilar PDF final e ajustar texto/tabelas/apêndice para caber em **8 páginas totais** sem overfull boxes.
5. **Bônus opcional (Janeiro/2027, se sobrar tempo):**
   - 1 tarde de voo real em campo com o drone do amigo (modo GCS passo-a-passo, 60 m / 100 m AGL dentro do limite ANAC de 120 m) para incluir 1 figura/vídeo de demonstração física em campo.

---

## 5. Cronograma até o Deadline (05/02/2027)

| Período | Marco | Entregáveis |
|---|---|---|
| **27/11 – 06/12/2026** | Gates iniciais P1-G0 e P2-G0 | Download e verificação de alinhamento dos ortomosaicos UAV CBPMESP (Canoas); inspeção e piloto de retrieval nos voos `UFRGS`/`ADM1`; benchmark de latência em CPU. |
| **07/12 – 20/12/2026** | Execução completa Plano 1 | Retrieval *cross-view* + ancoragem L0–L3 + filtro passivo nos 4 voos reais (`UFRGS`, `ADM1–3`) com DINOv2 (`ftA` e zero-shot). |
| **21/12/2026 – 10/01/2027** | Execução completa Plano 2 | Extração de descritores no ortomosaico UAV Canoas (Mathias Velho) + avaliação ativa pareada (PPO zero-shot vs. 5 heurísticas) + figuras/tabelas automatizadas. |
| **11/01 – 25/01/2027** | Integração no artigo (v5 ICUAS) | Nova subseção de validação com UAV real (Planos 1 e 2) + parágrafo de latência + fechamento de `refs.bib` + anonimização *double-blind*. |
| **26/01 – 04/02/2027** | Revisão final e PDF eXpress | Enxugamento para exatamente 8 páginas, revisão de inglês, validação IEEE PDF eXpress e submissão antes de **05/02/2027**. |
