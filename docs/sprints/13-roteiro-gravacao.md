# Roteiro de gravação — Sprint 13

Objetivo: ≥ 3 sessões × 4 rótulos (`arm_raised`, `arms_crossed`, `arms_open`,
`idle`). Cada clique em gravar = uma sessão nova. ~15 s por sessão.

Total: 12 gravações, ~5 minutos de câmera.

| Rodada | Distância da câmera | Luz | Roupa/ângulo |
|---|---|---|---|
| 1 | perto (tronco inteiro no quadro, ~1 m) | luz normal do ambiente | de frente |
| 2 | longe (~2 m, corpo até o quadril) | luz mais fraca / só tela | levemente de lado (~20°) |
| 3 | média (~1,5 m) | luz forte ou janela lateral | de frente, outra roupa se der |

Em cada rodada, grave os 4 rótulos nesta ordem:

1. **arm_raised**: um braço erguido acima da cabeça. Alterne o braço no meio da sessão.
2. **arms_crossed**: braços cruzados no peito; varie a altura um pouco.
3. **arms_open**: braços abertos para os lados, na altura dos ombros ou um pouco abaixo.
4. **idle**: parado, braços soltos ao lado do corpo; pequenos movimentos naturais são ok.

Regras:
- Comece a pose **antes** da contagem de 3 s terminar; segure a pose a gravação inteira.
- Ombros e quadril sempre visíveis (frame incompleto vira `skipped`).
- Não grave `wave` agora (o Sprint 14 tem roteiro próprio).
- Depois: `uv run python scripts/dataset_stats.py` deve dizer `ready`.
