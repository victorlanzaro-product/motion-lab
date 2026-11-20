# Sprint 16 — Hardening e avaliação ao vivo

Integração em `sprint-16` sobre a S15. Fechamento do [Roadmap v2](../ROADMAP-v2.md).

## Recorder e falhas do backend

Um objeto JSON com `type: "record_start"` sem rótulo textual válido recebe
uma rejeição privada `record_status`. Nenhuma reserva ou gravação é iniciada.
Os campos de sessão, contagens e dono refletem o estado no servidor para
aquele solicitante. Exemplo com servidor ocioso:

```json
{"type":"record_status","recording":false,"label":null,"session_id":null,"written":0,"skipped":0,"owner":false,"error":"Informe um rótulo de gravação válido."}
```

JSON ilegível, binário e comandos desconhecidos continuam ignorados. Mensagens
de erro do backend exibidas pela aplicação ficam em português, preservando a
causa útil para a pessoa que opera a câmera. Detalhes técnicos ficam no log
local. A conexão permanece ativa em falhas degradáveis e depois da rejeição
de um comando de gravação.

## Avaliação

- `uv run python scripts/evaluate_live.py --model-v1 CAMINHO --model-v2 CAMINHO`
  é o ponto de entrada local para Victor. Valide os dois bundles antes de
  abrir a câmera. Sem modelo válido, encerre com uma mensagem em português.
- Avalie v1, v2 e regra sobre os mesmos frames da mesma sessão. Reaproveite
  os classificadores, a janela e o motor de gestos da aplicação. Explique
  como os eventos da regra `wave` são comparados com rótulos por frame e
  relate o resultado por trecho de aceno para essa decisão.
- Use trechos de avaliação novos, separados do dataset de treino. O roteiro
  inclui `idle`, `arm_raised`, `arms_open`, `arms_crossed` e `wave`,
  com acenos nos dois braços e subidas sem reversão. Victor informa o rótulo
  esperado por trecho. Trocas e aquecimento podem ser excluídos da comparação
  se forem contados separadamente.
- Reporte cobertura por fonte, pares comparáveis e concordância por par. A
  taxa é `matches / comparable` e fica `null` com denominador zero.
  Compare v1 no vocabulário comum e mostre a cobertura própria de `wave`
  para v2 e regra. Concordância não substitui o confronto com o rótulo
  esperado dos trechos.
- A CLI pode emitir apenas um resumo JSON local. Exemplo de estrutura para
  um trecho de 90 frames com rótulos disponíveis em parte do trecho:

```json
{"schema_version":1,"segments":[{"label":"arm_raised","frames":90,"available":{"v1":90,"v2":62,"rule":86},"agreement":{"v1_v2":{"comparable":62,"matches":52,"rate":0.8387},"v1_rule":{"comparable":86,"matches":75,"rate":0.8721},"v2_rule":{"comparable":60,"matches":51,"rate":0.85}}}]}
```

Não persista imagem, vídeo ou dados faciais. O resumo contém contagens e
taxas agregadas, sem identificação pessoal. A arquitetura em produção mantém
uma câmera e o frontend não calcula features.

## Escopos e aceite

- Backend hardening: `backend/web/{app,pipeline,payload,recorder}.py` e
  testes do protocolo e da degradação. Frontend só se a mensagem em português
  exigir adaptação no consumo do protocolo.
- Avaliação: `scripts/evaluate_live.py`, módulo de avaliação em
  `backend/ml/` e testes determinísticos com entradas sintéticas. O mesmo
  caminho de inferência usado pela aplicação deve ser reaproveitado.
- Maestro: roteiro com comandos, revisão de arquitetura, registro de
  decisões e resultados medidos no README.
- Rode pytest, Node e smoke sem câmera. Peça a revisão de segurança da
  superfície local de câmera e WebSocket, além das revisões da sprint.
  Depois do checkpoint ao vivo com Victor, documente os números, a
  cobertura e a decisão de manter ou aposentar `_WaveDetector` com base
  nos acenos, nas subidas sem reversão e na disponibilidade do v2.
