# Motion Lab — Roadmap v3 (proposta, Sprints 17–20)

**24/11/2026 · Proposta de produto para revisão via PR.** A proposta não está aprovada para merge/execução. O repositório tem PRs abertas #5, #8, #9 e #10; PRs #5/#8 estão em ajustes conforme revisão de Helena. Nenhuma câmera foi usada.

## Entregas de planejamento desta noite

Seleção para o trabalho desta noite: (1) `scripts/doctor.py`, diagnóstico offline de primeira execução (implementado em PR #10, ainda não integrado); (2) demo estática e interativa sem câmera, Python ou rede, com dados sintéticos (protótipo local sem PR até resolver o gate de testes). Critérios abaixo separam teste técnico de hipótese de valor. A análise de trade-offs está em `oportunidades-features-v2.md`.

### Sprint 17 — Diagnóstico local de primeira execução

**Objetivo:** ajudar quem clona o projeto a identificar requisitos ausentes e o próximo passo sem inicializar câmera ou transmitir dados.

**Escopo:** PR #10 adiciona `scripts/doctor.py`, que verifica versão Python suportada, presença de dependências por extras, modelo de pose local, diretórios/permissões e indica que câmera/permissões serão necessárias apenas ao iniciar app ao vivo. Não instala pacotes, não baixa modelos, não acessa webcam/rede e não altera configuração.

**Pronto quando:**

- Casos saudáveis e ausentes geram instruções claras e status/exit code estáveis.
- Testes unitários cobrem versões incompatíveis, extras ausentes, modelo/diretório ausente e estado completo sem depender de câmera, rede ou modelo real.
- README tem uma seção de primeira execução; mensagens distinguem “dependência faltando” de “câmera ainda não testada”.
- Teste de valor futuro (hipótese): com 5 pessoas novas, ao menos 4 encontram o requisito faltando e entendem o próximo passo sem ajuda. A Íris não entrevistou ninguém; isso não está validado.

**Câmera Victor:** não requerida para implementar, testar ou avaliar este sprint.

### Sprint 18 — Demo estática offline com dados sintéticos

**Objetivo:** permitir que alguém explore a experiência principal antes de instalar Python ou conectar câmera.

**Escopo:** protótipo local em `demos/offline/index.html`; PR ainda não aberta. Uma página local independente com fluxos demonstráveis de diagnóstico/enquadramento, gesto detectado, ausência de pose e degradação. Usar fixtures marcadas persistentemente como **SIMULAÇÃO — dados fictícios**; nenhuma API, câmera, dependência Python ou asset remoto. Não imitar confiança/saúde como medição real.

**Pronto quando:**

- Abre como arquivo local com rede desligada e funciona sem backend/Python; nenhum recurso externo é requisitado.
- Cada tela usa payload sintético versionado, todos os estados e lacunas são explícitos; nenhuma feature visual sugere que houve inferência real.
- Testes determinísticos de navegação/transições ainda não foram adicionados. A execução do pytest no clone local para no baseline por falta do extra opcional `mediapipe`; a demo não tem teste frontend automatizado ainda. Antes de PR, adicionar teste determinístico independente e registrar claramente o gate Python que falha também na `main` limpa.
- Teste de valor futuro (hipótese): com 5 pessoas novas, ao menos 4 explicam o que o produto faz e reconhecem que a demonstração usa dados fictícios.

**Câmera Victor:** não requerida.

## Propostas futuras, condicionadas

### Sprint 19 — Calibração e diagnóstico guiados por sinais locais

**Objetivo:** orientar o usuário a ajustar enquadramento e tentar um gesto usando apenas sinais existentes (pose/juntas visíveis, estado do pipeline), sem gravar vídeo/frames/rosto/malha.

**Critério proposto:** estados sintéticos de corpo ausente, enquadramento incompleto, juntas necessárias visíveis, stream degradado e gesto reconhecido produzem instruções certas e não declaram postura/saúde. As instruções nunca devem inventar confiança nem interpretar ausência como `idle`.

**Checkpoint de câmera do Victor:** sim, após testes offline e aprovação técnica/privacidade; testar em sessão local se as instruções resolvem enquadramento e gesto sem ajuda. Métricas a registrar: conclusão, tempo, falhas e estados que não receberam orientação útil; não persistir imagens.

**Hipótese de valor a testar:** meta exploratória sugerida anteriormente: 4/5 pessoas completam enquadramento + gesto simples em até 2 minutos sem ajuda. Não é fato nem critério aprovado.

### Sprint 20 — Ações locais por gesto (inicialmente sem integração global)

**Objetivo:** ligar gestos estáveis a ações úteis, seguras e reversíveis dentro do próprio app.

**Escopo proposto:** 2–3 mapeamentos allowlisted; ação arm/disarm explícita, pausa, cooldown/debounce, indicador visível, cancelamento e timeout. Sem atalhos globais do SO, OSC ou comandos irreversíveis até definição de tarefa e salvaguardas.

**Pronto quando:** eventos sintéticos provam que gesto válido armado aciona exatamente uma ação permitida; gesto desconhecido, estado desarmado, baixa disponibilidade, aquecimento, gap ou reconexão não acionam; disarm/cancelamento sempre interrompem. Testes offline não provam reconhecimento de gesto.

**Checkpoint de câmera do Victor:** sim, com casos rotulados por trecho e tentativas de gesto/controle negativo; medir acertos, falsos disparos e disponibilidade por classe. Usar `scripts/evaluate_live.py` se a linha S16 já estiver aprovada/atualizada; confirmar primeiro o contrato e o estado das PRs, sem salvar vídeo/frames/face. Não liberar action mapping se falsos disparos não estiverem compreendidos.

## Gates humanos que permanecem antes de alegar qualidade do reconhecimento

- **S13 — rótulos posturais reais:** coletar sessions separadas e rotuladas `arm_raised`, `arms_crossed`, `arms_open`, `idle`, variando condições segundo o protocolo aprovado. Só Victor opera a câmera; sem persistir pixels. Depois verificar cobertura e executar LOSO por sessões inteiras, mantendo sessões de teste fora do treino. Se faltarem sessões/rótulos, resultado inconclusivo.
- **S14 — `wave` real:** coletar sessões rotuladas com reversões reais e ambos os braços/condições, além de `arm_raised` sem reversão. LOSO temporal v2 em sessões não usadas no treino; relatar F1 por classe e confusões fora da amostra. Sintéticos provam plumbing, não qualidade.
- **S16 — comparação ao vivo:** antes da sessão, verificar bundles v1/v2 compatíveis; usar trechos frescos não usados no treino e rótulos esperados informados por trecho. Comparar v1/v2/regra nos mesmos frames; contar aquecimento/transições separadamente; concordância só entre pares comparáveis e não substitui o rótulo esperado. A CLI deve emitir JSON agregado, sem imagem, vídeo ou informação facial persistida. Victor decide/realiza checkpoint local; aqui não foi executado.

**Privacidade invariável:** tudo no dispositivo; sem cloud; nenhum vídeo, frame, rosto ou malha facial em disco/logs/relatórios. Guardar somente features/dados rotulados permitidos e resultados agregados, sob consentimento/autorizações aplicáveis.

## Sequenciamento e dependências

- S17 e S18 podem ser prototipados independentemente de hardware. S17 é primeiro para melhorar onboarding; S18 oferece exploração zero-setup.
- S19 deve consumir sinais do pipeline atual, sem implementar um segundo pipeline no browser. S20 depende de segurança e estabilidade dos gestos e dos resultados dos checkpoints S14/S16.
- PRs #5 (S15) e #8 (S16) continuam abertas e com conflitos relatados; Helena recomenda #5 antes de #8. Comentários em ambas registram ajustes importantes. PR #10 (doctor) parte de `main` HEAD `ea1b714`; PR #9 (OSS readiness) também parte da `main`, HEAD após correção `4352121`. Não há PR para demo até testes focados existirem e o gate ser esclarecido. Antes de outra feature que dependa de S16, confirmar estado e seguir a sequência de Helena. Branch: S17/S18 independentes podem sair de `main` somente após alinhar HEAD; S19/S20 devem partir de `sprint-16` e manter PR empilhada nessa branch enquanto a base não integrar. Sem mudança no código nesta entrega.
- **OSS readiness:** escolher e adicionar licença explícita, preparar instruções de contribuição/segurança e garantir setup/checks reproduzíveis antes de chamar a experiência de pronta para contribuições públicas. O root público observado não listava `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md` nem workflow CI; confirmar no HEAD antes de priorizar.

## Entrega desta noite vs futuro

- **Entregue agora:** este rascunho com objetivos, escopo, critérios, distinção entre simulação/valor e checkpoints humanos.
- **Planejado, não executado:** scripts/demo S17–S18 e qualquer código, testes, PR, branch ou merge.
- **Futuro condicionado:** validação S13/S14/S16 com câmera pelo Victor e qualquer liberação de calibração ou ações por gesto.

### Fontes de base

- Repositório `victorlanzaro-product/motion-lab`: README, `docs/ROADMAP-v2.md`, `docs/sprints/15-dashboard.md` no head da PR #5 e `docs/sprints/16-hardening-avaliacao.md` no head da PR #8.
- `Shared/FATOS-E-REGRAS.md` e `Shared/produto/CONTEXTO.md`, lidos em 24/11/2026.
- Estados PR #5 (`cdcbc445`) e #8 (`44f837f`) consultados em 24/11/2026; revalidar antes de implementação/integração.
\n\n---\n\nCo-authored-by: OpenBot <335604065+openbotgit[bot]@users.noreply.github.com>\n