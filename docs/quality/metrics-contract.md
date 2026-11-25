# Contrato das métricas existentes na main

Este PR fixa por testes os contratos já presentes em `main` para `compute_agreement`, `rule_label` e `dataset_stats`.

- Concordância usa no denominador somente frames com rótulo ML comparável e regra determinável; `wave`, ML ausente e regra indeterminada não contam como comparação.
- Uma taxa medida de `0.0` é diferente de nenhuma comparação: nesse caso, `rate` continua presente e vale `null` no JSON.
- Valores numéricos observados iguais a zero não são dados ausentes. Um predicado ausente ou `None` permanece indeterminado e não pode ser convertido em `idle`.
- Dataset inexistente (`exists=false`) é distinto de CSV presente só com cabeçalho (`exists=true`, zero linhas, não pronto). Duas sessões válidas com duração observada zero continuam sendo observações válidas e podem satisfazer o mínimo de sessões.

## Fora deste PR

`main` no SHA de base não contém `backend/ml/live_eval.py` nem `scripts/evaluate_live.py`. Portanto este PR **não** declara cobertura de avaliação live, warmup, transições, segmentos elegíveis ou denominador de `wave`. Esses contratos dependem de integrar e revisar a Sprint 16; devem ser cobertos numa alteração futura baseada no código integrado.
