# Regras de trabalho dos agentes Maestri

Este repositório é público. Todo trabalho deve ser local e usar somente exemplos sintéticos e conteúdo já aprovado.

## Papéis

- **Programador:** implementa uma tarefa aberta, uma por vez, em sua branch de tarefa. Mantém o escopo e os critérios definidos em `TAREFAS.md`.
- **Revisor:** não altera arquivos. Confere o diff contra o objetivo, critérios, testes, privacidade e escopo; registra achados concretos.
- **Testador:** verifica o comportamento sem alterar código, executa testes e checks do projeto e registra comandos, resultados e limitações.

## Fluxo obrigatório

1. Leia este arquivo, `README.md`, o código relacionado e `TAREFAS.md`.
2. Pegue apenas uma tarefa com status **aberta**. Altere para **em andamento** e registre a branch.
3. Crie uma branch no formato `tN-descricao`, por exemplo `t1-foto-folha`. Faça todo o trabalho e commits nessa branch; nunca faça commit na `main`.
4. Antes de marcar **pronta para revisão**, rode os testes e checks aplicáveis documentados no README e no projeto. Para a suíte principal, use `uv run pytest -q`. Rode também a demonstração relevante quando o critério da tarefa pedir. Registre os resultados em `TAREFAS.md` ou na descrição do PR.
5. Solicite revisão por PR. Só marque **aprovada** após revisão da Helena (OpenBot) e aprovação do Victor.

## Limites

- Não use dados pessoais ou reais, segredos, tokens, credenciais, arquivos de configuração pessoal ou dados biométricos reais. Use apenas dados fictícios claramente identificados.
- Não grave, armazene, envie ou publique vídeo. Não adicione imagens ou gravações de pessoas reais.
- Mantenha processamento, dependências e demonstrações locais; não use serviços, APIs ou armazenamento remoto para processar dados.
- Não faça merge, publicação adicional, deploy, convite ou mudança de acesso. Merge na `main` só depois da revisão da Helena e do ok explícito do Victor.
