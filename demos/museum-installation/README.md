# Instalação interativa — gesto e luz

Protótipo local para demonstrar uma interação de museu/estúdio. Um botão simula o gesto voluntário de levantar a mão. Um fixture acima do limiar ativa a luz e explica o evento. Um fixture abaixo do limiar mostra por que nenhum evento foi aceito. Pausar bloqueia os gestos simulados; retomar reabilita-os; reiniciar limpa a sessão.

Todos os eventos e valores são sintéticos. A página não abre a câmera, não faz chamadas de rede externa, não grava estado e não se conecta ao runtime do Motion Lab. O cenário de setup não existe nesta demo e não foi testado; o protótipo cobre somente a interação local após a página abrir.

O protocolo anterior da Íris é um modelo sintético separado: usa confiança `≥ 0,80` e duração `≥ 1,0 s`. Esta UI implementa apenas pontuação sintética com limiar `0,72`, sem duração. A demo não invoca o script da Íris, e o protocolo não valida esta UI nem reconhecimento real.

## Abrir

Abra `index.html` diretamente no navegador. Para navegadores que bloqueiam módulos ES em `file://`, execute um servidor somente local na pasta do repositório:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Acesse `http://127.0.0.1:8000/demos/museum-installation/`. Esse servidor é opcional e permanece restrito à própria máquina.

## Testar

```sh
node --test demos/museum-installation/tests/experience.test.js
```

Os testes verificam lógica e os controlos da UI por um DOM fake determinístico: gesto claro e abaixo do limiar, pausa após evento aceito, suspensão sem duplicação, retomada sem replay, novo evento somente após ação nova, reinício do estado e rejeição de pontuação inválida. O DOM fake cobre o wiring dos botões e a renderização de estado; não é um teste de navegador real ou leitor de tela.

## Evidência visual

As capturas reais da UI executada e clicada estão em `evidence/` e têm os hashes SHA-256 registrados no índice compartilhado `Shared/produto/motion-lab/evidencia-instalacao-interativa-2026-11-26/README.md`. Foram feitas sem pessoa ou câmera, com banner explícito de simulação. A página foi servida apenas em loopback (`127.0.0.1`); não houve acesso à rede externa.
