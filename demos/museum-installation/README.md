# Instalação interativa — gesto e luz

Protótipo local para demonstrar uma interação de museu/estúdio. Um botão simula o gesto voluntário de levantar a mão. Um fixture acima do limiar ativa a luz e explica o evento. Um fixture abaixo do limiar mostra por que nenhum evento foi aceito. Pausar bloqueia os gestos simulados; retomar reabilita-os; reiniciar limpa a sessão.

Todos os eventos e valores são sintéticos. A página não abre a câmera, não usa rede, não grava estado e não se conecta ao runtime do Motion Lab.

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

Os testes verificam gesto claro, gesto abaixo do limiar, pausa/retomada e rejeição de pontuação inválida. Eles exercitam a lógica pura, sem browser, câmera ou rede.
