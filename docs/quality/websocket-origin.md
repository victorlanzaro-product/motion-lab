# Política de Origin do WebSocket local

O handshake de `/ws` aceita uma única origem HTTP(S) que corresponda à allowlist explícita. O servidor rejeita Origin ausente, malformado ou não listado antes de aceitar a conexão, enviar `hello` ou inscrever o cliente no pipeline. Valores repetidos no cabeçalho também são rejeitados.

O `Host` da requisição não autoriza nem restringe uma origem: um Host local não torna um Origin externo confiável, e um Host externo não amplia a allowlist. Isso é uma defesa contra páginas web não confiáveis que tentem usar o navegador para alcançar o serviço local; não autentica clientes que abram conexões WebSocket diretamente.

O CLI mantém bind loopback por padrão e permite modo LAN somente com `--allow-lan` e ao menos um `--allowed-origin` explícito. As origens locais padrão incluem o endereço usado no bind, `127.0.0.1`, `localhost` e `[::1]` na porta escolhida.

Revisões anteriores de Sprint 16 reproduziram a falha de confiar em Origin/Host externo coincidentes. Os testes desta PR fixam o caso no handshake e na política unitária, sem conexão externa, câmera ou dados reais. A validação de Origin não deve ser descrita como autenticação.
