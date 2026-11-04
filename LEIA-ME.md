# Cardápio Mesa 3D (protótipo)

Cardápio que abre pelo QR Code da mesa, mostra os pratos em 3D e, no celular, deixa o cliente ver o prato em tamanho real sobre a mesa (realidade aumentada). O pedido é enviado pelo WhatsApp.

## Arquivos
- `index.html`: o cardápio (pratos, preços e textos ficam na lista `DISHES` no fim do arquivo).
- `models/*.glb`: modelos 3D em escala real (metros).
- `gen_models.py`: script que gerou os modelos de exemplo (Python + numpy).

## Testar a realidade aumentada
A RA só funciona com o site publicado em HTTPS e aberto no celular:
1. Publique a pasta inteira num serviço de hospedagem estática gratuito (Netlify, Vercel, GitHub Pages ou Cloudflare Pages). No Netlify dá para arrastar a pasta para a página de deploy.
2. Abra o endereço no celular. O botão "Ver na sua mesa" aparece em cada prato.
   - Android: abre o Scene Viewer do Google (precisa do Google Play Services for AR).
   - iPhone: abre o Quick Look da Apple.

## Próximos passos
- Trocar os modelos ilustrativos por capturas reais dos pratos (fotogrametria), mantendo cada arquivo abaixo de ~3 MB.
- Colocar o número do restaurante no link do WhatsApp (`https://wa.me/55DDDNUMERO?text=`).
- Ler o número da mesa pelo QR Code (ex.: `?mesa=07`).
- Painel para o restaurante editar pratos e preços.
