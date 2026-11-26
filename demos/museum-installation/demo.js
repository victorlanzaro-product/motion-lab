import { createInitialState, simulateGesture, pause, resume, THRESHOLD } from './experience.js';

export function mountDemo(document) {
  let state = createInitialState();
  const byId = (id) => document.getElementById(id);
  function render() {
    byId('count').textContent = String(state.count);
    byId('pause').disabled = state.paused;
    byId('pause').setAttribute('aria-pressed', String(state.paused));
    byId('resume').disabled = !state.paused;
    byId('clear-gesture').disabled = state.paused;
    byId('ambiguous-gesture').disabled = state.paused;
    byId('scene-status').textContent = state.paused ? 'INSTALAÇÃO PAUSADA' : state.status === 'triggered' ? 'LUZ ATIVADA' : state.status === 'uncertain' ? 'GESTO INSUFICIENTE' : 'AGUARDANDO GESTO';
    byId('orb').classList.toggle('active', state.lastEvent?.accepted === true && !state.paused);
    byId('beam').classList.toggle('active', state.lastEvent?.accepted === true && !state.paused);
    const event = byId('event');
    event.className = `event${state.paused ? ' paused' : state.lastEvent?.accepted === false ? ' uncertain' : ''}`;
    if (state.paused) event.innerHTML = '<strong>Instalação pausada.</strong><p>Retome para simular novos gestos. A pausa não cria eventos.</p>';
    else if (state.lastEvent) event.innerHTML = `<strong>${state.lastEvent.accepted ? 'Evento: luz ativada' : 'Nenhum evento disparado'}</strong><p>${state.lastEvent.cause} Pontuação sintética: ${state.lastEvent.score.toFixed(2)}. Limiar: ${THRESHOLD.toFixed(2)}.</p>`;
    else event.innerHTML = '<strong>Pronta para começar.</strong><p>Escolha uma ação para ver como a regra funciona.</p>';
  }
  byId('clear-gesture').addEventListener('click', () => { state = simulateGesture(state, 0.91); render(); });
  byId('ambiguous-gesture').addEventListener('click', () => { state = simulateGesture(state, 0.48); render(); });
  byId('pause').addEventListener('click', () => { state = pause(state); render(); });
  byId('resume').addEventListener('click', () => { state = resume(state); render(); });
  byId('restart').addEventListener('click', () => { state = createInitialState(); render(); });
  render();
  return { getState: () => ({ ...state, lastEvent: state.lastEvent && { ...state.lastEvent } }) };
}

if (typeof document !== 'undefined') mountDemo(document);
