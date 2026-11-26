import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createInitialState, simulateGesture, pause, resume, THRESHOLD } from '../experience.js';
import { mountDemo } from '../demo.js';

const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');

const controlIds = ['count', 'pause', 'resume', 'clear-gesture', 'ambiguous-gesture', 'scene-status', 'orb', 'beam', 'event', 'restart'];

function fakeDocument() {
  const elements = new Map(controlIds.map((id) => [id, {
    id, textContent: '', innerHTML: '', className: '', disabled: false,
    attributes: {}, classes: new Set(), listeners: {},
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, callback) { this.listeners[name] = callback; },
    click() { if (!this.disabled) this.listeners.click?.(); },
    classList: { toggle(name, force) { if (force) elements.get(id).classes.add(name); else elements.get(id).classes.delete(name); } },
  }]));
  return { getElementById: (id) => elements.get(id), elements };
}

test('clear voluntary gesture crosses threshold and triggers one visual event', () => {
  const next = simulateGesture(createInitialState(), 0.91);
  assert.equal(next.status, 'triggered');
  assert.equal(next.lastEvent.accepted, true);
  assert.equal(next.count, 1);
  assert.match(next.lastEvent.cause, /ultrapassou o limiar/);
  assert.equal(THRESHOLD, 0.72);
});

test('ambiguous or below-threshold gesture explains why it was not accepted', () => {
  const next = simulateGesture(createInitialState(), 0.48);
  assert.equal(next.status, 'uncertain');
  assert.equal(next.lastEvent.accepted, false);
  assert.equal(next.count, 0);
  assert.match(next.lastEvent.cause, /abaixo do limiar/);
});

test('pause blocks events and resume permits a new event', () => {
  const paused = pause(createInitialState());
  assert.equal(simulateGesture(paused, 0.91).count, 0);
  assert.equal(simulateGesture(paused, 0.91).status, 'paused');
  const resumed = resume(paused);
  assert.equal(resumed.paused, false);
  assert.equal(simulateGesture(resumed, 0.91).count, 1);
});

test('invalid synthetic scores are rejected', () => {
  assert.throws(() => simulateGesture(createInitialState(), 1.2), RangeError);
});

test('pause and resume after an accepted event preserve count and allow the next event', () => {
  const accepted = simulateGesture(createInitialState(), 0.91);
  const paused = pause(accepted);
  assert.equal(simulateGesture(paused, 0.91).count, 1);
  const resumed = resume(paused);
  const next = simulateGesture(resumed, 0.91);
  assert.equal(next.count, 2);
  assert.equal(next.lastEvent.accepted, true);
});

test('restart returns the installation to its initial state', () => {
  const triggered = simulateGesture(createInitialState(), 0.91);
  const restarted = createInitialState();
  assert.deepEqual(restarted, { paused: false, count: 0, lastEvent: null, status: 'ready' });
  assert.notDeepEqual(restarted, triggered);
});

test('DOM wiring renders accepted and below-threshold fixture outcomes', () => {
  const acceptedDoc = fakeDocument();
  mountDemo(acceptedDoc);
  acceptedDoc.getElementById('clear-gesture').click();
  assert.equal(acceptedDoc.getElementById('scene-status').textContent, 'LUZ ATIVADA');
  assert.equal(acceptedDoc.getElementById('count').textContent, '1');
  assert.match(acceptedDoc.getElementById('event').innerHTML, /Evento: luz ativada/);
  assert.equal(acceptedDoc.getElementById('orb').classes.has('active'), true);

  const ambiguousDoc = fakeDocument();
  mountDemo(ambiguousDoc);
  ambiguousDoc.getElementById('ambiguous-gesture').click();
  assert.equal(ambiguousDoc.getElementById('scene-status').textContent, 'GESTO INSUFICIENTE');
  assert.equal(ambiguousDoc.getElementById('count').textContent, '0');
  assert.match(ambiguousDoc.getElementById('event').innerHTML, /Nenhum evento disparado/);
  assert.equal(ambiguousDoc.getElementById('orb').classes.has('active'), false);
});

test('DOM pause, resume, and restart preserve lifecycle without replay or duplicates', () => {
  const document = fakeDocument();
  const demo = mountDemo(document);
  const get = (id) => document.getElementById(id);
  get('clear-gesture').click();
  assert.equal(get('count').textContent, '1');

  get('pause').click();
  assert.equal(get('scene-status').textContent, 'INSTALAÇÃO PAUSADA');
  assert.equal(get('clear-gesture').disabled, true);
  assert.equal(get('ambiguous-gesture').disabled, true);
  get('clear-gesture').click();
  assert.equal(get('count').textContent, '1');
  assert.equal(demo.getState().lastEvent.accepted, true);

  get('resume').click();
  assert.equal(get('count').textContent, '1', 'resume must not replay or duplicate the accepted event');
  assert.equal(demo.getState().lastEvent.accepted, true);
  get('clear-gesture').click();
  assert.equal(get('count').textContent, '2', 'only a fresh click adds a new event');

  get('restart').click();
  assert.equal(get('count').textContent, '0');
  assert.equal(demo.getState().lastEvent, null);
  assert.match(get('event').innerHTML, /Pronta para começar/);
  assert.equal(get('scene-status').textContent, 'AGUARDANDO GESTO');
});

test('page labels synthetic fixtures and loads only its local module', () => {
  assert.match(html, /Simulação local · sem câmera/);
  assert.match(html, /fixtures sintéticas/);
  assert.match(html, /Não há captura de movimento, visitante ou medição real/);
  assert.match(html, /<script type="module" src="\.\/demo\.js"><\/script>/);
  assert.doesNotMatch(html, /https?:\/\/|\bfetch\s*\(|\bWebSocket\b|getUserMedia|localStorage|sessionStorage/i);
});
