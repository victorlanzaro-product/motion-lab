import test from 'node:test';
import assert from 'node:assert/strict';
import { createInitialState, simulateGesture, pause, resume, THRESHOLD } from '../experience.js';

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
