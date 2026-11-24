import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');

test('offline demo declares synthetic data and no real measurement', () => {
  assert.match(html, /Dados de demonstração/);
  assert.match(html, /não representam medição real/);
  assert.match(html, /Sequência sintética/);
});

test('offline demo has no network, camera, or external script access', () => {
  assert.doesNotMatch(html, /https?:\/\//i);
  assert.doesNotMatch(html, /\bfetch\s*\(|\bWebSocket\b|getUserMedia|<script\b[^>]*\bsrc\s*=/i);
});

test('offline demo exposes keyboard-operable controls and live event feedback', () => {
  assert.match(html, /<button\b[^>]*data-pose="neutral"/);
  assert.match(html, /<button\b[^>]*data-pose="raised"/);
  assert.match(html, /aria-pressed="true"/);
  assert.match(html, /aria-live="polite"/);
});
