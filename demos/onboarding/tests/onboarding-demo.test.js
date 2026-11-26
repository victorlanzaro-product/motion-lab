import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');

test('demo labels its events as synthetic and explains the limit', () => {
  assert.match(html, /fixture sintética/i);
  assert.match(html, /não representam medição real/);
  assert.match(html, /não descrevem uma pessoa/);
});

test('demo uses no camera, external script, or network API', () => {
  assert.doesNotMatch(html, /https?:\/\//i);
  assert.doesNotMatch(html, /\bfetch\s*\(|\bWebSocket\b|getUserMedia|<script\b[^>]*\bsrc\s*=/i);
});

test('filter controls expose accessible selected state and live event updates', () => {
  assert.match(html, /role="group" aria-label="Filtrar eventos"/);
  assert.match(html, /aria-pressed="true"/);
  assert.match(html, /aria-live="polite"/);
  assert.match(html, /event\.hidden=filter!=='all'/);
});
