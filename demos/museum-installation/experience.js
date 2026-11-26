export const THRESHOLD = 0.72;

export function createInitialState() {
  return { paused: false, count: 0, lastEvent: null, status: 'ready' };
}

export function simulateGesture(state, score) {
  if (state.paused) return { ...state, status: 'paused' };
  if (score < 0 || score > 1) throw new RangeError('score must be between 0 and 1');
  if (score < THRESHOLD) {
    return { ...state, lastEvent: { accepted: false, score, cause: 'A pontuação ficou abaixo do limiar.' }, status: 'uncertain' };
  }
  const count = state.count + 1;
  return { ...state, count, lastEvent: { accepted: true, score, cause: 'O gesto voluntário ultrapassou o limiar.' }, status: 'triggered' };
}

export function pause(state) {
  return { ...state, paused: true, status: 'paused' };
}

export function resume(state) {
  return { ...state, paused: false, status: 'ready' };
}
