const test = require('node:test');
const assert = require('node:assert/strict');
const { Store, key, valid } = require('../web/preset-store.js');
const settings = { density: 71, grain: 25, pitch: 6.7, wet: .5, reverb: 1, jitter: .05, pan: .3, seed: 2025, variant: 'standard', strategy: 'centroid', 'ir-ms': 20, 'bank-size': 64, 'long-ir': 120, duration: 10, 'sample-rate': 48000 };
const memory = () => { const values = new Map();return { getItem: k => values.get(k) ?? null, setItem: (k, v) => values.set(k, v) }; };

test('complete settings survive a new library instance', () => {
  const storage = memory(), store = new Store(storage), input = {...settings};
  const saved = store.save('冰蓝空间', input);input.reverb = 0;
  const reopened = new Store(storage).list();
  assert.equal(reopened.length, 1);assert.equal(reopened[0].name, '冰蓝空间');
  assert.equal(reopened[0].id, saved.id);assert.deepEqual(reopened[0].values, settings);
});
test('duplicate names create distinct copies without replacing the original', () => {
  const storage = memory(), store = new Store(storage);
  const a = store.save('Space', settings), b = store.save('space', {...settings, reverb: .3});
  assert.equal(b.name, 'space (2)');assert.notEqual(a.id, b.id);
  assert.equal(store.list().find(p => p.id === a.id).values.reverb, 1);
});
test('invalid snapshots or names leave the library untouched', () => {
  const storage = memory(), store = new Store(storage);store.save('Original', settings);
  const before = storage.getItem(key);
  for (const patch of [{reverb: NaN}, {wet: 2}, {density: 71.3}, {strategy: 'unknown'}, {'sample-rate': 96000}, {extra: 1}]) {
    assert.throws(() => store.save('Bad', {...settings, ...patch}));assert.equal(storage.getItem(key), before);
  }
  for (const name of ['', ' ', 'a'.repeat(65), 'a\nb']) assert.throws(() => store.save(name, settings));
  assert.equal(valid(settings), true);
});
test('a broken entry does not prevent loading valid entries', () => {
  const storage = memory(), store = new Store(storage);store.save('Valid', settings);
  const data = JSON.parse(storage.getItem(key));data.presets.push({id:'broken',name:'Broken',values:{}});
  storage.setItem(key, JSON.stringify(data));assert.equal(store.list().length, 1);
});
test('unreadable library and storage failures do not claim a save', () => {
  const storage = memory(), store = new Store(storage);storage.setItem(key, '{broken');
  assert.throws(() => store.save('New', settings));assert.equal(storage.getItem(key), '{broken');
  const denied = new Store({getItem: () => null, setItem: () => { throw new Error('QuotaExceededError'); }});
  assert.throws(() => denied.save('New', settings), /storage is full or unavailable/);
});
