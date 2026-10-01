const test = require('node:test');
const assert = require('node:assert/strict');
const {squareToDisc,discToSquare} = require('../web/orbit-geometry.js');
test('all parameter combinations stay in the circle and roundtrip', () => {
  for(let i=-25;i<=25;i++)for(let j=-25;j<=25;j++) {
    const x=i/25,y=j/25,d=squareToDisc(x,y),s=discToSquare(...d);
    assert.ok(Math.hypot(...d)<=1+1e-10);
    assert.ok(Math.abs(s[0]-x)<1e-7 && Math.abs(s[1]-y)<1e-7);
  }
});
test('outside drags project to the circle, preserving extremes and centre', () => {
  assert.deepEqual(discToSquare(0,0),[0,0]);
  for(const point of [[5,0],[-5,0],[0,5],[0,-5],[5,5],[-5,-5]]) {
    const s=discToSquare(...point),d=squareToDisc(...s);
    assert.ok(Math.abs(Math.hypot(...d)-1)<1e-7);
    assert.ok(Math.abs(d[0]-point[0]/Math.hypot(...point))<1e-7);
  }
});
