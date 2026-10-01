"use strict";
// A bounded stereo analyzer for each side. The source branch is inaudible.
class OrbitAudioDisplays {
  constructor(audio) {
    this.audio = audio; this.frame = 0; this.lastTime = 0; this.sourceVersion = 0;
    this.views = ["Waveform", "Spectrum", "Spectrogram"];
    this.colors = Array.from({length:256}, (_, i) => this.heat(i / 255));
    this.plots = ["input", "output"].map((name, i) => {
      let mode = i; try { const stored = localStorage.getItem(`orbit.view.${name}`); if (stored !== null && /^[0-2]$/.test(stored)) mode = Number(stored); } catch {}
      const history = document.createElement("canvas"); history.width = 192; history.height = 64;
      const plot = { name, mode, button: document.getElementById(`${name}-plot`), label: document.getElementById(`${name}-view-mode`),
        canvas: document.getElementById(i ? "output-spectrum" : "source-wave"), levels: new Float32Array(64), history,
        historyContext: history.getContext("2d"), column: 0, empty: 192, peaks: [] };
      plot.button.addEventListener("click", () => {
        plot.mode = (plot.mode + 1) % 3; this.label(plot); this.draw(plot);
        try { localStorage.setItem(`orbit.view.${name}`, String(plot.mode)); } catch {}
      });
      this.label(plot); this.clear(plot); return plot;
    });
    ["play", "pause", "ended"].forEach(event => audio.addEventListener(event, () => { this.syncSource(); this.schedule(); }));
    audio.addEventListener("seeking", () => { this.stopSource(); this.clearAll(); });
    audio.addEventListener("seeked", () => { this.syncSource(); this.schedule(); });
    document.addEventListener("visibilitychange", () => { if (!document.hidden) this.schedule(); });
  }
  label(plot) {
    plot.label.textContent = `${this.views[plot.mode].toUpperCase()} ›`;
    plot.button.setAttribute("aria-label", `${plot.name.toUpperCase()} visualization: ${this.views[plot.mode]}. Click to switch view`);
  }
  clear(plot) { plot.levels.fill(0); plot.historyContext.fillStyle = "#0c1118"; plot.historyContext.fillRect(0,0,192,64); plot.column = 0; plot.empty = 192; }
  clearAll() { this.plots.forEach(plot => this.clear(plot)); this.drawAll(); }
  setSource(url, peaks) { this.sourceUrl = url; this.sourceVersion++; this.inputBuffer = null; this.stopSource(); this.plots[0].peaks = peaks; this.clearAll(); }
  setOutput(peaks) { this.plots[1].peaks = peaks; this.clearAll(); }
  async prepare() {
    if (!this.context) {
      this.context = new AudioContext();
      this.outputNode = this.context.createMediaElementSource(this.audio);
      this.plots.forEach(plot => {
        plot.stereo = this.context.createGain(); plot.stereo.channelCount = 2; plot.stereo.channelCountMode = "explicit";
        plot.stereo.channelInterpretation = "speakers"; plot.splitter = this.context.createChannelSplitter(2); plot.stereo.connect(plot.splitter);
        plot.analyzers = [0,1].map(channel => {
          const node = this.context.createAnalyser(); node.fftSize = 4096; node.smoothingTimeConstant = 0;
          plot.splitter.connect(node, channel); return node;
        });
        plot.frequency = [new Float32Array(2048),new Float32Array(2048)];
        plot.wave = [new Float32Array(4096),new Float32Array(4096)];
      });
      this.outputNode.connect(this.plots[1].stereo); this.plots[1].stereo.connect(this.context.destination);
      this.mute = this.context.createGain(); this.mute.gain.value = 0;
      this.plots[0].stereo.connect(this.mute); this.mute.connect(this.context.destination);
    }
    await this.context.resume();
    if (!this.inputBuffer) {
      const version = this.sourceVersion, response = await fetch(this.sourceUrl);
      if (!response.ok) throw new Error("Source audio has expired. Import it again.");
      const buffer = await this.context.decodeAudioData(await response.arrayBuffer());
      if (version !== this.sourceVersion) throw new Error("Source changed. Start playback again.");
      this.inputBuffer = buffer;
    }
  }
  stopSource() { if (this.inputNode) { this.inputNode.stop(); this.inputNode.disconnect(); this.inputNode = null; } }
  syncSource() {
    this.stopSource();
    if (this.context && this.inputBuffer && !this.audio.paused && !this.audio.ended) {
      this.inputNode = this.context.createBufferSource(); this.inputNode.buffer = this.inputBuffer; this.inputNode.loop = true;
      this.inputNode.connect(this.plots[0].stereo); this.inputNode.start(0, this.audio.currentTime % this.inputBuffer.duration);
    }
  }
  schedule() { if (!this.frame) this.frame = requestAnimationFrame(time => this.tick(time)); }
  heat(value) {
    const stops = [[12,17,24],[32,67,101],[79,170,241],[228,247,255]], points = [0,.35,.75,1];
    const k = value < .35 ? 0 : value < .75 ? 1 : 2, t = Math.max(0, Math.min(1, (value-points[k])/(points[k+1]-points[k])));
    return `rgb(${stops[k].map((v,i) => Math.round(v+(stops[k+1][i]-v)*t)).join(",")})`;
  }
  tick(time) {
    this.frame = 0; if (document.hidden) return;
    const playing = this.context && !this.audio.paused && !this.audio.ended;
    if (time - this.lastTime >= 32) {
      this.lastTime = time;
      const rate = this.context?.sampleRate || 48000, upper = Math.min(20000, rate/2);
      this.plots.forEach(plot => {
        if (playing) plot.analyzers.forEach((node,i) => { node.getFloatFrequencyData(plot.frequency[i]); node.getFloatTimeDomainData(plot.wave[i]); });
        const write = playing || plot.empty < 192;
        plot.levels.forEach((level,i) => {
          let peak = -90;
          if (playing) {
            const first = Math.max(1, Math.floor(20*Math.pow(upper/20,i/64)*4096/rate));
            const last = Math.min(2047, Math.ceil(20*Math.pow(upper/20,(i+1)/64)*4096/rate));
            for (let bin = first; bin <= last; bin++) {
              const power = (Math.pow(10,plot.frequency[0][bin]/10)+Math.pow(10,plot.frequency[1][bin]/10))/2;
              peak = Math.max(peak,10*Math.log10(Math.max(power,1e-12)));
            }
          }
          const target = Math.max(0,Math.min(1,(peak+90)/90));
          plot.levels[i] = level+(target-level)*(target>level?.65:.16); if (plot.levels[i]<.001) plot.levels[i]=0;
          if (write) { plot.historyContext.fillStyle = this.colors[Math.round(target * 255)]; plot.historyContext.fillRect(plot.column,63-i,1,1); }
        });
        if (write) { plot.column = (plot.column+1)%192; plot.empty = playing ? 0 : plot.empty+1; }
        if (plot.canvas.getBoundingClientRect().width) this.draw(plot);
      });
    }
    if (playing || this.plots.some(plot => plot.empty<192 || plot.levels.some(v=>v>0))) this.schedule();
  }
  drawAll() { this.plots.forEach(plot => this.draw(plot)); }
  draw(plot) {
    const {width:w,height:h} = plot.canvas.getBoundingClientRect(); if (!w || !h) return;
    const ratio = Math.min(devicePixelRatio||1,2), ctx = plot.canvas.getContext("2d");
    if (plot.canvas.width!==Math.round(w*ratio) || plot.canvas.height!==Math.round(h*ratio)) { plot.canvas.width=Math.round(w*ratio); plot.canvas.height=Math.round(h*ratio); }
    ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
    const top=4,bottom=h-17,height=bottom-top,rate=this.context?.sampleRate||48000,upper=Math.min(20000,rate/2);
    const playing = this.context && !this.audio.paused && !this.audio.ended;
    if (plot.mode===0) {
      for (let channel=0;channel<2;channel++) {
        const centre=top+height*(channel?.75:.25), amplitude=height*.23;
        ctx.strokeStyle="#263447";ctx.lineWidth=.6;ctx.beginPath();ctx.moveTo(0,centre);ctx.lineTo(w,centre);ctx.stroke();
        ctx.strokeStyle=channel?"#82cfff99":"#82cfff";ctx.lineWidth=1;ctx.beginPath();
        const samples=playing?plot.wave[channel]:plot.peaks;
        const count=Math.min(180,samples?.length||0);
        for (let i=0;i<count;i++) {
          let low=0,high=0;
          for(let j=Math.floor(i*samples.length/count);j<Math.floor((i+1)*samples.length/count);j++) {
            if(playing){low=Math.min(low,samples[j]);high=Math.max(high,samples[j]);}else{low=Math.min(low,-samples[j]);high=Math.max(high,samples[j]);}
          }
          const x=i*w/Math.max(1,count-1);ctx.moveTo(x,centre-Math.min(1,high)*amplitude);ctx.lineTo(x,centre-Math.max(-1,low)*amplitude);
        }
        ctx.stroke();
      }
    } else if(plot.mode===1) {
      ctx.strokeStyle="#263447";ctx.lineWidth=.6;
      for(const level of [0,.5,1]){const y=bottom-level*height;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
      ctx.beginPath();ctx.moveTo(0,bottom);plot.levels.forEach((v,i)=>ctx.lineTo(i*w/63,bottom-v*height));ctx.lineTo(w,bottom);ctx.closePath();
      const fill=ctx.createLinearGradient(0,top,0,bottom);fill.addColorStop(0,"#82cfff30");fill.addColorStop(1,"#82cfff02");ctx.fillStyle=fill;ctx.fill();
      ctx.beginPath();plot.levels.forEach((v,i)=>{if(i)ctx.lineTo(i*w/63,bottom-v*height);else ctx.moveTo(0,bottom-v*height);});ctx.strokeStyle="#82cfff";ctx.lineWidth=1.2;ctx.stroke();
    } else {
      const remaining=192-plot.column,split=w*remaining/192;ctx.imageSmoothingEnabled=false;
      ctx.drawImage(plot.history,plot.column,0,remaining,64,0,top,split,height);
      if(plot.column)ctx.drawImage(plot.history,0,0,plot.column,64,split,top,w-split,height);
      ctx.fillStyle="#91a5bc";ctx.font="8px Oxanium, sans-serif";ctx.textAlign="left";ctx.textBaseline="top";ctx.fillText(`${Math.round(upper/1000)}k`,2,top+1);ctx.textBaseline="bottom";ctx.fillText("20",2,bottom);
    }
    ctx.fillStyle="#91a5bc";ctx.font="9px Oxanium, sans-serif";ctx.textBaseline="bottom";ctx.textAlign="left";
    ctx.fillText(plot.mode===0?(playing?`${Math.round(4096/rate*1000)} ms`:"OVERVIEW"):plot.mode===1?"20":"-6.4s",0,h);
    if(plot.mode===1){ctx.textAlign="center";ctx.fillText("1k",w*Math.log(50)/Math.log(upper/20),h);}
    ctx.textAlign="right";ctx.fillText(plot.mode===1?`${Math.round(upper/1000)}k`:plot.mode===2||playing?"NOW":"",w,h);
  }
}
