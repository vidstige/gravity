import init, { Simulation, step, render } from "../pkg/gravity";

function getPosition(canvas: HTMLCanvasElement, event: MouseEvent) {
  const rect = canvas.getBoundingClientRect();
  return {x: event.clientX - rect.left, y: event.clientY - rect.top};
}

class Gaussian {
  uniform: () => number;
  constructor(uniform: () => number) {
    this.uniform = uniform || Math.random;
  }
  sample(): number {
    // Box Müller sample
    // Use 1 - uniform to get range to (0-1]
    const u = 1 - this.uniform();
    const v = 1 - this.uniform();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
}

interface Point {
  x: number, y: number;
}

class UI {
  radius: number;
  canvas: HTMLCanvasElement;
  p: Point;
  constructor(simulation: Simulation, canvas: HTMLCanvasElement) {
    console.log(canvas);
    this.radius = 50;
    //
    this.canvas = canvas;
    this.canvas.onmousemove = e => {
      this.p = getPosition(this.canvas, e);
    };
    // sliders
    document.getElementById('G').onchange = function (e) {
      simulation.g = Number((e.target as HTMLInputElement).value);
    };
  }
  render() {
    if (!this.p) return;
      const ctx = this.canvas.getContext('2d');
      ctx.resetTransform();
      ctx.beginPath();
      ctx.arc(this.p.x, this.p.y, this.radius, 0, 2 * Math.PI, false);

      ctx.lineWidth = 1.0;
      ctx.strokeStyle = 'black';
      ctx.stroke();
  }
}

function run() {
  const simulation = new Simulation();
  simulation.add(0, -20, 5.0, 0, 1000.1);
  simulation.add(0, 20, -5.0, 0, 1000.1);
  simulation.g = Number((document.getElementById('G') as HTMLInputElement).value);
  const ui = new UI(simulation, document.getElementById('target') as HTMLCanvasElement);
  function frame(t: number) {
    render("target", simulation);
    ui.render();
    step(simulation, 0.1);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

init().then(run);