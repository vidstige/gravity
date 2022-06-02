use std::io::Write;
use std::{thread, time};
use std::collections::VecDeque;
use  std::f64::consts::TAU;

type Resolution = (usize, usize);

struct Frame {
    resolution: Resolution,
    pixels: Vec<u8>,
}

impl Frame {
    fn new((width, height): Resolution) -> Self {
        Frame{resolution: (width, height), pixels: vec![0; width * height * 4]}
    }
}
struct Color {
    r: u8, g: u8, b: u8, a: u8,
}
const WHITE: Color = Color {r: 255, g: 255, b: 255, a: 255};

fn clear(frame: &mut Frame) {
    for pixel in &mut frame.pixels {
        *pixel = 0;
    }
}

fn draw_pixel(frame: &mut Frame, x: usize, y: usize, color: Color) {
    let (width, _) = frame.resolution;
    let bpp = 4;
    let stride = width * bpp;
    let index = y * stride + x * bpp;
    frame.pixels[index + 0] = color.r;
    frame.pixels[index + 1] = color.g;
    frame.pixels[index + 2] = color.b;
    frame.pixels[index + 3] = color.a;
}


#[derive(Clone, Copy)]
struct Vec2<T> {
    x: T,
    y: T,
}

impl Vec2<f32> {
    fn make(x: f32, y: f32) -> Self {
        Vec2{x: x, y: y}
    }
    fn zero() -> Self {
        Vec2{x: 0.0, y: 0.0}
    }
    fn add(self, rhs: &Vec2<f32>) -> Self {
        Vec2{x: self.x + rhs.x, y: self.y + rhs.y}
    }
    fn sub(self, rhs: Vec2<f32>) -> Self {
        Vec2{x: self.x - rhs.x, y: self.y - rhs.y}
    }
    fn scale(self, rhs: f32) -> Self {
        Vec2{x: self.x * rhs, y: self.y * rhs}
    }
    fn norm2(self) -> f32 {
        self.x * self.x + self.y * self.y
    }
    fn cross(self) -> Self {
        Vec2{x: -self.y, y: self.x}
    }
}

fn add_scaled(first: &Vec<Vec2<f32>>, k: f32, second: &Vec<Vec2<f32>>) -> Vec<Vec2<f32>> {
    first.iter().zip(second.iter()).map(|(a, b)| a.add(&b.scale(k))).collect()
}

struct Zoom {
    center: Vec2<f32>,
    scale: f32,
    resolution: Resolution,
}

impl Zoom {
    fn to_screen(&self, world: &Vec2<f32>) -> Vec2<f32> {
        let (width, height) = self.resolution;
        Vec2::make(
            0.5 * width as f32 + self.scale * (world.x - self.center.x),
            0.5 * height as f32 + self.scale * (world.y - self.center.y),
        )
    }
}

fn inside(p: Vec2<f32>, resolution: Resolution) -> bool {
    let (width, height) = resolution;
    p.x >= 0.0 && p.x < width as f32 && p.y >= 0.0 && p.y < height as f32
}

type Trail = VecDeque<Vec2<f32>>;
fn add_points(trails: &mut Vec<Trail>, points: &Vec<Vec2<f32>>, length: usize) {
    for i in 0..points.len() {
        trails[i].push_back(points[i]);
        while trails[i].len() > length {
            trails[i].pop_front();
        }
    }
}

fn draw(frame: &mut Frame, trails: &Vec<Trail>, zoom: &Zoom) {
    for trail in trails {
        for p in trail {
            let screen = zoom.to_screen(p);
            if inside(screen, frame.resolution) {
                draw_pixel(frame, screen.x as usize, screen.y as usize, WHITE);
            }
        }
    }
}

struct State {
    positions: Vec<Vec2<f32>>,
    velocities: Vec<Vec2<f32>>,
}
impl State {
    fn len(&self) -> usize {
        self.positions.len()
    }
}

const G: f32 = 0.01;
struct Simulation {
    state: State,
    masses: Vec<f32>,
}
impl Simulation {
    fn new() -> Self {
        Simulation{
            state: State {
                positions: vec!(),
                velocities: vec!(),
            },
            masses: vec!(),
        }
    }
    fn add(&mut self, position: Vec2<f32>, velocity: Vec2<f32>, mass: f32) {
        self.state.positions.push(position);
        self.state.velocities.push(velocity);
        self.masses.push(mass);
    }
    fn energy(&self) -> f32 {
        let kinetic: f32 = self.state.velocities.iter().zip(self.masses.iter()).map(|(v, m)| m * v.norm2()).sum();
        let mut potential = 0.0;
        for i in 0..self.state.positions.len() - 1 {
            for j in i+1..self.state.positions.len() {
                potential += -G * self.masses[i] * self.masses[j] / self.state.positions[i].sub(self.state.positions[j]).norm2().sqrt();
            }
        }
        kinetic + potential
    }
}

fn gravity(state: &State, m: &Vec<f32>) -> Vec<Vec2<f32>> {
    let mut forces: Vec<Vec2<f32>> = state.positions.iter().map(|_| Vec2::zero()).collect();
    for i in 0..state.len()-1 {
        for j in i+1..state.len() {
            let pi = state.positions[i];
            let pj = state.positions[j];
            let delta = pi.sub(pj);
            let r2 = delta.norm2();
            let f = G * m[i] * m[j] / r2;  // gravity force
            let r = r2.sqrt();
            forces[i] = forces[i].add(&delta.scale(-f / r));
            forces[j] = forces[j].add(&delta.scale(f / r));
        }
    }
    forces
}

fn acceleration(state: &State, masses: &Vec<f32>) -> Vec<Vec2<f32>> {
    let forces = gravity(&state, masses);
    forces.iter().zip(masses.iter()).map(|(f, m)| f.scale(1.0 / m)).collect()
}

// Generic symplectic step for integrating hamiltonians
fn symplectic_step(simulation: &Simulation, dt: f32, coefficents: &Vec<(f32, f32)>) -> State {
    let mut q = simulation.state.positions.clone();
    let mut v = simulation.state.velocities.clone();
    for (c, d) in coefficents {
        v = add_scaled(&v, c * dt, &acceleration(&simulation.state, &simulation.masses));
        q = add_scaled(&q, d * dt, &v);
    }
    State{positions: q, velocities: v}
}

fn step(simulation: &mut Simulation, dt: f32) {
    //let euler = vec!(Vec2::make(1.0, 1.0));
    //let leap2 = vec!((0.5, 0.0), (0.5, 1.0));
    let ruth3 = vec!(
        (2.0/3.0, 7.0/24.0),
        (-2.0/3.0, 0.75),
        (1.0, -1.0/24.0),
    );
    /*let c = (2.0 as f32).powf(1.0 / 3.0);
    let ruth4 = vec!(
        (0.5 / (2.0 - c), 0.0),
        (0.5*(1.0-c)/ (2.0 - c), 1.0/ (2.0 - c)),
        (0.5*(1.0-c)/ (2.0 - c), -c/ (2.0 - c)),
        (0.5/ (2.0 - c), 1.0/ (2.0 - c)),
    );*/
    simulation.state = symplectic_step(simulation, dt, &ruth3);
}

// computes the velocities needed to maintain orbits
fn oribtal_velocity(simulation: &Simulation) -> Vec<Vec2<f32>> {
    // compute total mass and center of mass
    let mut mass = 0.0;
    let mut cm = Vec2::zero();
    for (p, m) in simulation.state.positions.iter().zip(simulation.masses.iter()) {
        mass += m;
        cm = cm.add(&p.scale(*m));
    }
    cm = cm.scale(1.0 / mass);

    simulation.state.positions.iter().map(|p| {
        let delta = p.sub(cm);
        let r = delta.norm2().sqrt();
        let vo = (G * mass / r).sqrt();
        delta.cross().scale(vo / r)
    }).collect()
}

fn galaxy(n: usize, radius: f32) -> Vec<Vec2<f32>> {
    /*let mut source = source::default().seed([1, 99]);
    let distribution = Gaussian::new(0.0, 1.0);
    let mut sampler = Independent(&distribution, &mut source);
    let mut positions = vec!();
    for _ in 0..n {
        let position = Vec2::make(
            sampler.next().unwrap() as f32,
            sampler.next().unwrap() as f32,
        );
        positions.push((position, 0.1));
    }
    positions*/
    let inner = 0.2 * radius;
    let arms = 15.0;
    let parameters: Vec<(f32, f32)> = (0..n).map(|i| i as f32 / n as f32).map(|t| (arms * t * TAU as f32, inner + t * (radius - inner))).collect();
    parameters.iter().map(|(a, r)| Vec2::make(r * a.cos(), r * a.sin())).collect()
}

const FPS: f32 = 30.0;
fn main() -> std::result::Result<(), std::io::Error> {
    let mut frame = Frame::new((506, 253));
    let zoom = Zoom{center: Vec2::zero(), scale: 25.0, resolution: frame.resolution};
    let mut simulation = Simulation::new();

    // add stars
    for p in galaxy(200, 5.0) {
        simulation.add(p, Vec2::zero(), 0.1);
    }
    simulation.state.velocities = oribtal_velocity(&simulation);
    // add black hole
    //simulation.add(Vec2::zero(), Vec2::zero(), 10.0);
    

    let dt = 1.0 / FPS as f32;
    let mut trails = simulation.masses.iter().map(|_| VecDeque::new()).collect();
    loop {
        step(&mut simulation, dt);
        add_points(&mut trails, &simulation.state.positions, 10);
        clear(&mut frame);
        draw(&mut frame, &trails, &zoom);
        std::io::stdout().write_all(&(frame.pixels)).unwrap();
        thread::sleep(time::Duration::from_secs_f32(dt));
        eprintln!("E={}", simulation.energy());
    }
}
