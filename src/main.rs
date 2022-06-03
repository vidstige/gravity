use std::io::Write;
use std::{thread, time};
use std::f64::consts::TAU;
use std::time::Instant;
use probability::prelude::*;

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

fn clear(frame: &mut Frame) {
    for pixel in &mut frame.pixels {
        *pixel = 0;
    }
}

fn draw_pixel(frame: &mut Frame, p: (i32, i32), intensity: u8) {
    if inside(p, frame.resolution) {
        let x = p.0 as usize;
        let y = p.1 as usize;
        let (width, _) = frame.resolution;
        let bpp = 4;
        let stride = width * bpp;
        let index = y * stride + x * bpp;
        frame.pixels[index + 0] = frame.pixels[index + 0].saturating_add(intensity);
        frame.pixels[index + 1] = frame.pixels[index + 1].saturating_add(intensity);
        frame.pixels[index + 2] = frame.pixels[index + 2].saturating_add(intensity);
        frame.pixels[index + 3] = 255;
    }
}

fn draw_star(frame: &mut Frame, p: (i32, i32)) {
    let (x, y) = p;
    draw_pixel(frame, (x, y), 100);
    draw_pixel(frame, (x - 1, y), 50);
    draw_pixel(frame, (x, y - 1), 50);
    draw_pixel(frame, (x + 1, y), 50);
    draw_pixel(frame, (x, y + 1), 50);
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
    fn sub(self, rhs: &Vec2<f32>) -> Self {
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

// quadtree
#[derive(Clone, Copy)]
struct BBox2 {
    top_left: Vec2<f32>,
    bottom_right: Vec2<f32>,
}

impl BBox2 {
    fn new(top_left: Vec2<f32>, bottom_right: Vec2<f32>) -> BBox2 {
        BBox2{top_left: top_left, bottom_right: bottom_right}
    }
    fn top(&self) -> f32 { self.top_left.y }
    fn left(&self) -> f32 { self.top_left.x }
    fn bottom(&self) -> f32 { self.bottom_right.y }
    fn right(&self) -> f32 { self.bottom_right.x }
    
    fn diagonal(&self) -> Vec2<f32> { self.bottom_right.sub(&self.top_left) }

    fn contains(self, p: &Vec2<f32>) -> bool {
        p.x >= self.top_left.x && p.x < self.bottom_right.x &&
        p.y >= self.top_left.y && p.y < self.bottom_right.y
    }
    fn center(&self) -> Vec2<f32> {
        self.top_left.add(&self.bottom_right).scale(0.5)
    }
}

fn bbox(points: &Vec<Vec2<f32>>) -> BBox2 {
    let xs: Vec<_> = points.iter().map(|p| p.x).collect();
    let ys: Vec<_> = points.iter().map(|p| p.y).collect();
    
    BBox2 {
        top_left: Vec2{
            x: xs.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            y: ys.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        },
        bottom_right: Vec2{
            x: xs.iter().fold(-f32::INFINITY, |a, &b| a.max(b)) + std::f32::EPSILON,
            y: ys.iter().fold(-f32::INFINITY, |a, &b| a.max(b)) + std::f32::EPSILON,
        },
    }
}

struct Node {
    bbox: BBox2,
    value: (Vec2<f32>, f32),
    children: Vec<Node>,
}

// returns the four quads of a bounding box
fn quads(bbox: &BBox2) -> [BBox2; 4] {
    let center = bbox.center();
    [
        BBox2::new(bbox.top_left, center),  // top left
        BBox2::new(Vec2{x: center.x, y: bbox.top()}, Vec2{x: bbox.right(), y: center.y}), // top right
        BBox2::new(Vec2{x: bbox.left(), y: center.y}, Vec2{x: center.x, y: bbox.bottom()}), // bottom left
        BBox2::new(center, bbox.bottom_right), // bottom right
    ]
}

fn center_of_mass(items: &Vec<(Vec2<f32>, f32)>) -> (Vec2<f32>, f32) {
    let mut center = Vec2::zero();
    let mut mass = 0.0;
    for (p, m) in items {
        center = center.add(p);
        mass += m;
    }
    (center.scale(1.0 / mass), mass)
}

impl Node {
    fn create(bbox: &BBox2, items: &Vec<(Vec2<f32>, f32)>) -> Node {
        assert_ne!(items.len(), 0, "Can't create empty node");
        if items.len() == 1 {
            return Node{bbox: *bbox, value: items[0], children: vec!()};
        }
        let mut childen = vec!();
        for quad in quads(bbox) {
            let inside: Vec<_> = items.iter().map(|x| *x).filter(|item| quad.contains(&item.0)).collect();
            if inside.len() > 0 {
                childen.push(Node::create(&quad, &inside));
            }
        }
        return Node{bbox: *bbox, value: center_of_mass(items), children: childen};
    }
    // find all contributions for a point and threshold (theta)
    fn contributions(&self, p: &Vec2<f32>, theta: f32) -> Vec<(Vec2<f32>, f32)> {
        let delta = p.sub(&self.value.0);
        let d2 = delta.norm2();
        let diagonal = self.bbox.diagonal();
        let s2 = diagonal.x * diagonal.y;
        // compare squared values to avoid sqrt as well as making it convenient to use both width and height of node
        if self.children.len() == 0 || s2 / d2 < theta * theta {
            return vec!(self.value);
        }
        return self.children.iter().flat_map(|node| node.contributions(p, theta)).collect();
    }
}
fn create_tree(items: &Vec<(Vec2<f32>, f32)>) -> Node {
    let positions: Vec<_> = items.iter().map(|(p, _)| *p).collect();
    Node::create(&bbox(&positions), &items)
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

fn inside(p: (i32, i32), resolution: Resolution) -> bool {
    let (width, height) = resolution;
    p.0 >= 0 && p.0 < width as i32 && p.1 >= 0 && p.1 < height as i32
}

fn draw(frame: &mut Frame, positions: &Vec<Vec2<f32>>, zoom: &Zoom) {
    for position in positions {
        let screen = zoom.to_screen(position);
        draw_star(frame, (screen.x as i32, screen.y as i32));
    }
}

struct State {
    positions: Vec<Vec2<f32>>,
    velocities: Vec<Vec2<f32>>,
}

const G: f32 = 0.2;
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
                potential += -G * self.masses[i] * self.masses[j] / self.state.positions[i].sub(&self.state.positions[j]).norm2().sqrt();
            }
        }
        kinetic + potential
    }
}

// Computes gravity force acting on (pi, mi) by (pj, mj) and reverse. This force is symetric
fn gravity(pi: &Vec2<f32>, pj: &Vec2<f32>, mi: f32, mj: f32) -> (Vec2<f32>, Vec2<f32>) {
    let delta = pi.sub(pj);
    let r2 = delta.norm2();
    // TODO: handle same point interaction better here
    if r2.abs() < std::f32::EPSILON {
        return (Vec2::zero(), Vec2::zero());
    }
    let f = G * mi * mj / r2;  // gravity force
    let r = r2.sqrt();
    (delta.scale(-f / r), delta.scale(f / r))
}

// approximate gravity
fn gravity_barnes_hut(items: &Vec<(Vec2<f32>, f32)>, theta: f32) -> Vec<Vec2<f32>> {
    let mut forces: Vec<Vec2<f32>> = items.iter().map(|_| Vec2::zero()).collect();
    let tree = create_tree(items);

    for (i, (p0, m0)) in items.iter().enumerate() {
        for (p1, m1) in tree.contributions(p0, theta) {
            let (fi, _) = gravity(p0, &p1, *m0, m1);
            forces[i] = forces[i].add(&fi);
        }
    }
    forces
}

fn gravity_direct(items: &Vec<(Vec2<f32>, f32)>) -> Vec<Vec2<f32>> {
    let mut forces: Vec<Vec2<f32>> = items.iter().map(|_| Vec2::zero()).collect();
    for i in 0..items.len()-1 {
        for j in i+1..items.len() {
            let (fi, fj) = gravity(&items[i].0, &items[j].0, items[i].1, items[j].1);
            forces[i] = forces[i].add(&fi);
            forces[j] = forces[j].add(&fj);
        }
    }
    forces
}

fn acceleration(state: &State, masses: &Vec<f32>) -> Vec<Vec2<f32>> {
    let items: Vec<_> = state.positions.iter().map(|x| *x).zip(masses.iter().map(|x| *x)).collect();
    //let forces = gravity_direct(&state, masses);
    let forces = gravity_barnes_hut(&items, 0.5);
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
    let euler = vec!((1.0, 1.0));
    //let leap2 = vec!((0.5, 0.0), (0.5, 1.0));
    /*let ruth3 = vec!(
        (2.0/3.0, 7.0/24.0),
        (-2.0/3.0, 0.75),
        (1.0, -1.0/24.0),
    );*/
    /*let c = (2.0 as f32).powf(1.0 / 3.0);
    let ruth4 = vec!(
        (0.5 / (2.0 - c), 0.0),
        (0.5*(1.0-c)/ (2.0 - c), 1.0/ (2.0 - c)),
        (0.5*(1.0-c)/ (2.0 - c), -c/ (2.0 - c)),
        (0.5/ (2.0 - c), 1.0/ (2.0 - c)),
    );*/
    simulation.state = symplectic_step(simulation, dt, &euler);
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
        let delta = p.sub(&cm);
        let r = delta.norm2().sqrt();
        let vo = (G * mass / r).sqrt();
        delta.cross().scale(vo / r)
    }).collect()
}

fn random_galaxy(n: usize, radius: f64) -> Vec<Vec2<f32>> {
    let mut source = source::default().seed([1, 99]);
    let distribution = Gaussian::new(0.0, radius.sqrt());
    let mut sampler = Independent(&distribution, &mut source);
    let mut positions = vec!();
    for _ in 0..n {
        let position = Vec2::make(
            sampler.next().unwrap() as f32,
            sampler.next().unwrap() as f32,
        );
        if position.norm2().sqrt() > radius as f32 * 0.1 {
            positions.push(position);
        }
        
    }
    positions
}

fn spiral_galaxy(n: usize, radius: f32) -> Vec<Vec2<f32>> {
    let inner = 0.2 * radius;
    let arms = 31.0;
    let parameters: Vec<(f32, f32)> = (0..n).map(|i| i as f32 / n as f32).map(|t| (arms * t * TAU as f32, inner + t * (radius - inner))).collect();
    parameters.iter().map(|(a, r)| Vec2::make(r * a.cos(), r * a.sin())).collect()
}

const FPS: f32 = 30.0;
fn main() -> std::result::Result<(), std::io::Error> {
    let mut frame = Frame::new((506, 253));
    let zoom = Zoom{center: Vec2::zero(), scale: 10.0, resolution: frame.resolution};
    let mut simulation = Simulation::new();

    // add stars
    for p in random_galaxy(2000, 10.0) {
        simulation.add(p, Vec2::zero(), 0.1);
    }
    simulation.state.velocities = oribtal_velocity(&simulation);
    // add black hole
    simulation.add(Vec2::zero(), Vec2::zero(), 200.0);

    let dt = 1.0 / FPS as f32;
    const STEPS: usize = 1;  // steps per frame
    for _ in 0..1000 {
        let t0 = Instant::now();
        for _ in 0..STEPS {        
            step(&mut simulation, dt / STEPS as f32);
        }
        let duration = t0.elapsed();
        clear(&mut frame);
        draw(&mut frame, &simulation.state.positions, &zoom);
        std::io::stdout().write_all(&(frame.pixels)).unwrap();
        thread::sleep(time::Duration::from_secs_f32(dt).saturating_sub(duration));
        //eprintln!("E={}, physics={}ms", simulation.energy(), duration.as_millis());
    }
    Ok(())
}
