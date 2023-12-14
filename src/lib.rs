
use std::{f64::consts::TAU, iter::zip};

use probability::prelude::*;
use rayon::prelude::*;

const G: f32 = 0.2; // Gravitational constant. (m^2⋅kg^-1⋅s^−2)
const SOFTENING: f32 = 0.05; // Softens hard accelerations. (m)
const THETA: f32 = 2.5; // Threshold value for Barnes-Hut. (m)


#[derive(Clone, Copy, PartialEq)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

impl Vec2<f32> {
    pub fn make(x: f32, y: f32) -> Self {
        Vec2{x: x, y: y}
    }
    pub fn zero() -> Self {
        Vec2{x: 0.0, y: 0.0}
    }
    pub fn add(self, rhs: &Vec2<f32>) -> Self {
        Vec2{x: self.x + rhs.x, y: self.y + rhs.y}
    }
    pub fn sub(self, rhs: &Vec2<f32>) -> Self {
        Vec2{x: self.x - rhs.x, y: self.y - rhs.y}
    }
    pub fn scale(self, rhs: f32) -> Self {
        Vec2{x: self.x * rhs, y: self.y * rhs}
    }
    pub fn norm2(self) -> f32 {
        self.x * self.x + self.y * self.y
    }
    pub fn cross(self) -> Self {
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
        center = center.add(&p.scale(*m));
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
        // return childs - filter out self matches
        self.children.iter().flat_map(|node| node.contributions(p, theta)).filter(|(point, _)| point != p).collect()
    }
}
fn create_tree(items: &Vec<(Vec2<f32>, f32)>) -> Node {
    let positions: Vec<_> = items.iter().map(|(p, _)| *p).collect();
    Node::create(&bbox(&positions), &items)
}

// physics
pub struct State {
    pub positions: Vec<Vec2<f32>>,
    pub velocities: Vec<Vec2<f32>>,
}

// compute the gravitational potential energy between a particle pair
fn potential_energy((pi, mi): (&Vec2<f32>, &f32), (pj, mj): (&Vec2<f32>, &f32)) -> f32 {
    -G * mi * mj / pi.sub(&pj).norm2().sqrt()
}


type Coefficients = Vec<(f32, f32)>;
pub struct Simulation {
    pub state: State,
    pub masses: Vec<f32>,
    pub barnes_hut: bool,
    pub theta: f32,
    pub coefficients: Coefficients, // coefficients for symplectic integrator
}

pub fn euler() -> Coefficients { vec!((1.0, 1.0)) }
pub fn leap2() -> Coefficients { vec!((0.5, 0.0), (0.5, 1.0)) }
pub fn ruth3() -> Coefficients {
    vec!(
        (2.0/3.0, 7.0/24.0),
        (-2.0/3.0, 0.75),
        (1.0, -1.0/24.0),
    )
}
pub fn ruth4() -> Coefficients {
    let c = (2.0 as f32).powf(1.0 / 3.0);
    vec!(
        (0.5 / (2.0 - c), 0.0),
        (0.5 * (1.0 - c) / (2.0 - c), 1.0 / (2.0 - c)),
        (0.5 * (1.0 - c) / (2.0 - c), -c / (2.0 - c)),
        (0.5 / (2.0 - c), 1.0 / (2.0 - c)),
    )
}

impl Simulation {
    pub fn new() -> Self {
        Simulation{
            state: State {
                positions: vec!(),
                velocities: vec!(),
            },
            masses: vec!(),
            barnes_hut: true,
            theta: THETA,
            coefficients: ruth3(),
        }
    }
    pub fn add(&mut self, position: &Vec2<f32>, velocity: &Vec2<f32>, mass: f32) {
        self.state.positions.push(*position);
        self.state.velocities.push(*velocity);
        self.masses.push(mass);
    }
    pub fn remove(&mut self, index: usize) {
        self.state.positions.remove(index);
        self.state.velocities.remove(index);
        self.masses.remove(index);
    }
    pub fn center_of_mass(&self) -> (Vec2<f32>, f32) {
        let mut mass = 0.0;
        let mut cm = Vec2::zero();
        for (p, m) in zip(self.state.positions.iter(), self.masses.iter()) {
            mass += m;
            cm = cm.add(&p.scale(*m));
        }
        cm = cm.scale(1.0 / mass);
        (cm, mass)
    }
    // Generic symplectic step for integrating hamiltonians
    fn symplectic_step(&self, dt: f32, coefficents: &Vec<(f32, f32)>) -> State {
        let mut q = self.state.positions.clone();
        let mut v = self.state.velocities.clone();
        for (c, d) in coefficents {
            v = add_scaled(&v, c * dt, &acceleration(&self.state, &self.masses, self.barnes_hut, self.theta));
            q = add_scaled(&q, d * dt, &v);
        }
        State{positions: q, velocities: v}
    }
    pub fn step(&mut self, dt: f32) {
        // ruth3
        self.state = self.symplectic_step(dt, &self.coefficients);
    }
    pub fn energy(&self) -> f32 {
        let items: Vec<_> = zip(self.state.positions.iter().map(|x| *x), self.masses.iter().map(|x| *x)).collect();
        let kinetic: f32 = zip(self.state.velocities.iter(), self.masses.iter()).map(|(v, m)| m * v.norm2()).sum();

        /*for i in 0..self.state.positions.len() - 1 {
            for j in i+1..self.state.positions.len() {
                potential += -G * self.masses[i] * self.masses[j] / self.state.positions[i].sub(&self.state.positions[j]).norm2().sqrt();
            }
        }*/
        let tree = create_tree(&items);
        let potential: f32 = items.par_iter().map(|(p0, m0)|
            tree.contributions(p0, self.theta).iter().map(|(p1, m1)| potential_energy((p0, m0), (p1, m1))).sum::<f32>()
        ).sum();
 
        kinetic + potential
    }
}

// Computes gravity force acting on (pi, mi) by (pj, mj) and reverse. This force is symetric
fn gravity((pi, mi): (Vec2<f32>, f32), (pj, mj): (Vec2<f32>, f32)) -> Vec2<f32> {
    let delta = pi.sub(&pj);
    let r2 = delta.norm2();
    let f = G * mi * mj / (r2 + SOFTENING*SOFTENING);  // gravity force
    let r = r2.sqrt();
    delta.scale(-f / r)
}

// approximate gravity
fn gravity_barnes_hut(items: &Vec<(Vec2<f32>, f32)>, theta: f32) -> Vec<Vec2<f32>> {
    //let mut forces: Vec<Vec2<f32>> = items.iter().map(|_| Vec2::zero()).collect();
    let tree = create_tree(items);
    items.par_iter().map(|(p0, m0)| {
        let mut force = Vec2::zero();
        for (p1, m1) in tree.contributions(p0, theta) {
            force = force.add(&gravity((*p0, *m0), (p1, m1)));
        }
        force
    }).collect()
}

fn gravity_exact(items: &Vec<(Vec2<f32>, f32)>) -> Vec<Vec2<f32>> {
    let mut forces: Vec<Vec2<f32>> = items.iter().map(|_| Vec2::zero()).collect();
    for i in 0..items.len()-1 {
        for j in i+1..items.len() {
            forces[i] = forces[i].add(&gravity(items[i], items[j]));
            forces[j] = forces[j].add(&gravity(items[j], items[i]));
        }
    }
    forces
}

fn acceleration(state: &State, masses: &Vec<f32>, barnes_hut: bool, theta: f32) -> Vec<Vec2<f32>> {
    let items: Vec<_> = state.positions.iter().map(|x| *x).zip(masses.iter().map(|x| *x)).collect();
    let forces = if barnes_hut {
        gravity_barnes_hut(&items, theta)
    } else {
        gravity_exact(&items)
    };
    forces.iter().zip(masses.iter()).map(|(f, m)| f.scale(1.0 / m)).collect()
}

// computes the velocities needed to maintain orbits
pub fn orbital_velocity((center, mass): &(Vec2<f32>, f32), p: &Vec2<f32>) -> Vec2<f32> {
    if mass == &0.0 {
        return Vec2::zero();
    }
    let delta = p.sub(&center);
    let r = delta.norm2().sqrt();
    let vo = (G * mass / r).sqrt();
    delta.cross().scale(vo / r)
}

fn random_galaxy(n: usize, radius: f64) -> Vec<Vec2<f32>> {
    let mut source = source::default().seed([1337, 1337]);
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

fn spiral_galaxy(n: usize, radius: f64) -> Vec<Vec2<f32>> {
    let inner = 0.2 * radius as f32;
    let arms = 31.0;
    let parameters: Vec<(f32, f32)> = (0..n).map(|i| i as f32 / n as f32).map(|t| (arms * t * TAU as f32, inner + t * (radius as f32 - inner))).collect();
    parameters.iter().map(|(a, r)| Vec2::make(r * a.cos(), r * a.sin())).collect()
}

fn add_galaxy(simulation: &mut Simulation, n: usize, center: &Vec2<f32>, velocity: &Vec2<f32>, mass: f32, radius: f64) {
    let stars_fraction = 0.5;  // half of the mass is for stars
    let star_mass = stars_fraction * mass / n as f32;
    let black_hole_mass = mass - stars_fraction * mass; // the rest is for the black hole
    // add stars
    let items: Vec<_> = random_galaxy(n, radius).iter().map(|p| (center.add(p), star_mass)).collect();
    //let items: Vec<_> = spiral_galaxy(n, radius).iter().map(|p| (center.add(p), star_mass)).collect();
    // add black hole
    let center_of_mass = simulation.center_of_mass();
    for (p, m) in items.iter() {
        let v = orbital_velocity(&center_of_mass, p);
        simulation.add(p, &v.add(velocity), *m);
    }
    simulation.add(center, &velocity, black_hole_mass);
}

