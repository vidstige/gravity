
use std::iter::{zip, Sum};

use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
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

impl Sum for Vec2<f32> {
    fn sum<I>(iter: I) -> Self where I: Iterator<Item = Vec2<f32>>
    {
        let mut total = Vec2 { x: 0.0, y: 0.0 };
         for point in iter {
            total.x += point.x;
            total.y += point.y;
        }
        total
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
    value: Option<(Vec2<f32>, f32)>,
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
        if items.is_empty() {
            return Node {bbox: *bbox, value: None, children: Vec::new()};
        }
        if items.len() == 1 {
            return Node{bbox: *bbox, value: Some(items[0]), children: vec!()};
        }
        let mut childen = vec!();
        for quad in quads(bbox) {
            let inside: Vec<_> = items.iter().map(|x| *x).filter(|item| quad.contains(&item.0)).collect();
            if inside.len() > 0 {
                childen.push(Node::create(&quad, &inside));
            }
        }
        return Node{bbox: *bbox, value: Some(center_of_mass(items)), children: childen};
    }
    // find all contributions for a point and threshold (theta)
    fn contributions(&self, p: &Vec2<f32>, theta2: f32) -> Vec<(Vec2<f32>, f32)> {
        if let Some(value) = self.value {
            let delta = p.sub(&value.0);
            let d2 = delta.norm2();
            let diagonal = self.bbox.diagonal();
            let s2 = diagonal.x * diagonal.y;
            // compare squared values to avoid sqrt as well as making it convenient to use both width and height of node
            if self.children.len() == 0 || s2 / d2 < theta2 {
                return vec!(value);
            }
            // return childs - filter out self matches
            self.children.iter().flat_map(|node| node.contributions(p, theta2)).filter(|(point, _)| point != p).collect()
        } else {
            vec![]            
        }
    }
}
fn create_tree(items: &Vec<(Vec2<f32>, f32)>) -> Node {
    let positions: Vec<_> = items.iter().map(|(p, _)| *p).collect();
    Node::create(&bbox(&positions), &items)
}

// physics
#[derive(Clone)]
pub struct State {
    pub positions: Vec<Vec2<f32>>,
    pub velocities: Vec<Vec2<f32>>,
}

// compute the gravitational potential energy between a particle pair
fn potential_energy((pi, mi): (&Vec2<f32>, &f32), (pj, mj): (&Vec2<f32>, &f32), g: f32) -> f32 {
    -g * mi * mj / pi.sub(&pj).norm2().sqrt()
}


type Coefficients = Vec<(f32, f32)>;
pub struct Simulation {
    pub state: State,
    pub masses: Vec<f32>,
    pub g: f32,  // Gravitational constant. (m^2⋅kg^-1⋅s^−2)
    pub softening: f32,  // Softens hard accelerations. (m)
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

fn gravitational_field(g: f32, p0: &Vec2<f32>, p1: &Vec2<f32>, m1: &f32, softening: f32) -> Vec2<f32> {
    let r = p1.sub(p0);
    let d2 = r.norm2();
    let d = d2.sqrt();
    r.scale(1.0 / (d + softening)).scale(-g * m1 / (d2 + softening*softening))
}

impl Simulation {
    pub fn new() -> Self {
        Simulation{
            state: State {
                positions: vec!(),
                velocities: vec!(),
            },
            masses: vec!(),
            g: 0.2,
            softening: 0.05,
            barnes_hut: true,
            theta: 0.5,
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
    pub fn center_of_mass(&self, indices: &Vec<usize>) -> (Vec2<f32>, f32) {
        let mut mass = 0.0;
        let mut cm = Vec2::zero();
        for (p, m) in indices.iter().map(|i| (self.state.positions[*i], self.masses[*i])) {
            mass += m;
            cm = cm.add(&p.scale(m));
        }
        cm = cm.scale(1.0 / mass);
        (cm, mass)
    }
    // Generic symplectic step for integrating hamiltonians
    fn symplectic_step(&self, dt: f32, coefficents: &Vec<(f32, f32)>) -> State {
        let mut state = self.state.clone();
        for (c, d) in coefficents 
        {
            let a = acceleration(&state, &self.masses, self.barnes_hut, self.theta, self.g, self.softening);
            state.velocities = add_scaled(&state.velocities, c * dt, &a);
            state.positions = add_scaled(&state.positions, d * dt, &state.velocities);
        }
        state
    }
    pub fn step(&mut self, dt: f32) {
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
        let theta = self.theta * tree.bbox.diagonal().norm2();
        let potential: f32 = items.par_iter().map(|(p0, m0)|
            tree.contributions(p0, theta).iter().map(|(p1, m1)| potential_energy((p0, m0), (p1, m1), self.g)).sum::<f32>()
        ).sum();
 
        kinetic + potential
    }

    pub fn field(&self, world: &[Vec2<f32>]) -> Vec<Vec2<f32>> {
        let items: Vec<_> = zip(self.state.positions.iter().map(|x| *x), self.masses.iter().map(|x| *x)).collect();
        let tree = create_tree(&items);

        let theta = self.theta * tree.bbox.diagonal().norm2();
        world.iter().map(|p0|
            tree.contributions(p0, theta).iter().map(|(p1, m1)| gravitational_field(self.g, p0, p1, m1, self.softening)).sum()
        ).collect()
    }
}

// Computes gravity force acting on (pi, mi) by (pj, mj) and reverse. This force is symetric
fn gravity((pi, mi): (Vec2<f32>, f32), (pj, mj): (Vec2<f32>, f32), g: f32, softening: f32) -> Vec2<f32> {
    let delta = pi.sub(&pj);
    let r2 = delta.norm2();
    let f = g * mi * mj / (r2 + softening*softening);  // gravity force
    let r = r2.sqrt();
    delta.scale(-f / r)
}

// approximate gravity
fn gravity_barnes_hut(items: &Vec<(Vec2<f32>, f32)>, theta: f32, g: f32, softening: f32) -> Vec<Vec2<f32>> {
    //let mut forces: Vec<Vec2<f32>> = items.iter().map(|_| Vec2::zero()).collect();
    let tree = create_tree(items);
    items.par_iter().map(|(p0, m0)| {
        let mut force = Vec2::zero();
        for (p1, m1) in tree.contributions(p0, theta * tree.bbox.diagonal().norm2()) {
            force = force.add(&gravity((*p0, *m0), (p1, m1), g, softening));
        }
        force
    }).collect()
}

fn gravity_exact(items: &Vec<(Vec2<f32>, f32)>, g: f32, softening: f32) -> Vec<Vec2<f32>> {
    let mut forces: Vec<Vec2<f32>> = items.iter().map(|_| Vec2::zero()).collect();
    for i in 0..items.len()-1 {
        for j in i+1..items.len() {
            forces[i] = forces[i].add(&gravity(items[i], items[j], g, softening));
            forces[j] = forces[j].add(&gravity(items[j], items[i], g, softening));
        }
    }
    forces
}

fn acceleration(state: &State, masses: &Vec<f32>, barnes_hut: bool, theta: f32, g: f32, softening: f32) -> Vec<Vec2<f32>> {
    let items: Vec<_> = state.positions.iter().map(|x| *x).zip(masses.iter().map(|x| *x)).collect();
    let forces = if barnes_hut {
        gravity_barnes_hut(&items, theta, g, softening)
    } else {
        gravity_exact(&items, g, softening)
    };
    forces.iter().zip(masses.iter()).map(|(f, m)| f.scale(1.0 / m)).collect()
}

// computes the velocities needed to maintain orbits
pub fn orbital_velocity((center, mass): &(Vec2<f32>, f32), p: &Vec2<f32>, g: f32) -> Vec2<f32> {
    if mass == &0.0 {
        return Vec2::zero();
    }
    let delta = p.sub(&center);
    let r = delta.norm2().sqrt();
    let vo = (g * mass / r).sqrt();
    delta.cross().scale(vo / r)
}
