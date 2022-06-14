//pub const SOFTENING: f32 = 0.15; 
//pub const THETA: f32 = 2.5; 

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

// Bounding box

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

// Barnes-Hut
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
struct State {
    positions: Vec<Vec2<f32>>,
    velocities: Vec<Vec2<f32>>,
}

// compute the gravitational potential energy between a particle pair
fn potential_energy((pi, mi): (&Vec2<f32>, &f32), (pj, mj): (&Vec2<f32>, &f32), G: f32) -> f32 {
    -G * mi * mj / pi.sub(&pj).norm2().sqrt()
}

pub struct Simulation {
    pub G: f32,         // Gravitational "constant" (m^3⋅kg^-1⋅s^−2) 
    pub theta: f32,     // Threshold value for Barnes-Hut. ()
    pub softening: f32, // Softens hard accelerations. (m)
    state: State,
    masses: Vec<f32>,
}
impl Simulation {
    pub fn new() -> Self {
        Simulation{
            G: 6.674e-11,
            theta: 0.0,
            softening: 0.15,
            state: State {
                positions: vec!(),
                velocities: vec!(),
            },
            masses: vec!(),
        }
    }
    pub fn add(&mut self, position: &Vec2<f32>, velocity: &Vec2<f32>, mass: f32) {
        self.state.positions.push(*position);
        self.state.velocities.push(*velocity);
        self.masses.push(mass);
    }
    pub fn positions(&self) -> &Vec<Vec2<f32>> {
        &self.state.positions
    }
    pub fn energy(&self) -> f32 {
        let items: Vec<_> = self.state.positions.iter().map(|x| *x).zip(self.masses.iter().map(|x| *x)).collect();
        let kinetic: f32 = items.iter().map(|(v, m)| m * v.norm2()).sum();

        /*for i in 0..self.state.positions.len() - 1 {
            for j in i+1..self.state.positions.len() {
                potential += -G * self.masses[i] * self.masses[j] / self.state.positions[i].sub(&self.state.positions[j]).norm2().sqrt();
            }
        }*/
        let tree = create_tree(&items);
        let potential: f32 = items.iter().map(|(p0, m0)|
            tree.contributions(p0, self.theta).iter().map(|(p1, m1)| potential_energy((p0, m0), (p1, m1), self.G)).sum::<f32>()
        ).sum();
 
        kinetic + potential
    }
}

// Computes gravity force acting on (pi, mi) by (pj, mj) and reverse. This force is symetric
fn gravity((pi, mi): (Vec2<f32>, f32), (pj, mj): (Vec2<f32>, f32), G: f32, softening: f32) -> Vec2<f32> {
    let delta = pi.sub(&pj);
    let r2 = delta.norm2();
    let f = G * mi * mj / (r2 + softening*softening);  // gravity force
    let r = r2.sqrt();
    delta.scale(-f / r)
}

// approximate gravity
fn gravity_barnes_hut(items: &Vec<(Vec2<f32>, f32)>, G: f32, softening: f32, theta: f32) -> Vec<Vec2<f32>> {
    //let mut forces: Vec<Vec2<f32>> = items.iter().map(|_| Vec2::zero()).collect();
    let tree = create_tree(items);

    items.iter().map(|(p0, m0)| {
        let mut force = Vec2::zero();
        for (p1, m1) in tree.contributions(p0, theta) {
            force = force.add(&gravity((*p0, *m0), (p1, m1), G, softening));
        }
        force
    }).collect()
}

fn gravity_direct(items: &Vec<(Vec2<f32>, f32)>, G: f32, softening: f32) -> Vec<Vec2<f32>> {
    let mut forces: Vec<Vec2<f32>> = items.iter().map(|_| Vec2::zero()).collect();
    for i in 0..items.len()-1 {
        for j in i+1..items.len() {
            forces[i] = forces[i].add(&gravity(items[i], items[j], G, softening));
            forces[j] = forces[j].add(&gravity(items[j], items[i], G, softening));
        }
    }
    forces
}

fn acceleration(state: &State, masses: &Vec<f32>, G: f32, softening: f32, theta: f32) -> Vec<Vec2<f32>> {
    let items: Vec<_> = state.positions.iter().map(|x| *x).zip(masses.iter().map(|x| *x)).collect();
    //let forces = gravity_direct(&items);
    let forces = gravity_barnes_hut(&items, G, softening, theta);
    forces.iter().zip(masses.iter()).map(|(f, m)| f.scale(1.0 / m)).collect()
}

fn add_scaled(first: &Vec<Vec2<f32>>, k: f32, second: &Vec<Vec2<f32>>) -> Vec<Vec2<f32>> {
    first.iter().zip(second.iter()).map(|(a, b)| a.add(&b.scale(k))).collect()
}

// Generic symplectic step for integrating hamiltonians
fn symplectic_step(simulation: &Simulation, dt: f32, coefficents: &Vec<(f32, f32)>) -> State {
    let mut q = simulation.state.positions.clone();
    let mut v = simulation.state.velocities.clone();
    for (c, d) in coefficents {
        v = add_scaled(&v, c * dt, &acceleration(&simulation.state, &simulation.masses, simulation.G, simulation.softening, simulation.theta));
        q = add_scaled(&q, d * dt, &v);
    }
    State{positions: q, velocities: v}
}

pub fn step(simulation: &mut Simulation, dt: f32) {
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
