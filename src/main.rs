use std::io::Write;
use std::{thread, time};
use std::f64::consts::TAU;
use std::convert::TryFrom;
use std::time::Instant;
use std::env;

mod gravity;
use gravity::{G, Vec2, Simulation, step, THETA};

//use probability::prelude::*;


#[derive(Clone, Copy)]
struct Resolution {
    width: usize,
    height: usize,
}

impl TryFrom<String> for Resolution {
    type Error = ();

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let mut parts = value.split('x');
        let width = parts.next().unwrap().parse().unwrap();
        let height = parts.next().unwrap().parse().unwrap();
        Ok(Resolution{width: width, height: height})
    }
}

struct Frame {
    resolution: Resolution,
    pixels: Vec<u8>,
}

impl Frame {
    fn new(resolution: Resolution) -> Self {
        Frame{resolution: resolution, pixels: vec![0; resolution.width * resolution.height * 4]}
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
        let bpp = 4;
        let stride = frame.resolution.width * bpp;
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

struct Zoom {
    center: Vec2<f32>,
    scale: f32,
    resolution: Resolution,
}

impl Zoom {
    fn to_screen(&self, world: &Vec2<f32>) -> Vec2<f32> {
        Vec2::make(
            0.5 * self.resolution.width as f32 + self.scale * (world.x - self.center.x),
            0.5 * self.resolution.height as f32 + self.scale * (world.y - self.center.y),
        )
    }
}

fn inside(p: (i32, i32), resolution: Resolution) -> bool {
    p.0 >= 0 && p.0 < resolution.width as i32 && p.1 >= 0 && p.1 < resolution.height as i32
}

fn draw(frame: &mut Frame, positions: &Vec<Vec2<f32>>, zoom: &Zoom) {
    for position in positions {
        let screen = zoom.to_screen(position);
        draw_star(frame, (screen.x as i32, screen.y as i32));
    }
}


// computes the velocities needed to maintain orbits
fn orbital_velocity(items: &Vec<(Vec2<f32>, f32)>) -> Vec<Vec2<f32>> {
    // compute total mass and center of mass
    let mut mass = 0.0;
    let mut cm = Vec2::zero();
    for (p, m) in items {
        mass += m;
        cm = cm.add(&p.scale(*m));
    }
    cm = cm.scale(1.0 / mass);

    items.iter().map(|(p, _)| {
        let delta = p.sub(&cm);
        let r = delta.norm2().sqrt();
        let vo = (G * mass / r).sqrt();
        delta.cross().scale(2.0 * vo / r)
    }).collect()
}

/*
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
}*/

fn spiral_galaxy(n: usize, radius: f64) -> Vec<Vec2<f32>> {
    let inner = 0.2 * radius as f32;
    let arms = 31.0;
    let parameters: Vec<(f32, f32)> = (0..n).map(|i| i as f32 / n as f32).map(|t| (arms * t * TAU as f32, inner + t * (radius as f32 - inner))).collect();
    parameters.iter().map(|(a, r)| Vec2::make(r * a.cos(), r * a.sin())).collect()
}

fn add_galaxy(simulation: &mut Simulation, n: usize, center: &Vec2<f32>, velocity: &Vec2<f32>, mass: f32, radius: f64) {
    let stars_fraction = 0.1;  // half of the mass is for stars
    let star_mass = stars_fraction * mass / n as f32;
    let black_hole_mass = mass - stars_fraction * mass; // the rest is for the black hole
    // add stars
    let items: Vec<_> = spiral_galaxy(n, radius).iter().map(|p| (center.add(p), star_mass)).collect();
    //let items: Vec<_> = spiral_galaxy(n, radius).iter().map(|p| (center.add(p), star_mass)).collect();
    // add black hole
    let velocities = orbital_velocity(&items);
    for ((p, m), v) in items.iter().zip(velocities.iter())  {
        simulation.add(p, &v.add(velocity), *m);
    }
    simulation.add(center, &velocity, black_hole_mass);
}

const FPS: f32 = 30.0;
fn main() -> std::result::Result<(), std::io::Error> {
    let tmp = env::var("RESOLUTION").or::<()>(Ok("506x253".to_string())).unwrap();
    let resolution = Resolution::try_from(tmp).unwrap();
    let mut frame = Frame::new(resolution);
    let zoom = Zoom{center: Vec2::zero(), scale: 16.0, resolution: frame.resolution};
    let mut simulation = Simulation::new();

    add_galaxy(&mut simulation, 
        500,
        &Vec2{x: 0.0, y: 0.0},
        &Vec2{x: 0.0, y: 0.1},
        400.0, 10.0);
    /*add_galaxy(&mut simulation,
        800,
        &Vec2{x: 20.0, y: 0.0},
        &Vec2{x: 0.0, y: -0.1},
        400.0, 10.0);*/

    let dt = 1.0 / FPS as f32;
    const STEPS: usize = 10;  // steps per frame
    for _ in 0..5000 {
        let t0 = Instant::now();
        for _ in 0..STEPS {        
            step(&mut simulation, dt / STEPS as f32);
        }
        let duration = t0.elapsed();
        clear(&mut frame);
        draw(&mut frame, simulation.positions(), &zoom);
        std::io::stdout().write_all(&(frame.pixels)).unwrap();
        thread::sleep(time::Duration::from_secs_f32(dt).saturating_sub(duration));
        eprintln!("E={}, physics={}ms", simulation.energy(THETA), duration.as_millis());
    }
    Ok(())
}
